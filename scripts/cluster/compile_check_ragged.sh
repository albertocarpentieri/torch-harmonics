#!/bin/bash
#
# Compile the ragged attention kernels without a GPU and without the scheduler.
#
# enroot can start the training image on the login node once the nvidia hook is
# skipped, and compiling needs the toolkit but not the driver. That is the only
# reason this exists: it is what let the register-blocking work be checked at all
# while the Slurm controller was unreachable, and it is much faster than a queue
# round trip afterwards.
#
# Both architectures are required, not one: the backward's atomic epilogue is split
# on __CUDA_ARCH__ < 900, so sm_90a and sm_100a each leave half of it uncompiled.
#
# Note -Wall is not an nvcc flag. Passing it produces "nvcc fatal", which does not
# contain the word "error" and so slips past a grep for one -- hence checking the
# return code and grepping for "fatal" as well.

set -uo pipefail

PROJECT="/home/acarpentieri/healda_project"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/acarpentieri/healda_project/containers/healpix_container.sqsh}"
OUT="${OUT:-${PROJECT}/logs/verify}"
ARCHES="${ARCHES:-sm_90a sm_100a}"
FILES="${FILES:-attention_cuda_fwd_ragged attention_cuda_bwd_ragged}"

mkdir -p "${OUT}"
rm -f "${OUT}"/*.log "${OUT}"/*.o

NVIDIA_VISIBLE_DEVICES=void enroot start \
  --mount "${PROJECT}:/workspace" \
  --env "ARCHES=${ARCHES}" --env "FILES=${FILES}" \
  "${CONTAINER}" bash -lc '
set -uo pipefail
cd /workspace/torch-harmonics-hpx
OUT=/workspace/logs/verify
D="torch_harmonics/attention/optimized/kernels_cuda"

INC=$(python -c "from torch.utils.cpp_extension import include_paths; print(\" \".join(\"-I\"+p for p in include_paths(\"cuda\")))" 2>/dev/null)
PYINC=$(python -c "import sysconfig; print(\"-I\"+sysconfig.get_paths()[\"include\"])")

fail=0
printf "%-30s %-9s %5s %5s %5s %6s\n" file arch rc err warn fatal
for ARCH in ${ARCHES}; do
  for F in ${FILES}; do
    L="${OUT}/${F}_${ARCH}.log"
    nvcc -c "${D}/${F}.cu" -o "${OUT}/${F}_${ARCH}.o" \
      -std=c++20 -O3 -DNDEBUG -arch=${ARCH} -Xptxas=-v \
      --expt-relaxed-constexpr --expt-extended-lambda \
      -I"${D}" ${INC} ${PYINC} \
      -DTORCH_EXTENSION_NAME=_C -DTORCH_API_INCLUDE_EXTENSION_H > "${L}" 2>&1
    rc=$?
    err=$(grep -ciE "error" "${L}")
    warn=$(grep -ciE "warning" "${L}")
    fatal=$(grep -ciE "fatal" "${L}")
    printf "%-30s %-9s %5s %5s %5s %6s\n" "${F}" "${ARCH}" "${rc}" "${err}" "${warn}" "${fatal}"
    [ "${rc}" -ne 0 ] && fail=1
    [ "${err}" -ne 0 ] && fail=1
    [ "${warn}" -ne 0 ] && fail=1
    [ "${fatal}" -ne 0 ] && fail=1
  done
done

echo
for ARCH in ${ARCHES}; do
  for F in ${FILES}; do
    echo "### ${F} ${ARCH}"
    python benchmarks/ptxas_register_report.py "${OUT}/${F}_${ARCH}.log" 2>/dev/null \
      | grep -E "^kernel|^-----|NLOC= 3|generic|SPILLING" | head -40
    echo
  done
done

echo "COMPILE_CHECK_FAIL=${fail}"
' 2>&1 | grep -v cargo
