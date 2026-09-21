#!/bin/bash
#
# Same work as rebuild_and_validate_ragged.sh -- rebuild, gate on correctness, then
# A/B the timing -- but driven by enroot directly instead of srun, so it does not
# need the Slurm controller.
#
# It exists because the controller has been unreachable for a day (scontrol ping
# fails with "family = 0, port = 0", so it is broken rather than busy) while the
# register-blocked kernels sit unvalidated. Run this from any shell that can see a
# GPU; the login node cannot, so it will refuse there rather than produce a
# confusing failure deep inside the test suite.
#
# Usage:
#   bash sbatch_scripts/run_ragged_validate_enroot.sh
#
# Sweeping the tunables, same as the sbatch path:
#   NVCC_APPEND_FLAGS="-DTH_ATTENTION_RAGGED_THREADS=128" bash sbatch_scripts/run_ragged_validate_enroot.sh
#
# SKIP_BUILD=1 reuses the .so already in the tree, for when only the timing is
# wanted and nothing has changed since the last build.

set -uo pipefail

PROJECT="/home/acarpentieri/healda_project"
CONTAINER="${CONTAINER:-${PROJECT}/containers/healpix_container.sqsh}"
WORKDIR="${WORKDIR:-torch-harmonics-hpx}"

TIMING_NSIDES="${TIMING_NSIDES:-32 64}"
CHANNELS="${CHANNELS:-96}"
NUM_HEADS="${NUM_HEADS:-1}"
CUTOFF_DEG="${CUTOFF_DEG:-10.0}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"
JOBS="${JOBS:-16}"

[[ -f "${CONTAINER}" ]] || { echo "ERROR: missing ${CONTAINER}" >&2; exit 2; }

# Fail here, loudly, rather than 40 minutes into a rebuild. Every gate below needs a
# CUDA device: the kernels are CUDA-only and the oracle they are checked against runs
# on the same device.
if ! nvidia-smi -L > /dev/null 2>&1; then
  echo "ERROR: no GPU visible from $(hostname)." >&2
  echo "       The correctness gates need one -- the ragged kernels are CUDA-only." >&2
  echo "       Get onto a GPU node and re-run, or use the sbatch path once the" >&2
  echo "       Slurm controller is answering again." >&2
  echo >&2
  echo "       What you CAN do here without a GPU:" >&2
  echo "         bash sbatch_scripts/compile_check_ragged.sh   # compiles + register report" >&2
  exit 3
fi

mkdir -p "${PROJECT}/logs"

echo "=== ragged attention: rebuild, validate, A/B ==="
echo "    host=$(hostname)  workdir=${WORKDIR}"
echo "    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "    nvcc_append=${NVCC_APPEND_FLAGS:-none}"
echo

in_container() {
  enroot start \
    --mount "${PROJECT}:/workspace" \
    --env "NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-}" \
    "${CONTAINER}" bash -lc "cd /workspace/${WORKDIR} && $1"
}

# ---------------------------------------------------------------------------
# 1. Rebuild in place. The .so lives in the mounted source tree, not the image,
#    so this is what makes an edit take effect.
# ---------------------------------------------------------------------------
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "--- rebuilding extension (this is the slow part) ---"
  # MAX_JOBS, not setup.py -j: see rebuild_and_validate_ragged.sh. -j builds the
  # four extensions concurrently and their link steps race their own compiles.
  # rm -rf build so the object graph cannot half-match the image's prebuilt .so.
  in_container "rm -rf build && MAX_JOBS=${JOBS} python setup.py build_ext --inplace" || {
    echo "ERROR: build failed; not running the gates." >&2; exit 4; }
  echo
else
  echo "--- SKIP_BUILD=1, using the .so already in the tree ---"
  echo
fi

# ---------------------------------------------------------------------------
# 2. Correctness gates, over the switch combinations.
#
#    Forward and backward have independent switches, so a failure attributes to
#    one kernel with no rebuild. "neither new" must pass or the harness itself is
#    suspect. A failure here stops the script: the speed of a wrong kernel is not
#    information.
#
#    The backward carries a third switch, TORCH_HARMONICS_RAGGED_BWD_TWO_PASS,
#    which is orthogonal to the other two: it picks the formulation (one walk of
#    each neighbourhood or two) rather than the kernel. The one-walk form is exact
#    in exact arithmetic but regroups fp32 sums, so a gradient that drifts has to
#    be attributable to the formulation or to the register blocking, separately.
#    Sweeping it against both backward kernels is what makes that possible.
# ---------------------------------------------------------------------------
gate() {
  local label="$1" fwd="$2" bwd="$3" two="${4:-0}"
  echo "--- gate: ${label} ---"
  in_container "TORCH_HARMONICS_RAGGED_GENERIC=${fwd} TORCH_HARMONICS_RAGGED_BWD_GENERIC=${bwd} \
    TORCH_HARMONICS_RAGGED_BWD_TWO_PASS=${two} \
    python -u -m pytest tests/test_attention_ragged.py -x -q" || {
    echo >&2
    echo "GATE FAILED: ${label} (fwd_generic=${fwd}, bwd_generic=${bwd}, bwd_two_pass=${two})." >&2
    echo "  fwd_generic=1,bwd_generic=1,bwd_two_pass=1 passing but this failing isolates the change." >&2
    exit 5
  }
  echo
}

echo "=== correctness gates, against the dense masked-softmax oracle ==="
echo
gate "baseline, neither kernel new, two-pass backward" 1 1 1
gate "new forward only, two-pass backward" 0 1 1
gate "two-pass backward, register blocked" 1 0 1
gate "single-pass backward, generic" 1 1 0
gate "new forward only" 0 1
gate "new backward only" 1 0
gate "both, the shipping configuration" 0 0

# ---------------------------------------------------------------------------
# 3. A/B timing. Only reached if every gate passed.
# ---------------------------------------------------------------------------
bench() {
  in_container "TORCH_HARMONICS_RAGGED_GENERIC=$1 TORCH_HARMONICS_RAGGED_BWD_GENERIC=$2 \
    TORCH_HARMONICS_RAGGED_BWD_TWO_PASS=${3:-0} \
    python -u -m benchmarks.healpix_ragged_report \
      --nsides ${TIMING_NSIDES} --timing-nsides ${TIMING_NSIDES} \
      --dtypes float32 --batch 1 --channels ${CHANNELS} --num-heads ${NUM_HEADS} \
      --cutoff-deg ${CUTOFF_DEG} --iters ${ITERS} --warmup ${WARMUP} \
      --skip-reference --skip-equiangular"
}

echo "=== A/B timing ==="
echo
echo "--- BEFORE: both generic, two-pass backward ---";      bench 1 1 1; echo
echo "--- new forward only, two-pass backward ---";           bench 0 1 1; echo
echo "--- register-blocked backward, two passes ---";         bench 1 0 1; echo
echo "--- register-blocked backward, one pass ---";           bench 1 0 0; echo
echo "--- AFTER: both, shipping configuration ---";           bench 0 0 0; echo

echo "=== done."
echo "=== Reading it: the forward was 9.3 ms of the 33.3 ms fwd+bwd at level 5, so"
echo "=== the fwd row and the fwd+bwd row move by different factors, and the"
echo "=== backward arm decides whether the total is competitive with FlexAttention."
