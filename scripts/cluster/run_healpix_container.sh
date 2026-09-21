#!/usr/bin/env bash
# Run a command inside the HEALPix container.
#
#   sbatch_scripts/run_healpix_container.sh pytest tests/test_grid_assumptions.py
#   GPU=1 sbatch_scripts/run_healpix_container.sh pytest tests/test_attention.py
#
# Defaults to a CPU node; set GPU=1 for a single-GPU allocation. The working
# directory inside the container is the torch-harmonics checkout.

set -euo pipefail

PROJECT="/home/acarpentieri/healda_project"
CONTAINER="${CONTAINER:-${PROJECT}/containers/healpix_container.sqsh}"
WORKDIR="${WORKDIR:-/workspace/torch-harmonics-hpx}"

[[ -f "${CONTAINER}" ]] || {
  echo "ERROR: missing ${CONTAINER}; run sbatch_scripts/build_healpix_container.sh first" >&2
  exit 2
}

if [[ "${GPU:-0}" == "1" ]]; then
  # Four GPUs, not one: the QOS rejects a single-GPU request with QOSMinGRES. The
  # work is single-process and only ever touches cuda:0.
  alloc=(-p batch -N1 --gpus-per-node "${GPUS:-4}")
else
  # cpu-normal is the QOS build_container.sh uses for the cpu partition here.
  alloc=(-p cpu -q cpu-normal -N1 --cpus-per-task "${CPUS:-16}")
fi

exec srun -A coreai_devtech_all "${alloc[@]}" \
  --time "${TIME:-01:00:00}" \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre,"${PROJECT}":/workspace \
  --container-workdir="${WORKDIR}" \
  bash -lc "$(printf '%q ' "$@")"
