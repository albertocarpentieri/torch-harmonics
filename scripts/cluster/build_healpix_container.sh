#!/usr/bin/env bash
# Clone the new fcn3_project training container and add the repos healda needs,
# plus the torch-harmonics HEALPix working tree, into a dedicated image.
#
# The fcn3 image already carries torch, the CUDA toolchain and the makani/FCN3
# stack; what it lacks for healda is earth2grid (git-only), earth2studio and
# physicsnemo. torch-harmonics is installed from the local HEALPix branch rather
# than from PyPI so that the compiled DISCO/attention extensions match the
# checkout we develop against.
#
#SBATCH --job-name=build-healpix-img
#SBATCH --account=coreai_climate_earth2
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/home/acarpentieri/healda_project/logs/build_healpix_container_%j.out
#SBATCH --error=/home/acarpentieri/healda_project/logs/build_healpix_container_%j.err

set -euo pipefail

PROJECT="/home/acarpentieri/healda_project"
CONTAINERS="/lustre/fsw/portfolios/coreai/users/acarpentieri/healda_project/containers"
BASE="/lustre/fsw/portfolios/coreai/users/acarpentieri/fcn3_project/containers/ngc_pytorch_26.08_train.sqsh"
FINAL="${CONTAINERS}/healpix_container.sqsh"
OUTPUT="${FINAL}.build-${SLURM_JOB_ID:-manual}"

[[ -f "${BASE}" ]] || { echo "ERROR: missing base image ${BASE}" >&2; exit 2; }
[[ -d "${PROJECT}/torch-harmonics-hpx" ]] || { echo "ERROR: missing ${PROJECT}/torch-harmonics-hpx" >&2; exit 2; }
[[ -d "${PROJECT}/healda" ]] || { echo "ERROR: missing ${PROJECT}/healda" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "ERROR: build image already exists: ${OUTPUT}" >&2; exit 2; }

[[ -f "${PROJECT}/sbatch_scripts/setup_healpix_container.sh" ]] || { echo "ERROR: missing setup_healpix_container.sh" >&2; exit 2; }

mkdir -p "${PROJECT}/logs" "${CONTAINERS}"
trap 'rm -f "${OUTPUT}"' EXIT

# The provisioning steps live in setup_healpix_container.sh rather than inline here:
# nesting python snippets inside a quoted `bash -lc` string strips their quotes.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --container-image="${BASE}" \
  --container-save="${OUTPUT}" \
  --container-mounts=/lustre:/lustre,"${PROJECT}":/workspace \
  bash /workspace/sbatch_scripts/setup_healpix_container.sh

mv "${OUTPUT}" "${FINAL}"
trap - EXIT

echo "Built image: ${FINAL}"
