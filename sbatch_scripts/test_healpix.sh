#!/usr/bin/env bash
# Run the torch-harmonics HEALPix test suite in the HEALPix container.
#
# Submitted rather than run interactively so it survives the slurm controller being
# intermittent: the job queues and runs whenever the controller comes back.
#
# By default runs the foundation suites -- the grid descriptor, the truncation
# policy, the descriptor cache and the arc-segment neighbourhood precompute -- which
# are CPU-only. Override with TESTS to run something else:
#
#   TESTS="tests/" sbatch sbatch_scripts/test_healpix.sh
#
#SBATCH --job-name=test-healpix
#SBATCH --account=coreai_climate_earth2
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/acarpentieri/healda_project/logs/test_healpix_%j.out
#SBATCH --error=/home/acarpentieri/healda_project/logs/test_healpix_%j.err

set -euo pipefail

PROJECT="/home/acarpentieri/healda_project"
CONTAINER="/lustre/fsw/portfolios/coreai/users/acarpentieri/healda_project/containers/healpix_container.sqsh"

TESTS="${TESTS:-tests/test_grid_assumptions.py tests/test_truncation.py tests/test_cache.py tests/test_neighborhood.py}"
PYTEST_ARGS="${PYTEST_ARGS:--q --no-header -rf --durations=10}"

# Which repo the paths in TESTS are relative to. Defaults to the torch-harmonics
# checkout; set WORKDIR=healda to run the healda suite (e.g. the attention bridge):
#
#   WORKDIR=healda TESTS="tests/unit/test_spherical_attention.py" sbatch sbatch_scripts/test_healpix.sh
#
WORKDIR="${WORKDIR:-torch-harmonics-hpx}"

[[ -f "${CONTAINER}" ]] || { echo "ERROR: missing ${CONTAINER}" >&2; exit 2; }

mkdir -p "${PROJECT}/logs"

# Echoed so a log identifies its own run: TESTS and WORKDIR arrive through the
# environment, and a job whose variables failed to propagate otherwise looks exactly
# like a deliberate run of the defaults.
echo "=== job ${SLURM_JOB_ID:-?} | workdir=${WORKDIR} | tests=${TESTS} ==="

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre,"${PROJECT}":/workspace \
  --container-workdir="/workspace/${WORKDIR}" \
  python -m pytest ${PYTEST_ARGS} ${TESTS}
