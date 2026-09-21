#!/usr/bin/env bash
# FlexDiskSpatialAttention (MR 56) against NeighborhoodSpatialAttention
# (ac/nnja-harmonics-latlon), run from the bench-disk worktree, which is MR 56 with
# the torch-harmonics layer file laid on top so both classes live in one healda.
#
# The healpix image carries earth2grid, torch-harmonics with its CUDA kernels, and
# healda editable from /workspace/healda -- see setup_healpix_container.sh. That
# editable install registers a meta-path finder that beats PYTHONPATH, so the
# worktree is installed over it at job start; container writes are ephemeral, so it
# affects this job only.
#
#   sbatch sbatch_scripts/bench_disk_attention.sh
#   sbatch --export=ALL,EXTRA_ARGS="--skip timing --radius-deg 12" \
#          sbatch_scripts/bench_disk_attention.sh
#   sbatch --export=ALL,SCRIPT="benchmarks/disk_attention_blocksize_sweep.py" \
#          sbatch_scripts/bench_disk_attention.sh
#
# gpu:4 rather than gpu:1 because the QOS rejects a single-GPU request with
# QOSMinGRES; the benchmark is single-process and only touches cuda:0.
#SBATCH --job-name=bench-disk-attn
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Per-node, not job-scoped: the cli_filter here rejects --gpus/-G.
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/acarpentieri/healda_project/logs/bench_disk_attention_%j.out
#SBATCH --error=/home/acarpentieri/healda_project/logs/bench_disk_attention_%j.err

set -euo pipefail

PROJECT="/home/acarpentieri/healda_project"
CONTAINER="${CONTAINER:-${PROJECT}/containers/healpix_container.sqsh}"
WORKTREE="${WORKTREE:-bench-disk}"
SCRIPT="${SCRIPT:-benchmarks/disk_attention_shootout.py}"
# Most probes are python; the profiler wrapper is a shell script, since ncu has to
# be the parent process of the interpreter rather than the other way round.
RUNNER="${RUNNER:-python -u}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

[[ -f "${CONTAINER}" ]] || { echo "ERROR: missing ${CONTAINER}" >&2; exit 2; }
[[ -d "${PROJECT}/${WORKTREE}" ]] || { echo "ERROR: missing worktree ${PROJECT}/${WORKTREE}" >&2; exit 2; }

mkdir -p "${PROJECT}/logs"

echo "=== job ${SLURM_JOB_ID:-?} | disk attention shootout | worktree=${WORKTREE} ==="
echo ">> git rev: $(git -C "${PROJECT}/${WORKTREE}" rev-parse --short HEAD) on $(git -C "${PROJECT}/${WORKTREE}" branch --show-current)"
echo ">> extra args: ${EXTRA_ARGS:-(none)}"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre,"${PROJECT}":/workspace \
  --container-workdir="/workspace/${WORKTREE}" \
  bash -c "
set -euo pipefail
python -m pip install -q --no-deps --no-build-isolation -e /workspace/${WORKTREE}
python -c 'import healda, torch_harmonics, earth2grid; print(\"healda from\", healda.__file__)'
${RUNNER} ${SCRIPT} ${EXTRA_ARGS}
"
