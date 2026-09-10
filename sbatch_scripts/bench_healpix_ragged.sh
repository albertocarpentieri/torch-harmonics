#!/bin/bash
#
# What the ragged HEALPix neighbourhood costs: pattern memory, and the arc kernel
# against the torch reference. Answers the two questions the correctness suite does
# not -- whether raggedness costs memory (it does, structurally, by npoints/nrings)
# and whether it costs speed (unknown until measured).
#
# Overrides come from the environment:
#
#   sbatch --export=ALL,NSIDES="64",DTYPES="bfloat16" sbatch_scripts/bench_healpix_ragged.sh
#
# gpu:4 rather than gpu:1 for the same reason as test_healpix_gpu.sh: the QOS
# enforces a minimum GRES count and rejects a single-GPU request with QOSMinGRES.
# The benchmark is single-process and only ever touches cuda:0.
#SBATCH --job-name=bench-healpix-ragged
#SBATCH --account=coreai_climate_earth2
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
# The torch reference is the slow half and the equiangular arm doubles the
# configurations; 30 minutes was not enough to finish even one table.
#SBATCH --time=01:30:00
#SBATCH --output=/home/acarpentieri/healda_project/logs/bench_healpix_ragged_%j.out
#SBATCH --error=/home/acarpentieri/healda_project/logs/bench_healpix_ragged_%j.err

set -euo pipefail

PROJECT="/home/acarpentieri/healda_project"
CONTAINER="/lustre/fsw/portfolios/coreai/users/acarpentieri/healda_project/containers/healpix_container.sqsh"

WORKDIR="${WORKDIR:-torch-harmonics-hpx}"

# nside 64 is healda's working resolution; the smaller two are there so the
# npoints/nrings ~ 3*nside scaling is visible rather than asserted.
NSIDES="${NSIDES:-16 32 64}"
TIMING_NSIDES="${TIMING_NSIDES:-16 32 64}"
DTYPES="${DTYPES:-float32 bfloat16}"
BATCH="${BATCH:-1}"
CHANNELS="${CHANNELS:-64}"
NUM_HEADS="${NUM_HEADS:-1}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"

[[ -f "${CONTAINER}" ]] || { echo "ERROR: missing ${CONTAINER}" >&2; exit 2; }

mkdir -p "${PROJECT}/logs"

echo "=== job ${SLURM_JOB_ID:-?} | bench | workdir=${WORKDIR} | nsides=${NSIDES} | dtypes=${DTYPES} ==="

# The same registration check the test job prints. A benchmark that silently fell
# back to the reference would report a speedup of 1.0 and look like a real result,
# so establish that the kernel is actually present before trusting any number.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre,"${PROJECT}":/workspace \
  --container-workdir="/workspace/${WORKDIR}" \
  python -c '
import torch

# Importing torch_harmonics is what loads the compiled extension and populates the
# attention_kernels namespace. Without it torch.ops.attention_kernels is empty and
# every op below reports "unavailable" on a build that in fact has them all -- a
# false negative that cost a benchmark run.
import torch_harmonics  # noqa: F401

print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
for name in ("forward_ragged", "backward_ragged"):
    qualname = f"attention_kernels::{name}"
    try:
        getattr(torch.ops.attention_kernels, name)
        keys = torch._C._dispatch_dump(qualname)
        has_cuda = "CUDA" in {ln.split(":")[0].strip() for ln in keys.splitlines() if ":" in ln}
    except Exception as exc:
        print(f"{qualname}: unavailable ({exc})")
        continue
    print(f"{qualname}: CUDA={has_cuda}")
'

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre,"${PROJECT}":/workspace \
  --container-workdir="/workspace/${WORKDIR}" \
  python -u -m benchmarks.healpix_ragged_report \
    --nsides ${NSIDES} \
    --timing-nsides ${TIMING_NSIDES} \
    --dtypes ${DTYPES} \
    --batch "${BATCH}" \
    --channels "${CHANNELS}" \
    --num-heads "${NUM_HEADS}" \
    --iters "${ITERS}" \
    --warmup "${WARMUP}"
