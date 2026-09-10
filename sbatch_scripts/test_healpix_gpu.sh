#!/bin/bash
#
# GPU counterpart of test_healpix.sh.
#
# The CPU script cannot exercise the compiled kernels: the tests that touch them
# are guarded on torch.cuda.is_available(), so on the cpu partition they report as
# skipped and a broken kernel looks exactly like a green run. This one asks for a
# single GPU so those tests actually execute.
#
# Same interface as test_healpix.sh -- WORKDIR and TESTS come from the environment:
#
#   sbatch --export=ALL,TESTS="tests/test_attention_ragged.py" sbatch_scripts/test_healpix_gpu.sh
#
# gpu:4 rather than gpu:1 because the QOS enforces a minimum GRES count and a
# single-GPU request is rejected outright with QOSMinGRES. The suite is a
# single-process pytest run that only ever touches cuda:0, so ntasks stays 1 and the
# remaining three GPUs go unused -- the allocation shape is the cluster's policy, not
# a requirement of the tests.
#SBATCH --job-name=test-healpix-gpu
#SBATCH --account=coreai_climate_earth2
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/acarpentieri/healda_project/logs/test_healpix_gpu_%j.out
#SBATCH --error=/home/acarpentieri/healda_project/logs/test_healpix_gpu_%j.err

set -euo pipefail

PROJECT="/home/acarpentieri/healda_project"
CONTAINER="/lustre/fsw/portfolios/coreai/users/acarpentieri/healda_project/containers/healpix_container.sqsh"

TESTS="${TESTS:-tests/test_attention_ragged.py tests/test_neighborhood.py}"
PYTEST_ARGS="${PYTEST_ARGS:--q --no-header -rf -rs --durations=10}"

WORKDIR="${WORKDIR:-torch-harmonics-hpx}"

[[ -f "${CONTAINER}" ]] || { echo "ERROR: missing ${CONTAINER}" >&2; exit 2; }

mkdir -p "${PROJECT}/logs"

echo "=== job ${SLURM_JOB_ID:-?} | gpu | workdir=${WORKDIR} | tests=${TESTS} ==="

# Reported before pytest runs, because "0 failed" on a node where the op never got
# compiled is the failure mode this script exists to rule out, and it is invisible
# from the test summary alone.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre,"${PROJECT}":/workspace \
  --container-workdir="/workspace/${WORKDIR}" \
  python -c '
import torch
from torch_harmonics.attention.optimized.attention_optimized import _op_is_declared

print("cuda available :", torch.cuda.is_available())
print("device         :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
print("forward_ragged :", _op_is_declared("forward_ragged"))
print()

# Which backends each op is actually registered for. `forward` is the reference
# point: it has had a CUDA kernel since long before this branch, so comparing the
# two separates "the new .cu did not make it into the build" from "this image has
# no attention CUDA kernels at all". Those need opposite fixes, and the declared
# flag above cannot tell them apart -- m.def and the CUDA registration live in the
# same extension but not in the same source file.
for name in ("forward", "backward", "forward_ragged", "backward_ragged"):
    qualname = f"attention_kernels::{name}"
    try:
        getattr(torch.ops.attention_kernels, name)
    except (AttributeError, RuntimeError) as exc:
        print(f"{qualname}: NOT DECLARED ({exc})")
        continue
    try:
        dump = torch._C._dispatch_dump(qualname)
    except Exception as exc:
        print(f"{qualname}: dump unavailable ({exc})")
        continue
    keys = sorted({ln.split(":")[0].strip() for ln in dump.splitlines() if ":" in ln})
    has_cuda = "CUDA" in keys
    has_cpu = "CPU" in keys
    noise = ("Autograd", "Autocast", "FuncTorch")
    interesting = [k for k in keys if not k.startswith(noise)]
    print(f"{qualname}: CUDA={has_cuda}  CPU={has_cpu}")
    print(f"    keys: {interesting}")
'

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre,"${PROJECT}":/workspace \
  --container-workdir="/workspace/${WORKDIR}" \
  python -m pytest ${PYTEST_ARGS} ${TESTS}
