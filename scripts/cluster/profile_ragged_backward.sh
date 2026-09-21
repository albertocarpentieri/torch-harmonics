#!/usr/bin/env bash
# What the ragged BACKWARD's time goes on, and specifically what the dk/dv scatter's
# atomics cost.
#
# profile_ragged_kernel.sh asks this of the forward. The backward is the larger half
# and has a cost the forward does not: dk/dv are scatter-accumulated with one atomic
# add per channel per neighbour, into rows that overlapping disks contend for. At
# nside 64, 1536 channels, batch*time 2 that is 2 * 2 * 49152 * 410.6 * 1536, about
# 1.24e11 atomic adds per layer.
#
# The question is whether that is where the time goes, because the proposed fix is a
# rewrite. The neighbourhood is exactly symmetric -- measured, zero one-directional
# entries -- so the scatter could become a gather over the same arcs with no atomics
# at all and no transposed precompute. Worth doing only if the atomics are actually
# expensive, and the history here is against guessing: the forward's audit blamed the
# arc gather pattern, the pixel order and the dtype in turn, and each was worth a few
# percent.
#
# Note the counters to read are the RED ones, not ATOM. atomicAdd whose return value
# is unused compiles to a reduction, which is a different pipe.
#
#   sbatch sbatch_scripts/profile_ragged_backward.sh
#   sbatch --export=ALL,LEVEL=6 sbatch_scripts/profile_ragged_backward.sh
#
#SBATCH --job-name=profile-ragged-bwd
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/acarpentieri/healda_project/logs/profile_ragged_bwd_%j.out
#SBATCH --error=/home/acarpentieri/healda_project/logs/profile_ragged_bwd_%j.err

set -uo pipefail

PROJECT="/home/acarpentieri/healda_project"
CONTAINER="${CONTAINER:-${PROJECT}/containers/healpix_container_e2g.sqsh}"
WORKDIR="${WORKDIR:-torch-harmonics-hpx}"
LEVEL="${LEVEL:-6}"
DTYPE="${DTYPE:-bf16}"

[[ -f "${CONTAINER}" ]] || { echo "ERROR: missing ${CONTAINER}" >&2; exit 2; }
mkdir -p "${PROJECT}/logs"

cat > "${PROJECT}/logs/_one_bwd.py" <<'PY'
"""One forward+backward of the ragged kernel, nothing else, so the profile is clean."""
import math
import os

import torch

from torch_harmonics import HealpixGrid, NeighborhoodAttentionS2

level = int(os.environ.get("LEVEL", "6"))
dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("DTYPE", "bf16")]
channels, heads, bt = 1536, 16, 2
npix = 12 * 4**level
grid = HealpixGrid(nside=2**level)
attn = NeighborhoodAttentionS2(
    grid_in=grid, grid_out=grid, in_channels=channels, out_channels=channels,
    num_heads=heads, theta_cutoff=math.radians(10.46),
).cuda()
print(f"optimized_kernel={attn.optimized_kernel} npix={npix} dtype={dtype}", flush=True)

x = torch.randn(bt, channels, npix, device="cuda", requires_grad=True)


def step():
    with torch.autocast(device_type="cuda", dtype=dtype):
        attn(x).sum().backward()
    x.grad = None


for _ in range(3):          # warm up outside the capture range
    step()
torch.cuda.synchronize()
# torch.cuda.profiler.start/stop, not cudart().profilerStart: that spelling does not
# exist in torch 2.14 -- the cudart binding is cudaProfilerStart. profile_ragged_kernel.sh
# still has the old spelling and would fail the same way.
torch.cuda.profiler.start()
step()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
PY

run_in_container() {
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre,"${PROJECT}":/workspace \
    --container-workdir="/workspace/${WORKDIR}" \
    env "PYTHONPATH=/workspace/${WORKDIR}" "LEVEL=${LEVEL}" "DTYPE=${DTYPE}" "$@"
}

echo "=== job ${SLURM_JOB_ID:-?} | ragged backward profile | level ${LEVEL} ${DTYPE} ==="
echo

echo "--- is ncu present and permitted? ---"
run_in_container bash -lc 'command -v ncu >/dev/null && ncu --version | head -2 || echo "ncu ABSENT"'
echo

OUT="/workspace/logs/ncu_ragged_bwd_level${LEVEL}_${DTYPE}"

# The RED counters are the point of this run. SpeedOfLight and MemoryWorkloadAnalysis
# place the kernel on the roofline; the explicit metrics separate the reduction pipe
# from ordinary stores, which is what the gather rewrite would remove.
run_in_container bash -lc "
set -uo pipefail
if ! command -v ncu >/dev/null; then
  echo '--- ncu unavailable; torch profiler fallback ---'
  python -u /workspace/logs/_one_bwd.py
  python - <<'PYF'
import math, os, torch
from torch.profiler import ProfilerActivity, profile
from torch_harmonics import HealpixGrid, NeighborhoodAttentionS2
level = int(os.environ.get('LEVEL', '6'))
dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[os.environ.get('DTYPE', 'bf16')]
channels, heads, bt = 1536, 16, 2
npix = 12 * 4**level
grid = HealpixGrid(nside=2**level)
attn = NeighborhoodAttentionS2(grid_in=grid, grid_out=grid, in_channels=channels,
    out_channels=channels, num_heads=heads, theta_cutoff=math.radians(10.46)).cuda()
x = torch.randn(bt, channels, npix, device='cuda', requires_grad=True)
def step():
    with torch.autocast(device_type='cuda', dtype=dtype):
        attn(x).sum().backward()
    x.grad = None
for _ in range(3): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10): step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
PYF
  exit 0
fi

echo '--- sections: roofline, memory, occupancy, stalls ---'
ncu --target-processes all --profile-from-start off \
    --kernel-name-base function --kernel-name 'regex:s2_attn_bwd_ragged' \
    --section SpeedOfLight --section Occupancy \
    --section WarpStateStats --section MemoryWorkloadAnalysis \
    --csv --log-file '${OUT}_sections.csv' \
    python -u /workspace/logs/_one_bwd.py 2>&1 | tail -25

echo
echo '--- explicit reduction/atomic counters ---'
ncu --target-processes all --profile-from-start off \
    --kernel-name-base function --kernel-name 'regex:s2_attn_bwd_ragged' \
    --metrics \
lts__t_sectors_op_red.sum,\
lts__t_sectors_op_atom.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_red.sum,\
lts__average_t_sector_op_read_hit_rate.pct,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum \
    --csv --log-file '${OUT}_atomics.csv' \
    python -u /workspace/logs/_one_bwd.py 2>&1 | tail -25

echo
echo '--- instruction mix: is the SM busy with arithmetic or with bookkeeping? ---'
# 77% SM throughput means the busiest sub-pipe is 77% busy, not that 77% of peak
# FLOPs are being issued. If the fma pipe is idle while alu and lsu are saturated,
# the kernel is spending itself on addresses and arc walking, and the lever is
# per-neighbour overhead rather than arithmetic.
ncu --target-processes all --profile-from-start off \
    --kernel-name-base function --kernel-name 'regex:s2_attn_bwd_ragged' \
    --metrics \
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_integer_pred_on.sum,\
sm__inst_executed.sum,\
smsp__warps_launched.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    --csv --log-file '${OUT}_instmix.csv' \
    python -u /workspace/logs/_one_bwd.py 2>&1 | tail -20

echo
echo '--- instmix csv ---'; head -30 '${OUT}_instmix.csv' 2>/dev/null
echo
echo '--- sections csv ---'; head -40 '${OUT}_sections.csv' 2>/dev/null
echo
echo '--- atomics csv ---';  head -40 '${OUT}_atomics.csv' 2>/dev/null
"
echo
echo "=== done. Read the RED sector count against the arithmetic: if the reduction"
echo "=== pipe is not near saturation, the gather rewrite is not where the time is."
