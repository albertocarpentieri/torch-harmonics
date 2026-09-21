#!/usr/bin/env bash
# grid_vs_flex.py run once per ragged kernel configuration, against one flex arm.
#
# The §8 table compares "ours" against flex, but "ours" was always the shipping
# configuration. This varies it. All the switches used here are runtime, so the
# whole sweep is one job and one build -- the build-time tunables
# (TH_ATTENTION_RAGGED_NB and friends) would need a rebuild per point and are
# deliberately not swept here.
#
# Flex is re-measured in every arm rather than measured once. That is not waste:
# it is the only check that the machine and the harness are steady across the
# sweep, and a flex column that drifts between arms invalidates the comparison
# that the whole table exists to make.
#
#   sbatch sbatch_scripts/bench_flex_kernel_sweep.sh
#   sbatch --export=ALL,GRIDS="hpx:5",RADII="2.0 10.46" sbatch_scripts/bench_flex_kernel_sweep.sh
#
#SBATCH --job-name=flex-kernel-sweep
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Per-node, not job-scoped: the cli_filter here rejects --gpus/-G.
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=03:30:00
#SBATCH --output=/home/acarpentieri/healda_project/logs/flex_kernel_sweep_%j.out
#SBATCH --error=/home/acarpentieri/healda_project/logs/flex_kernel_sweep_%j.err

set -euo pipefail

PROJECT="/home/acarpentieri/healda_project"
# The earth2grid image, not the plain one: grid_vs_flex.py needs a NEST grid to
# build flex's mask, and without it the comparison is not fair enough to report.
CONTAINER="${CONTAINER:-${PROJECT}/containers/healpix_container_e2g.sqsh}"
WORKDIR="${WORKDIR:-bench-disk}"

GRIDS="${GRIDS:-hpx:5 hpx:6}"
RADII="${RADII:-2.0 3.0 10.46}"
# Which kernel configurations to run. The wide grid/radius table wants only
# "shipping"; the kernel-variant question wants all five but over fewer points,
# because the cost is arms x grids x radii and both cannot be wide at once.
ARMS="${ARMS:-baseline fwd bwd shipping onepass_bf16}"
CHANNELS="${CHANNELS:-1536}"
HEADS="${HEADS:-16}"
BT="${BT:-2}"
BLOCK="${BLOCK:-128}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"
# bf16 is what training uses and is where the ragged backward takes its two-pass
# path; fp32 takes the single-pass one. The two are not the same kernel, so the
# comparison against flex is a different question in each.
DTYPE="${DTYPE:-bf16}"
# Every (grid, radius) is a new shape and each burns one recompile against a
# default budget of 8, so a twelve-row sweep silently drops both arms to eager
# partway through. Sized to the sweep, not left at the default.
RECOMPILE_LIMIT="${RECOMPILE_LIMIT:-64}"

[[ -f "${CONTAINER}" ]] || { echo "ERROR: missing ${CONTAINER}; run derive_earth2grid_container.sh first" >&2; exit 2; }
[[ -d "${PROJECT}/${WORKDIR}" ]] || { echo "ERROR: missing ${PROJECT}/${WORKDIR}" >&2; exit 2; }

mkdir -p "${PROJECT}/logs"

run_in_container() {
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre,"${PROJECT}":/workspace \
    --container-workdir="/workspace/${WORKDIR}" \
    env "PYTHONPATH=/workspace/torch-harmonics-hpx" "$@"
}

echo "=== job ${SLURM_JOB_ID:-?} | flex vs ragged kernel configurations ==="
echo "    grids=${GRIDS}  radii=${RADII}  channels=${CHANNELS} heads=${HEADS} bt=${BT}"
echo "    dtype=${DTYPE}  recompile_limit=${RECOMPILE_LIMIT}  arms=${ARMS}"
echo ">> harmonics rev: $(git -C "${PROJECT}/torch-harmonics-hpx" rev-parse --short HEAD)"
echo ">> bench rev:     $(git -C "${PROJECT}/${WORKDIR}" rev-parse --short HEAD)"
echo

# Establish that the kernels are present and come from the mounted tree before
# measuring anything. A run that silently fell back would still produce a table.
echo "--- provenance ---"
run_in_container python -c '
import sys
import torch
import torch_harmonics as th
import earth2grid.healpix as healpix

print("torch_harmonics:", th.__file__)
if not th.__file__.startswith("/workspace/"):
    sys.exit("FATAL: torch_harmonics is not the mounted tree")
for name in ("forward_ragged", "backward_ragged"):
    dump = torch._C._dispatch_dump(f"attention_kernels::{name}")
    keys = {ln.split(":")[0].strip() for ln in dump.splitlines() if ":" in ln}
    # Bound to a name rather than inlined: a backslash-escaped quote inside an
    # f-string expression is a SyntaxError.
    has_cuda = "CUDA" in keys
    print(f"  {name}: CUDA={has_cuda}")
    if not has_cuda:
        sys.exit(f"FATAL: {name} has no CUDA kernel")
print("  earth2grid NEST level 5 pixels:", healpix.Grid(5, pixel_order=healpix.NEST).lat.size)
print("  device:", torch.cuda.get_device_name(0))
'
echo

want() { [[ " ${ARMS} " == *" $1 "* ]]; }

arm() {
  local key="$1" label="$2" fwd="$3" bwd="$4" two_pass="$5"
  want "${key}" || return 0

  echo "================================================================"
  echo "ARM: ${label}"
  echo "  RAGGED_GENERIC=${fwd}  BWD_GENERIC=${bwd}  BWD_TWO_PASS=${two_pass}"
  echo "================================================================"

  local extra=()
  [[ "${two_pass}" != "default" ]] && extra+=("TORCH_HARMONICS_RAGGED_BWD_TWO_PASS=${two_pass}")

  run_in_container env \
    "TORCH_HARMONICS_RAGGED_GENERIC=${fwd}" \
    "TORCH_HARMONICS_RAGGED_BWD_GENERIC=${bwd}" \
    "${extra[@]}" \
    python -u benchmarks/grid_vs_flex.py \
      --grids ${GRIDS} \
      --radius-deg ${RADII} \
      --channels "${CHANNELS}" \
      --heads "${HEADS}" \
      --bt "${BT}" \
      --block-size "${BLOCK}" \
      --iters "${ITERS}" \
      --warmup "${WARMUP}" \
      --dtype "${DTYPE}" \
      --recompile-limit "${RECOMPILE_LIMIT}"
  echo
}

# Ordered so each arm isolates one change from the one above it.
arm baseline "baseline: neither kernel new"        1 1 default
arm fwd      "new forward only"                    0 1 default
arm bwd      "new backward only"                   1 0 default
arm shipping "shipping: both new (bf16 -> 2-pass)" 0 0 default

# The §6 question, asked directly. 3ce0d4e routes bf16 to two passes because the
# collapse is not numerically safe there, and this benchmark runs bf16 -- so the
# arm above never exercises the single-pass backward at all.
#
# This arm is a SPEED PROBE ONLY. Its gradients are known to exceed the test
# suite's 3e-2 tolerance in bf16; job 3835420 failed dk and dq while dv, the one
# gradient not involving `integral`, passed. It says what the collapse would buy
# if the accuracy problem were solved by an fp32 output copy -- nothing more.
arm onepass_bf16 "single-pass at bf16 (SPEED ONLY, not numerically valid)" 0 0 0

echo "=== done."
echo "=== Reading the columns: 'harm' is the eager harmonics layer, 'harmC' the"
echo "=== compiled one, and the ratio uses harmC because flex_attention is only"
echo "=== ever compiled -- uncompiled it falls back to a dense implementation, which"
echo "=== is what tried to allocate 4608 GiB at level 7. harm is there so the"
echo "=== compile benefit stays visible; do not compare harm against flex."
echo "==="
echo "=== Before believing any row: the tiles column must read 824 at level 5."
echo "=== A different number means flex was handed the wrong pixel ordering and"
echo "=== was doing more work than it should, which understates it everywhere."
