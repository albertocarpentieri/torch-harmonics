#!/bin/bash
#
# Rebuild the ragged attention extension, prove it still computes the right thing,
# then time the new register-blocked kernel against the old one.
#
# The order matters and is the whole point of having this as one job. The new kernel
# leaves all but the last of its NLOC accumulator registers unguarded, which is sound
# only if NLOC is exactly DIV_UP(nchan, 32); get that wrong and it reads past the
# channel block and still produces a plausible, smooth field. So the correctness
# suite is a gate, not a report: if tests/test_attention_ragged.py fails, the job
# stops and no timing is printed, because a wrong kernel's speed is not information.
#
# The A/B is run in one job for the same reason: same GPU, same clocks, same build,
# so the only difference between the two numbers is which kernel ran.
#
# Sweeping the two tunables needs no source edit -- torch passes NVCC_APPEND_FLAGS
# through to nvcc:
#
#   sbatch --export=ALL,NVCC_APPEND_FLAGS="-DTH_ATTENTION_RAGGED_THREADS=128 -DTH_ATTENTION_RAGGED_NB=8" \
#     sbatch_scripts/rebuild_and_validate_ragged.sh
#
# gpu:4 rather than gpu:1 because the QOS rejects a single-GPU request with
# QOSMinGRES; the work is single-process and only touches cuda:0.
#SBATCH --job-name=ragged-rebuild-validate
#SBATCH --account=coreai_climate_earth2
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
# The rebuild dominates: the special kernel is templated over NLOC 1..16 for each of
# three dtypes, so nvcc has ~48 new instantiations to chew through.
#SBATCH --time=02:30:00
#SBATCH --output=/home/acarpentieri/healda_project/logs/ragged_rebuild_validate_%j.out
#SBATCH --error=/home/acarpentieri/healda_project/logs/ragged_rebuild_validate_%j.err

set -euo pipefail

PROJECT="/home/acarpentieri/healda_project"
CONTAINER="/lustre/fsw/portfolios/coreai/users/acarpentieri/healda_project/containers/healpix_container.sqsh"
WORKDIR="${WORKDIR:-torch-harmonics-hpx}"

# Benchmark shape: healda's dit-5B runs 96 channels per head at nside 64 with a 10
# degree radius, so that is the configuration whose time we actually care about.
# nside 32 is kept because the arithmetic ceiling differs between the two -- at 64
# the exact-work FLOPs alone need ~95% of vector peak to match FlexAttention, at 32
# only ~35%, so the two resolutions answer different questions.
TIMING_NSIDES="${TIMING_NSIDES:-32 64}"
CHANNELS="${CHANNELS:-96}"
NUM_HEADS="${NUM_HEADS:-1}"
CUTOFF_DEG="${CUTOFF_DEG:-10.0}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"

[[ -f "${CONTAINER}" ]] || { echo "ERROR: missing ${CONTAINER}" >&2; exit 2; }
mkdir -p "${PROJECT}/logs"

run_in_container() {
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre,"${PROJECT}":/workspace \
    --container-workdir="/workspace/${WORKDIR}" \
    "$@"
}

echo "=== job ${SLURM_JOB_ID:-?} | rebuild + validate ragged attention ==="
echo "    workdir=${WORKDIR}  nvcc_append=${NVCC_APPEND_FLAGS:-none}"
echo

# ---------------------------------------------------------------------------
# 1. Rebuild in place.
#
# The .so lives in the mounted source tree, not in the image, so building in place
# is what makes the edit take effect. TORCH_CUDA_ARCH_LIST is left to the image's
# default: setup.py keys its sm90a/sm100a paths off it, and overriding it here would
# silently change which of those get compiled.
# ---------------------------------------------------------------------------
echo "--- rebuilding extension ---"
run_in_container bash -lc '
set -euo pipefail
echo "arch list: ${TORCH_CUDA_ARCH_LIST:-<image default>}"
echo "nvcc append: ${NVCC_APPEND_FLAGS:-<none>}"

# From scratch. An incremental build here is not worth the risk: the .so in the
# tree was produced by the image build, under a configuration we do not control,
# and a half-matching object graph is exactly the failure that looks like a code
# bug for an hour.
rm -rf build

# Parallelism via MAX_JOBS, NOT setup.py -j.
#
# Torch drives these extensions through ninja, and MAX_JOBS is what ninja reads,
# so it parallelizes the compiles within an extension -- which is where the time
# goes, since the special kernels are templated over 16 NLOC values per dtype.
#
# setup.py -j instead builds the four extensions concurrently, and that races:
# job 3787269 died with "cannot find disco_helpers.o" because the disco_helpers
# link step ran before its own compile had produced the object. The log showed
# two interleaved ninja pools, [12/17] and [6/7], which is the tell.
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"
echo "MAX_JOBS=${MAX_JOBS}"
python setup.py build_ext --inplace
'
echo

# ---------------------------------------------------------------------------
# 2. Correctness gate.
#
# test_attention_ragged.py checks the fp32 kernel against an independent dense
# masked-softmax oracle to atol 1e-5, forward and backward, which is exactly the
# property the register blocking could break. set -e makes a failure end the job.
# ---------------------------------------------------------------------------
#
# Forward and backward have independent switches, so all four combinations are run.
# That is what makes a failure bisectable without a rebuild: "both generic" is the
# pre-existing code and must pass or the harness itself is suspect, and the two mixed
# rows attribute any failure to one kernel. Run before the timing and in this order,
# cheapest suspicion first.
#
# -x so the first failure stops that combination, but each combination is its own
# invocation, so a forward failure does not hide a backward one.
gate() {
  local label="$1" fwd="$2" bwd="$3"
  echo "--- gate: ${label}  (fwd=$([[ ${fwd} == 1 ]] && echo generic || echo special), bwd=$([[ ${bwd} == 1 ]] && echo generic || echo special)) ---"
  run_in_container env "TORCH_HARMONICS_RAGGED_GENERIC=${fwd}" "TORCH_HARMONICS_RAGGED_BWD_GENERIC=${bwd}" \
    python -u -m pytest tests/test_attention_ragged.py -x -q
  echo
}

echo "=== correctness gates, against the dense masked-softmax oracle ==="
echo
gate "baseline, neither kernel new" 1 1
gate "new forward only" 0 1
gate "new backward only" 1 0
gate "both, the shipping configuration" 0 0

# ---------------------------------------------------------------------------
# 3. A/B timing. Only reached if both gates passed.
# ---------------------------------------------------------------------------
bench() {
  run_in_container env "TORCH_HARMONICS_RAGGED_GENERIC=$1" "TORCH_HARMONICS_RAGGED_BWD_GENERIC=$2" \
    python -u -m benchmarks.healpix_ragged_report \
      --nsides ${TIMING_NSIDES} \
      --timing-nsides ${TIMING_NSIDES} \
      --dtypes float32 \
      --batch 1 \
      --channels "${CHANNELS}" \
      --num-heads "${NUM_HEADS}" \
      --cutoff-deg "${CUTOFF_DEG}" \
      --iters "${ITERS}" \
      --warmup "${WARMUP}" \
      --skip-reference \
      --skip-equiangular
}

# Four arms again, for the same reason as the gates but the other way round: the
# report gives a fwd row and a fwd+bwd row, and only by attributing each change to
# one kernel can the backward's contribution be read off at all, since it is
# reported as a difference of the two rows.
echo "--- BEFORE: both generic (accumulator in shared, one neighbour in flight) ---"
bench 1 1
echo

echo "--- new forward only ---"
bench 0 1
echo

echo "--- new backward only ---"
bench 1 0
echo

echo "--- AFTER: both, the shipping configuration ---"
bench 0 0
echo

echo "=== done."
echo "=== Reading it: the forward was 9.3 ms of the 33.3 ms fwd+bwd at level 5, so the"
echo "=== fwd row and the fwd+bwd row should move by different factors, and the"
echo "=== backward arm is the one that decides whether the total is competitive."
echo "=== Still untouched in the backward, and the largest remaining item: it"
echo "=== traverses every neighbourhood twice, and scatters dk/dv with a global"
echo "=== atomicAdd per channel per neighbour into rows that adjacent output points"
echo "=== contend for, since their disks overlap by 74%."
