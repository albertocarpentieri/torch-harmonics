#!/usr/bin/env bash
# Build the dedicated HEALPix kernel image.
#
# This is deliberately a separate image from the aga-* one the rest of healda runs
# on, even though both derive from the same base. The kernel work installs
# torch-harmonics editable against the mounted source tree and rebuilds it
# constantly; doing that inside the shared healda image would make every healda job
# depend on whatever state the kernel tree happens to be in.
#
# Submit, do not run on the login node: enroot unpacks ~28 GB and re-squashes it,
# which needs the node-local /raid.
#
#   sbatch sbatch_scripts/build_healpix_container.sh
#
# The mechanics follow build_container.sh, which is the pattern already proven on
# this cluster: enroot create, write into the rootfs with --root --rw so the change
# is captured, then export. The original version of this script used
# `srun --container-save`; that is not what the working recipe here does.
#
#SBATCH -A coreai_devtech_all
#SBATCH -p cpu
#SBATCH -q cpu-normal
#SBATCH -J build-healpix-img
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 16 CPUs / 64 GB as in build_container.sh: enroot.conf caps mksquashfs at
# "-processors 8 -mem 23G", and leaving --mem unset makes Slurm derive it from
# DefMemPerCPU and never schedule.
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# Longer than build_container.sh's 2h because this one also compiles the CUDA
# extension: templated over 16 NLOC values per dtype, for two architectures.
#SBATCH -t 04:00:00
#SBATCH -o /home/acarpentieri/healda_project/logs/build_healpix_container_%j.out
#SBATCH -e /home/acarpentieri/healda_project/logs/build_healpix_container_%j.err

set -euxo pipefail

PROJECT="/home/acarpentieri/healda_project"
CONTAINERS="${PROJECT}/containers"
WHEELS="${CONTAINERS}/build/wheels"

# A stock NGC PyTorch image, not the e2s one the rest of healda runs on. Two
# reasons beyond keeping the images separate:
#
#   - it carries torch 2.14 / CUDA 13.4, which is the toolchain the handover's
#     measurements were taken on. The e2s base is torch 2.12 / CUDA 13.2, so every
#     register and occupancy number would have to be re-established against it.
#   - the kernel work needs torch, pytest and the checkout, nothing else. The e2s
#     base adds earth2studio, physicsnemo, transformer-engine, numba and zarr, none
#     of which any ragged test or benchmark imports.
#
# nvcr.io answers /v2/ with 401 here, so this is a local copy rather than a pull.
# It belongs to another user, which matters only at build time: the exported image
# is ours and self-contained. Override BASE to point somewhere else.
BASE="${BASE:-/lustre/fsw/portfolios/coreai/projects/coreai_devtech_all/users/zhenghangr/work_dir/.enroot/pytorch-26.08-py3-doca35.sqsh}"
FINAL="${CONTAINERS}/healpix_container.sqsh"

[[ -f "${BASE}" ]] || { echo "ERROR: missing base image ${BASE}" >&2; exit 2; }
[[ -d "${PROJECT}/torch-harmonics-hpx" ]] || { echo "ERROR: missing ${PROJECT}/torch-harmonics-hpx" >&2; exit 2; }
[[ -f "${PROJECT}/sbatch_scripts/setup_healpix_container.sh" ]] || { echo "ERROR: missing setup_healpix_container.sh" >&2; exit 2; }
# healda is deliberately not required: nothing in the ragged tests or benchmarks
# imports it. See the header of setup_healpix_container.sh.
for w in pytest pytest_regtest parameterized; do
  ls "${WHEELS}/${w}"-*.whl >/dev/null 2>&1 || { echo "ERROR: no ${w} wheel in ${WHEELS}" >&2; exit 2; }
done
[[ ! -e "${FINAL}" ]] || { echo "ERROR: ${FINAL} already exists; move it aside first" >&2; exit 2; }

mkdir -p "${PROJECT}/logs" "${CONTAINERS}"

NAME="healpix-build-${SLURM_JOB_ID:-manual}"
cleanup() { enroot remove -f "${NAME}" 2>/dev/null || true; }

df -h /raid

# Create first, arm the trap only once we own the container.
#
# The trap used to be armed before this, which made a second invocation destroy
# the first one's work: without SLURM_JOB_ID the name is fixed, so `enroot create`
# fails with "File already exists", and the failing run's own EXIT trap then
# removed the container the *running* build was using. That is how a 32-minute
# compile was lost. Submitting through Slurm gives each job a unique name, but the
# ordering here is what makes the manual path safe too.
enroot create --name "${NAME}" "${BASE}"
trap cleanup EXIT

ROOTFS="${ENROOT_DATA_PATH:-/raid/containers/data/user-$(id -u)}/${NAME}"

# Stage the wheels inside the rootfs rather than bind-mounting them: enroot's
# --mount requires the target to already exist in the image, which neither
# /workspace nor /lustre does in this base. Create both before mounting over them.
mkdir -p "${ROOTFS}/tmp/wheels" "${ROOTFS}/workspace" "${ROOTFS}/lustre"
cp "${WHEELS}"/*.whl "${ROOTFS}/tmp/wheels/"

# Copy the provisioning script in rather than running it from the bind mount. Bash
# reads a script incrementally and remembers a byte offset, so editing the file
# while a build is running makes the running shell resume at that offset inside the
# new contents -- which is a silent, arbitrary corruption of a job that can take an
# hour. The snapshot means the build is immune to edits made after it starts.
cp "${PROJECT}/sbatch_scripts/setup_healpix_container.sh" "${ROOTFS}/tmp/setup_healpix_container.sh"

# Provisioning lives in setup_healpix_container.sh rather than inline: nesting its
# python snippets inside a quoted bash -lc string strips their quotes.
#
# NVIDIA_VISIBLE_DEVICES=void skips enroot's 98-nvidia.sh hook, which needs
# nvidia-container-cli and fails outright on any node without the driver -- the
# login and vscode nodes included. Nothing here needs a GPU: the provisioning is
# written for a CPU node, which is why it sets TORCH_HARMONICS_BUILD_CUDA_EXTENSION
# and an explicit arch list instead of autodetecting. So this is unconditional
# rather than guarded, and the build behaves the same wherever it runs.
NVIDIA_VISIBLE_DEVICES=void enroot start --root --rw \
  --mount "${PROJECT}:/workspace" \
  --mount /lustre:/lustre \
  --env "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-10.0a 10.3a}" \
  "${NAME}" \
  bash /tmp/setup_healpix_container.sh

# Export to a temp name and rename, so an interrupted build never leaves a truncated
# .sqsh that looks usable.
enroot export --output "${FINAL}.partial" "${NAME}"
mv "${FINAL}.partial" "${FINAL}"

ls -lh "${FINAL}"
unsquashfs -s "${FINAL}" | head -5
echo "built image: ${FINAL}"
