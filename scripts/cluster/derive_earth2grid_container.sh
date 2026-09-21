#!/usr/bin/env bash
# Add earth2grid to the HEALPix image, without recompiling the CUDA kernels.
#
# grid_vs_flex.py needs earth2grid for one thing, and it is not a cosmetic thing:
# it builds the NEST-ordered grid that flex's block mask is derived from. Handing
# flex a RING ordering instead does not fail, it quietly triples flex's tile count
# -- 2760 active tiles at level 5 where NEST gives 824 -- and every ratio in the
# table then understates flex. So the comparison cannot be run without it.
#
# earth2grid is not on PyPI, and the copy in the e2s image cannot be reused: its
# _healpix_bare extension links libc10, libtorch_cpu and libtorch_python, so it is
# bound to that image's torch 2.12 while this one is 2.14. It has to be compiled
# against our torch, from the vendored source tarball.
#
# This derives from the finished image rather than rebuilding it, so the 40 minutes
# of ragged kernel compilation is not repeated; the cost is one small C extension
# plus the re-squash.
#
#   sbatch sbatch_scripts/derive_earth2grid_container.sh
#
#SBATCH -A coreai_devtech_all
#SBATCH -p cpu
#SBATCH -q cpu-normal
#SBATCH -J derive-e2g-img
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /home/acarpentieri/healda_project/logs/derive_earth2grid_%j.out
#SBATCH -e /home/acarpentieri/healda_project/logs/derive_earth2grid_%j.err

set -euxo pipefail

PROJECT="/home/acarpentieri/healda_project"
CONTAINERS="${PROJECT}/containers"
SRC="${CONTAINERS}/build/src/earth2grid-main.tar.gz"

BASE="${BASE:-${CONTAINERS}/healpix_container.sqsh}"
FINAL="${CONTAINERS}/healpix_container_e2g.sqsh"

[[ -f "${BASE}" ]] || { echo "ERROR: missing base image ${BASE}" >&2; exit 2; }
[[ -f "${SRC}" ]] || { echo "ERROR: missing earth2grid source ${SRC}" >&2; exit 2; }
[[ -d "${PROJECT}/torch-harmonics-hpx" ]] || { echo "ERROR: missing torch-harmonics-hpx" >&2; exit 2; }
[[ ! -e "${FINAL}" ]] || { echo "ERROR: ${FINAL} already exists; move it aside first" >&2; exit 2; }

mkdir -p "${PROJECT}/logs"

NAME="e2g-derive-${SLURM_JOB_ID:-manual}"
cleanup() { enroot remove -f "${NAME}" 2>/dev/null || true; }

df -h /raid

# Create first, arm the trap only once we own the container, so a name collision
# with a concurrent run cannot delete the other run's work.
enroot create --name "${NAME}" "${BASE}"
trap cleanup EXIT

ROOTFS="${ENROOT_DATA_PATH:-/raid/containers/data/user-$(id -u)}/${NAME}"
mkdir -p "${ROOTFS}/workspace" "${ROOTFS}/lustre"
cp "${SRC}" "${ROOTFS}/tmp/earth2grid-main.tar.gz"

# NVIDIA_VISIBLE_DEVICES=void skips enroot's nvidia hook, which needs
# nvidia-container-cli and fails on nodes without a driver. Nothing here needs a
# GPU: earth2grid's extension is CPU-only C against the torch headers.
NVIDIA_VISIBLE_DEVICES=void enroot start --root --rw \
  --mount "${PROJECT}:/workspace" \
  --mount /lustre:/lustre \
  "${NAME}" \
  bash -c '
set -euxo pipefail
unset PIP_CONSTRAINT CONDA_PREFIX

# --no-build-isolation so the extension links against the torch in the image
# rather than a copy pip would fetch, and --no-deps so pip cannot pull a
# different torch or numpy over the ones the ragged kernels were compiled
# against. Everything earth2grid needs at runtime -- numpy, einops, scipy -- is
# already present; only the optional netCDF4 extra is absent, and nothing in
# grid_vs_flex.py touches it.
python -m pip install -v --no-cache-dir --no-build-isolation --no-deps \
    /tmp/earth2grid-main.tar.gz
rm -f /tmp/earth2grid-main.tar.gz

python - <<PY
import importlib.metadata as md
import torch

import earth2grid.healpix as healpix

print("earth2grid", md.version("earth2grid"))

# The exact call grid_vs_flex.py makes. A NEST grid is the whole reason for this
# image, so prove it constructs and has the right extent before shipping.
grid = healpix.Grid(5, pixel_order=healpix.NEST)
lat = torch.as_tensor(grid.lat)
lon = torch.as_tensor(grid.lon)
npix = lat.numel()
print("level 5 NEST:", npix, "pixels", "lat", float(lat.min()), float(lat.max()))
if npix != 12 * 32 * 32:
    raise SystemExit(f"ERROR: level 5 should have {12*32*32} pixels, got {npix}")

# And prove the derive did not disturb what the image was built for.
import torch_harmonics as th
print("torch_harmonics from", th.__file__)
if not th.__file__.startswith("/workspace/"):
    raise SystemExit("ERROR: torch_harmonics no longer resolves to the mounted tree")
for name in ("forward_ragged", "backward_ragged"):
    dump = torch._C._dispatch_dump(f"attention_kernels::{name}")
    keys = {ln.split(":")[0].strip() for ln in dump.splitlines() if ":" in ln}
    # Bound to a name rather than inlined: a backslash-escaped quote inside an
    # f-string expression is a SyntaxError, and it cost this job a whole run.
    has_cuda = "CUDA" in keys
    print(f"attention_kernels::{name}: CUDA={has_cuda}")
    if not has_cuda:
        raise SystemExit(f"ERROR: {name} lost its CUDA kernel")
print("ok: earth2grid added, ragged kernels intact")
PY
'

enroot export --output "${FINAL}.partial" "${NAME}"
mv "${FINAL}.partial" "${FINAL}"

ls -lh "${FINAL}"
echo "derived image: ${FINAL}"
