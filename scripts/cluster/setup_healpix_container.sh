#!/usr/bin/env bash
# Provisioning steps that run *inside* the container, invoked by
# build_healpix_container.sh. Kept as its own file rather than inlined into the
# enroot command line so that the python snippets below can use quotes freely;
# nesting them in a quoted `bash -lc` string silently strips them.
#
# Expects the project mounted at /workspace and the vendored wheels in /tmp.
#
# The base is a stock NGC PyTorch image, which carries torch and the CUDA toolchain
# and nothing else this needs. That is the whole point of choosing it: the kernel
# work depends on torch, pytest and the checkout, and on nothing in the healda or
# earth2studio stack. tests/test_attention_ragged.py imports torch, parameterized,
# attention_helpers and torch_harmonics; the one test that wants earth2grid calls
# skipTest when it is absent. So earth2grid and healda are deliberately NOT
# installed here -- the FlexAttention comparison in §8 needs them and should get
# its own image or its own install step, rather than making every kernel run carry
# a stack it does not use.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"

# PIP_CONSTRAINT is baked into the NGC images and would fight the installs below.
unset PIP_CONSTRAINT CONDA_PREFIX

echo "=== base image inventory ==="
python -c "import sys; print('python', sys.version)"
python - <<'PY'
import importlib.metadata as md

for name in ["torch", "numpy", "pytest", "parameterized", "torch-harmonics"]:
    try:
        print(f"  {name}: {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name}: MISSING")
PY
python -c "import torch; print('  torch cuda', torch.version.cuda)"
nvcc --version | tail -2 | head -1

echo "=== test dependencies, from the vendored wheels ==="
# --no-index so the build needs no outbound network at all, which is what lets it
# run on a compute node. The wheels were resolved on this same base image, so the
# set is complete and the one binary wheel among them (cachebox) is aarch64.
python -m pip install --no-cache-dir --no-index --find-links /tmp/wheels \
    pytest pytest-regtest parameterized

echo "=== torch-harmonics from the local HEALPix branch ==="
# Drop anything a previous build left behind. The tree is bind-mounted, so build
# artifacts outlive the container that produced them, and an incremental build_ext
# that reuses them can link a .so whose contents do not match the current source
# list -- which is exactly how a newly added .cu came to be silently absent from an
# otherwise successful build.
rm -rf "${WORKSPACE}/torch-harmonics-hpx/build" "${WORKSPACE}/torch-harmonics-hpx"/*.egg-info
find "${WORKSPACE}/torch-harmonics-hpx" -name "*.so" -delete

# setup.py decides between CUDAExtension and CppExtension with
#   BUILD_CUDA = TORCH_HARMONICS_BUILD_CUDA_EXTENSION or (torch.cuda.is_available() and CUDA_HOME)
# and this build step runs without a GPU, so the autodetect arm is always False.
# The variable name has to match exactly: an unrecognised one (this script used to
# pass FORCE_CUDA_EXTENSION) leaves BUILD_CUDA False, and the build then succeeds
# while quietly producing CPU-only extensions -- every op still gets declared by
# attention_interface.cpp, optimized_kernels_is_available() still returns True, and
# the omission only surfaces as "Could not run ... with arguments from the 'CUDA'
# backend" the first time a test calls a kernel that has no CPU fallback.
#
# TORCH_CUDA_ARCH_LIST is required for the same reason: with no GPU present nvcc has
# nothing to autodetect and would target the toolkit default, so the kernels would
# not load on the test nodes. 10.0a covers GB200, 10.3a covers GB300. It is taken
# from the environment so it can be corrected without editing this file -- a wrong
# value here builds cleanly and then fails to load at test time.
#
# --no-build-isolation so the extension links against the torch already in the
# image instead of a second copy pip would resolve, and --no-deps so pip cannot
# resolve a different torch over the one it is compiling against.
TORCH_HARMONICS_BUILD_CUDA_EXTENSION=1 \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0a 10.3a}" \
    python -m pip install -v --no-cache-dir --no-build-isolation --no-deps -e "${WORKSPACE}/torch-harmonics-hpx"

echo "=== which attention CUDA sources were compiled ==="
python - <<'PY'
import pathlib
import re

# Read back the source list setup.py would produce, so the log records what the
# build was asked to compile independently of what pip reported doing.
setup = pathlib.Path("/workspace/torch-harmonics-hpx/setup.py").read_text()
for m in re.finditer(r'"(torch_harmonics/attention/optimized/kernels_cuda/[^"]+\.cu)"', setup):
    print("  listed:", m.group(1))
PY

echo "=== verification ==="
python - <<'PY'
import pathlib

import torch

import torch_harmonics as th

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torch_harmonics", th.__version__)
print("torch_harmonics from", th.__file__)
print("grid types", th.grid_types())

# The editable install has to resolve to the mounted tree, or every later run
# silently measures the stock package instead of the kernels just compiled.
if not th.__file__.startswith("/workspace/"):
    raise SystemExit(f"ERROR: torch_harmonics resolved to {th.__file__}, not the mounted tree")

hpx = th.HealpixGrid(nside=64)
print("HealpixGrid(nside=64):", hpx.npoints, "pixels on", hpx.nlat, "rings")

from torch_harmonics.attention import optimized_kernels_is_available as attention_kernels
from torch_harmonics.disco import optimized_kernels_is_available as disco_kernels

print("attention optimized kernels:", attention_kernels())
print("disco optimized kernels:", disco_kernels())

# Declared is not the same as implemented, and the difference is invisible until a
# test actually calls the op: `m.def` lives in attention_interface.cpp and always
# compiles, while the CUDA kernel is registered from its own .cu. A build that
# missed that .cu still declares the operator, still reports the optimized kernels
# as available, and then fails at call time with "Could not run ... with arguments
# from the 'CUDA' backend".
#
# `forward` is the control rather than another subject: it has had a CUDA kernel
# since long before this branch, so if the probe cannot see one for `forward` then
# the probe is wrong and any verdict it gives about the others is worthless. Only a
# control that passes turns a missing kernel into a build failure -- a check that
# can fail for its own reasons is worse than no check, because it stops the build
# for reasons that have nothing to do with the build.
CONTROL = "forward"
SUBJECTS = ["backward", "forward_ragged", "backward_ragged"]


def cuda_registered(name):
    qualname = f"attention_kernels::{name}"
    try:
        getattr(torch.ops.attention_kernels, name)
    except (AttributeError, RuntimeError):
        return None, "not declared"
    try:
        dump = torch._C._dispatch_dump(qualname)
    except Exception as exc:  # noqa: BLE001 - probe, not policy
        return None, f"dump failed: {exc}"
    keys = [line.split(":")[0].strip() for line in dump.splitlines() if ":" in line]
    return ("CUDA" in keys), ", ".join(k for k in keys if "Autograd" not in k)


control_ok, control_keys = cuda_registered(CONTROL)
print(f"control {CONTROL}: cuda={control_ok} keys=[{control_keys}]")

results = {}
for name in SUBJECTS:
    ok, keys = cuda_registered(name)
    results[name] = ok
    print(f"        {name}: cuda={ok} keys=[{keys}]")

# The control distinguishes the two failure modes, which need opposite fixes:
# if even `forward` has no CUDA kernel then nothing CUDA was compiled and the
# problem is the BUILD_CUDA gate in setup.py, not any individual source file.
if not control_ok:
    raise SystemExit(
        f"ERROR: {CONTROL} has no CUDA kernel, so this build produced CPU-only "
        "extensions. setup.py gates on TORCH_HARMONICS_BUILD_CUDA_EXTENSION=1 "
        "(plus TORCH_CUDA_ARCH_LIST, since a CPU build node has no GPU to "
        "autodetect); check those are exported for the pip install above."
    )

missing = [n for n, ok in results.items() if not ok]
if missing:
    raise SystemExit(
        f"ERROR: {CONTROL} has a CUDA kernel but these do not: {', '.join(missing)}. "
        "Their .cu sources were not compiled into torch_harmonics/attention/_C."
    )
PY
