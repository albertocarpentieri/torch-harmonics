# HEALPix backend for torch-harmonics — status and cluster handoff

Written 2026-09-10, ahead of a move to a different cluster.

Two independent workstreams live in this workspace:

1. **The HEALPix backend for `torch-harmonics`** — functionally complete and validated
   on GPU, one benchmark outstanding. **Pushed and safe** as of 2026-09-10.
2. **The `nnja-nnjaConv-104ch-windDrop50` training run** — 96.7% finished, idle since
   Sep 6. **Abandoned** by decision at the cluster move; not migrated.

The backend is commit `ec8f597` on branch `acarpentieri/healpix-backend` of the personal
fork `git@github.com:albertocarpentieri/torch-harmonics.git`. Recloning that branch on
the new cluster recovers all of the code. The Slurm glue — the six
`sbatch_scripts/*healpix*` scripts and this document — is on the separate branch
`acarpentieri/healpix-cluster-glue` of the same fork, kept apart from the code so the
backend branch stays clean for an eventual upstream MR.
See [Migration checklist](#migration-checklist).

---

## 1. What the backend does

`torch-harmonics` assumed every grid is a latitude/longitude product grid: `nlat` rings
of `nlon` points each, addressable as `(nlat, nlon)`. HEALPix is not one. It has
`12 * nside**2` equal-area pixels on `4 * nside - 1` rings whose lengths vary from `4`
at the poles to `4 * nside` at the equator, so a field cannot be stored as a rectangle.
This work makes the grid layer and neighborhood attention handle that "ragged" case.

The scope was deliberately limited to the grid descriptors and `NeighborhoodAttentionS2`.
Spherical harmonic transforms were excluded, and resampling is left to `earth2grid`.

### Grid layer

`GridS2` became an abstract base holding only what every sphere grid can answer, with a
new `RegularGridS2` in between carrying the `nlat`/`nlon` product-grid assumptions that
`EquiangularGrid`, `LegendreGaussGrid`, `LobattoGrid` and `EquiangularTrapezoidalGrid`
inherit. `HealpixGrid` (new file `torch_harmonics/healpix.py`) descends directly from
`GridS2` and computes its ring structure analytically in RING order.

Two details worth knowing:

- **RING order, not NEST.** The arc-segment kernels need the points of a ring to be
  contiguous in memory, which is exactly what RING ordering gives and NEST does not.
  `healda` internally uses a face-based padded order (`HEALPIX_PAD_XY`), so a conversion
  is required at the boundary.
- **`theta_cutoff` is anisotropic.** HEALPix ring spacing and within-ring spacing differ,
  so `HealpixGrid.theta_cutoff()` takes `max(max_latitude_spacing, max_longitude_spacing)`
  rather than latitude spacing alone.

### Neighborhood attention

A product grid exploits rotational self-similarity: one neighbor list per output *ring*
is computed and slid along the ring by a "p-shift". A ragged grid has no such symmetry —
rotating a ring by one of its own points leaves the rings above and below misaligned,
because they hold a different number of points. The pattern must therefore be keyed per
output **point**, which inflates it by `npoints/nrings ≈ 3 * nside` (about 193× at
nside 64). This is the structural cost of raggedness and it cannot be designed away.

Two things blunt it. The **arc encoding** stores `(ring, start, len)` per run of
consecutive neighbors instead of one index per neighbor, so it compresses by the mean arc
length. And the **backward kernel replays the arcs** rather than storing per-neighbor
attention weights, so the blowup is not inherited by the backward pass.

`NeighborhoodAttentionS2` now branches on whether both grids are regular. Because the
ragged kernels are CUDA-only, the choice between the optimized handle and the torch
reference is made **per call** on `query_scaled.is_cuda`, not once at construction — a
CPU module must fall back even on a machine whose extension has the CUDA kernels.

### Files

New:

| File | Lines | What |
|---|---|---|
| `torch_harmonics/healpix.py` | 487 | `HealpixGrid`, analytic RING geometry |
| `torch_harmonics/neighborhood.py` | 481 | arc precompute, `NeighborhoodArcsS2`, `to_csr` |
| `torch_harmonics/attention/kernels_torch/attention_ragged_torch.py` | 342 | torch reference |
| `.../kernels_cuda/attention_cuda_fwd_ragged.cu` | 312 | ragged forward CUDA |
| `.../kernels_cuda/attention_cuda_bwd_ragged.cu` | 479 | ragged backward CUDA |
| `tests/test_attention_ragged.py` | 751 | ragged attention + kernel tests |
| `tests/test_neighborhood.py` | 446 | precompute and CSR expansion tests |
| `benchmarks/healpix_ragged_report.py` | 387 | memory and timing report |

Modified: `grid.py` (+343), `attention/attention.py` (+350), `truncation.py`,
`attention_optimized.py`, `attention_interface.cpp`, `attention_cuda.cuh`, `setup.py`,
`__init__.py`, `_attention_utils.py`, `tests/test_grid_assumptions.py` (+483),
`tests/test_truncation.py`, `tests/test_cache.py`.

---

## 2. Validation evidence

| Job | Date | Result |
|---|---|---|
| 3548669 | Sep 4 | Ragged fwd+bwd CUDA on GB300: **101 passed, 81 subtests, 0 failed** |
| 3547103 | Sep 4 | Container build: all four ops registered with CUDA |
| 3527224 | Sep 3 | Vectorized `to_csr`: 88 passed |
| 3523402 | Sep 3 | `healda` bridge: 23 passed, locality + `TransformerBlock` swap |
| 3522883 | Sep 3 | Regression: regular attention + grid assumptions unaffected |
| 3525330 | Sep 3 | `truncate_support` warning fix: 676 passed |

Operator registration as of job 3548669:

```
attention_kernels::forward:         CUDA=True  CPU=True
attention_kernels::backward:        CUDA=True  CPU=True
attention_kernels::forward_ragged:  CUDA=True  CPU=False
attention_kernels::backward_ragged: CUDA=True  CPU=False
```

The ragged ops being `CPU=False` is by design, not a gap — hence the per-call device
dispatch described above.

---

## 3. What is not done

**The benchmark has never produced numbers.** Job 3629258 (Sep 8) failed for two reasons,
both now fixed in `sbatch_scripts/bench_healpix_ragged.sh` but not yet re-run:

- Its preflight imported only `torch`, so `torch.ops.attention_kernels` was an empty
  namespace and both ragged ops reported "unavailable". **This was a false negative** —
  the kernels were present; importing `torch_harmonics` is what loads the extension.
  Do not read that log as evidence of a broken build.
- It hit the 30-minute wall clock, and because stdout was block-buffered to a file, even
  the completed pattern table was lost. Now `python -u`, and the limit is 1h30.

So the memory and speed claims made so far are **analytical, not measured**:
raggedness costs a factor `npoints/nrings` in pattern memory, partly recovered by the arc
encoding; the ragged kernel should beat the torch reference comfortably but run somewhat
slower than the product-grid kernel at equal point counts, because it reads a distinct
pattern per point instead of reusing one per ring. The benchmark now has an equiangular
arm at matched point count (`nlat = round(sqrt(6) * nside)`, within 1%) and matched
cutoff specifically to test that last claim.

**No performance numbers exist yet at all.** The benchmark is the only functional gap.

**Known inefficiency, not addressed:** `_setup_ragged` builds and registers *both* the arc
and CSR pattern buffers, but the CUDA path reads only the arcs. `psi_col_idx` is the
largest allocation and is dead weight on GPU runs. Worth dropping once the benchmark says
whether pattern memory actually matters at nside 64.

**MR feedback for Thorsten** on the upstream `tkurth/grid-descriptor` `GridS2` layout was
drafted but never delivered. The HEALPix work exposed real divergences, chiefly that
`nlat`/`nlon` cannot live on the base class.

---

## 4. Migration checklist

### Done

The backend is committed as `ec8f597` and pushed to the personal fork:

```bash
git clone git@github.com:albertocarpentieri/torch-harmonics.git
git checkout acarpentieri/healpix-backend
```

It deliberately does **not** live on `NVIDIA/torch-harmonics`. It was briefly pushed
there and has been deleted; work in progress belongs on the fork until it is ready to
be proposed upstream.

Two notes on the push, since they cost time. The `origin` remote is HTTPS and there are
no stored GitHub credentials and no `gh` CLI, so the push went over SSH via a separate
`nvidia-ssh` remote. And the system SSH config here is malformed (`Bad owner or
permissions on /etc/ssh/ssh_config.d/cm.conf`), so it has to be bypassed, with the key
named explicitly — `id_ed25519` is the one GitHub accepts, `id_ecdsa` is not:

```bash
export GIT_SSH_COMMAND="ssh -F /dev/null -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519"
```

### The cluster glue

`/home/acarpentieri/healda_project` is not a git repository, so the Slurm scripts and
this document are preserved on their own branch of the same fork:

```bash
git fetch origin acarpentieri/healpix-cluster-glue
git checkout acarpentieri/healpix-cluster-glue
```

It is an orphan branch — unrelated history, no `torch-harmonics` source — holding only
the six scripts and this file. It is kept out of `acarpentieri/healpix-backend` so that
branch stays proposable upstream, and out of the `healda` repo because the glue has
nothing to do with the training work that lives there.

The scripts are cluster-specific and would need rewriting for new hardware anyway, but
the build recipe in `setup_healpix_container.sh` encodes hard-won detail (see the gating
trap below) and is worth carrying over as a reference rather than reconstructing.

### Cluster-specific values to change on the far side

These are hardcoded to the current cluster and will all need editing:

| Thing | Current value |
|---|---|
| Account | `coreai_climate_earth2` |
| Partitions | `batch`, `batch_long`, `cpu` |
| QOS | `normal`, `cpu-normal`, `interactive` |
| Project path | `/home/acarpentieri/healda_project` (mounted as `/workspace`) |
| Lustre root | `/lustre/fsw/portfolios/coreai/users/acarpentieri/` |
| Base image | `.../fcn3_project/containers/ngc_pytorch_26.08_train.sqsh` |
| **GPU arch** | `TORCH_CUDA_ARCH_LIST="10.0a 10.3a"` |

The GPU architecture list is the one most likely to bite. `10.0a`/`10.3a` targets the
GB300s here; on different hardware the extension will build but the kernels will not run.

There is also a **`QOSMinGRES` policy on this cluster requiring at least 4 GPUs**, which
is why the test and bench scripts request `--gres=gpu:4` for what is a single-GPU job.
On a cluster without that policy, drop it back to 1.

### Rebuilding the container

```bash
sbatch sbatch_scripts/build_healpix_container.sh
```

This submits the outer job, which runs `setup_healpix_container.sh` inside the base image.
Do not submit the setup script directly — it has no `#SBATCH` headers and will be rejected
for having no account.

The build must export `TORCH_HARMONICS_BUILD_CUDA_EXTENSION=1` *and* `TORCH_CUDA_ARCH_LIST`.
`setup.py` gates CUDA compilation on that variable or on `torch.cuda.is_available()`, and
the build runs on a CPU node where the latter is false — without both, you silently get a
CPU-only extension that fails at runtime with "could not run ... with arguments from the
CUDA backend". The build script verifies registration afterward and fails loudly, using
the long-standing `forward` op as a control to distinguish "this image has no attention
CUDA kernels at all" from "the new `.cu` did not make it in".

### Re-running the work

```bash
# validation (needs a GPU node)
sbatch sbatch_scripts/test_healpix_gpu.sh

# the outstanding benchmark
sbatch sbatch_scripts/bench_healpix_ragged.sh
```

---

## 5. The training run — abandoned

Separate from the backend. **Decision at the cluster move: abandoned at 96.7%, not
resumed and not migrated.** Recorded here because the artifacts still exist and because
the run is 97 kimg from a usable end state, should anyone want to revive it.

`v2-videoDA-nnja-nnjaConv-104ch-windDrop50` is at **2,902,636 of 3,000,000 images
(96.7%)** and has been idle since Sep 6 18:38. It did not crash: job 3548660 hit its
4-hour wall clock, caught `SIGTERM`, saved cleanly and exited. The chain stopped only
because each resume was submitted by hand and nobody submitted the next.

- Run dir: `/lustre/fsw/portfolios/coreai/users/acarpentieri/healda_project/nnja_nnjaConv_104ch_windDrop50/training-runs/v2-videoDA-nnja-nnjaConv-104ch-windDrop50`
- Latest checkpoint: `training-state-002902636.checkpoint` (18 GB; several older ones alongside)
- Code: `healda` at `dd6c2e40`, clean worktree
- Throughput: ~199 sec/kimg, ~69.5 kimg per 4-hour job
- **Remaining: ~97 kimg, about two more jobs**

Reviving it on this cluster would take about two 4-hour jobs. Reviving it elsewhere means
migrating the 18 GB checkpoints and re-establishing the data paths (`UFS_OBS`, the ERA5
104-channel zarr, several S3 profiles), which is what made abandoning it the cheaper
option. If it is ever resumed here, chain the jobs so it does not stall again:

```bash
cd /home/acarpentieri/healda_project
export LOOP_NAME=v2-videoDA-nnja-nnjaConv-104ch-windDrop50
export RUN_ROOT=/lustre/fsw/portfolios/coreai/users/acarpentieri/healda_project/nnja_nnjaConv_104ch_windDrop50

J1=$(sbatch --parsable --job-name=nnja-104ch-wd50-resume \
      --export=ALL,LOOP_NAME,RUN_ROOT sbatch_scripts/train_nnja_nnjaConv_4node.sh)
sbatch --dependency=afterany:$J1 --job-name=nnja-104ch-wd50-resume \
      --export=ALL,LOOP_NAME,RUN_ROOT sbatch_scripts/train_nnja_nnjaConv_4node.sh
```

`RUN_ROOT` must be set explicitly: the script defaults to `nnja_nnjaConv_era5static`,
which is a different run. Resume is automatic from the newest checkpoint in the run
directory and is idempotent, so a failed `sbatch` can simply be repeated.

One unexplained detail: job 3548658's log is 247 KB against the ~60 KB every other job in
the chain produced. It ended normally and the chain continued through it, so it is not
blocking, but something noisy happened mid-run.

---

## 6. Traps already paid for

Recorded so they are not rediscovered on the new cluster.

- **`setup.py` gating.** Covered above; the most expensive single mistake here, because it
  fails at runtime rather than build time.
- **Registration checks need `import torch_harmonics` first.** Otherwise
  `torch.ops.attention_kernels` is empty and everything looks missing. This wasted the one
  benchmark run.
- **`opcheck` cannot run `test_aot_dispatch_dynamic` on the ragged ops.** Data-dependent
  indexing raises `GuardOnDataDependentSymNode` during AOT tracing. The tests restrict to
  `test_schema`, `test_autograd_registration`, `test_faketensor` deliberately.
- **`register_fake` must be conditional.** Registering a fake for an op an older compiled
  extension does not declare makes the whole package fail to import. Hence `_op_is_declared`.
- **`GridS2` cannot define `nlat`/`nlon` as properties.** It breaks dataclass inheritance in
  the subclasses with "property has no setter". This is also the core of the upstream MR
  feedback.
- **Slurm here is flaky**, and intermittently demands `--account` after outages. Submission
  loops with retries are worth keeping.
- **Do not run Python on the login node.** Use the containers.
