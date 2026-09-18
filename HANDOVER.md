# HEALPix ragged neighbourhood attention — handover

Branch: `acarpentieri/healpix-backend` — on the **fork**,
`github.com/albertocarpentieri/torch-harmonics`, *not* `NVIDIA/torch-harmonics`.
That trips people up: the NVIDIA remote is also configured here (as `origin` and
`nvidia-ssh`) and this branch is not on it.

Companion branches, both on the healda **fork**
`gitlab-master.nvidia.com/acarpentieri/healda`, not on `earth-2/healda`:

- `ac/bench-disk-attn` — the benchmarks, on top of MR 56's FlexAttention layer.
  No kernel code.
- `ac/nnja-harmonics-latlon` — the healda-side model work: the neighbourhood
  attention block and the lat/lon decoder. Not needed to run or continue the
  kernel work, listed so it is not lost.

To get everything:

```bash
git clone -b acarpentieri/healpix-backend git@github.com:albertocarpentieri/torch-harmonics.git torch-harmonics-hpx
git clone -b ac/bench-disk-attn ssh://git@gitlab-master.nvidia.com:12051/acarpentieri/healda.git bench-disk
```

Every branch here lives on a personal fork rather than the upstream project, and
in both cases the upstream is also configured as a remote in the working copy. So
looking for this work on `NVIDIA/torch-harmonics` or `earth-2/healda` finds
nothing, which has already cost one person an afternoon.

This documents where the work stands, how to run it, and what is still open. It is
written for someone picking it up cold, possibly on a different cluster.

---

## 1. What this is about

`torch-harmonics` implements spherical neighbourhood attention: each point on a
sphere attends only to the points within a geodesic disk around it. There are two
grid families and they have separate CUDA kernels:

- **product grids** (equiangular lat/lon) — `attention_cuda_{fwd,bwd}.cu`
- **ragged grids** (HEALPix) — `attention_cuda_{fwd,bwd}_ragged.cu`

The ragged kernels are the ones healda needs, and they were far slower than they
should have been. The work on this branch makes them faster, and separately
measures them against PyTorch's FlexAttention, which is the alternative.

**The headline finding, before any of the optimisation:** the ragged kernels had
never received optimisations that the product-grid kernels already had. A comment in
`attention_cuda_fwd.cu` records the product-grid forward being measured at 1.6
TFLOP/s and diagnosed as latency-bound, then fixed. The ragged forward measured the
same 1.6 TFLOP/s and still had the unfixed structure. Most of the speedup here is
porting fixes that already existed in the sibling file.

---

## 2. Layout

```
~/healda_project/
  torch-harmonics-hpx/        <- the kernels. THIS branch.
  bench-disk/                 <- healda worktree holding MR 56's FlexAttention layer
                                 plus our benchmarks. Branch ac/bench-disk-attn.
  healda/, healda-harmonics/  <- the model. Not touched by this work.
  sbatch_scripts/             <- NOT under version control. All the run scripts.
  logs/                       <- job output.
```

`sbatch_scripts/` in the project root is untracked and will not come with a clone.
The ones this work needs are therefore vendored into **`scripts/cluster/`** in this
repo, so they travel with the branch. They refer to each other by the
`sbatch_scripts/...` paths they were written with; either run them from a checkout
that has been copied into a `sbatch_scripts/` directory, or adjust the paths. They
also hardcode this cluster's account, partition and container path — see below.

**Everything in them that is cluster-specific, and must be changed elsewhere:**

| what | value here | where |
|---|---|---|
| Slurm account | `coreai_climate_earth2` | `#SBATCH --account` in every job script |
| partition / QOS | `batch` / `interactive` | `#SBATCH` lines |
| GPU request | `--gres=gpu:4` | the QOS here rejects single-GPU requests with `QOSMinGRES`; the work is single-process and only uses `cuda:0` |
| container | `/lustre/.../containers/healpix_container.sqsh` | `CONTAINER=` at the top of each script |
| project root | `/home/acarpentieri/healda_project` | `PROJECT=` at the top of each script |

The container is the part that needs real work on a new cluster, not just a path
edit — see §2.1.

### 2.1 The container, and what to do if you do not have it

`/lustre/.../containers/healpix_container.sqsh` (~30 GB). CUDA 13.4, torch 2.14, and
— importantly — `torch_harmonics` installed **editable, pointing at
`/workspace/torch-harmonics-hpx`**. So the mounted source tree *is* the installed
package, and the compiled `.so` lives in the tree rather than in the image. Building
in place is what makes an edit take effect.

**On another cluster this image will not exist**, and a generic torch image will not
substitute: without the editable install, `import torch_harmonics` resolves to
whatever is baked in and your edits do nothing — silently, which is the bad part.
Verify before trusting any measurement:

```bash
python -c "import torch_harmonics as th; print(th.__file__)"
# must print a path under the mounted source tree, not site-packages
```

`scripts/cluster/setup_healpix_container.sh` and `build_healpix_container.sh` are how
this image was made; they install `earth2grid`, `torch-harmonics` and `healda` into a
base image. Rebuilding is the clean route. The quicker route, if you have any image
with a matching CUDA and torch, is to do the editable install at job start:

```bash
python -m pip install -q --no-deps --no-build-isolation -e /workspace/torch-harmonics-hpx
```

which is exactly what `bench_disk_attention.sh` already does for the healda worktree.
Container writes are ephemeral, so it has to be repeated per job.

---

## 3. Running things

### Compiling without a GPU or a scheduler

This is the single most useful trick here and it should survive the cluster move.
`enroot` can start the container on a login node once the nvidia hook is skipped,
and compiling needs the toolkit but not the driver:

```bash
NVIDIA_VISIBLE_DEVICES=void enroot start --mount $PWD:/workspace <container> bash -lc '...'
```

`sbatch_scripts/compile_check_ragged.sh` wraps that: it compiles both ragged kernels
for `sm_90a` and `sm_100a`, greps for errors/warnings/fatals, and runs
`benchmarks/ptxas_register_report.py` to print registers, spills and occupancy per
instantiation. Takes ~6 minutes. Use it on every kernel edit.

```bash
cd ~/healda_project
bash sbatch_scripts/compile_check_ragged.sh          # both files
FILES=attention_cuda_bwd_ragged bash sbatch_scripts/compile_check_ragged.sh
```

Both architectures are needed: the backward's atomic epilogue is split on
`__CUDA_ARCH__ < 900`, so each arch leaves half of it uncompiled.

Caveats learned the hard way:
- `/tmp` does not persist between `enroot start` invocations. Write under `/workspace`.
- `-std=c++20` is required; torch's headers reject older.
- `-Wall` is **not** an nvcc flag. It produces "nvcc fatal", which does not contain
  the word "error" and slips past a naive grep. The script checks the return code and
  greps for "fatal" too.

### The validation gate

```bash
sbatch sbatch_scripts/rebuild_and_validate_ragged.sh
```

Rebuilds in place, then runs `tests/test_attention_ragged.py` across **four switch
combinations**, then prints A/B timings. Roughly 50 minutes, of which ~40 is the
rebuild.

The four combinations exist so a failure is attributable without a rebuild:

| forward | backward | meaning |
|---|---|---|
| generic | generic | pre-existing code. Must pass or the harness is suspect. |
| special | generic | isolates the new forward |
| generic | special | isolates the new backward |
| special | special | what ships |

This has already paid for itself twice — once identifying a shared-Python-code bug
from a baseline-arm failure, once identifying which gradient term was at fault.

**Timings only print if all four gates pass.** A wrong kernel's speed is not
information.

### Benchmarks

```bash
# ragged kernel vs torch reference, and HEALPix vs equiangular
sbatch sbatch_scripts/bench_healpix_ragged.sh

# ours vs FlexAttention, sweeping grid size and radius
sbatch --export=ALL,SCRIPT="benchmarks/grid_vs_flex.py" sbatch_scripts/bench_disk_attention.sh
```

`grid_vs_flex.py` lives in `bench-disk/benchmarks/`. It takes `--grids hpx:<level>`
or `eq:<nlat>x<nlon>`, `--radius-deg` (accepts several), `--block-size`,
`--max-harmonics-pairs`.

### If Slurm is down

It was intermittent for days here. `sbatch_scripts/autosubmit_ragged.sh` polls and
submits when the controller answers, logging to `logs/autosubmit_ragged.log`, exiting
after one successful submission. `sbatch_scripts/run_ragged_validate_enroot.sh` does
the same work as the validate job without the scheduler, from any shell that can see
a GPU; it refuses with a clear message if none is visible.

---

## 4. What was changed, and what it bought

In order, oldest first. All on `acarpentieri/healpix-backend`.

| commit | change |
|---|---|
| `1b239b4` | Register-blocked forward kernel: accumulator in registers, `q` staged once per warp, neighbours in groups of 4 for memory-level parallelism. Port of the product grid's `s2_attn_fwd_special_vec_k`. |
| `6253cae` | Removed `sh_alpha_vw_`, an array holding one entry per channel where every entry always held the same warp-uniform scalar. A third of pass 1's shared-memory traffic. The redundancy is upstream's — the product-grid kernels have it too. |
| `b185b93` | Pinned the group size from `ptxas -v`: 4 is free relative to 2, 8 spills. |
| `196fed3` | Register-blocked backward kernel, ported from the product grid's `s2_attn_bwd_special_vec_k`. |
| `fde285b` | **Test coverage that made the rest meaningful.** Every existing ragged test used ≤32 channels, so `NLOC` was 1 and the kernels' unguarded register loops never executed. The suite could not see the register blocking at all. Added 96, 80, 40 and 192-channel cases. |
| `e9092eb` | Neighbour grouping in the backward's two walks. Measured +4.5%. |
| `1baa7c0` | Ragged CSR column list built on demand. It is only read by the torch reference; on the CUDA path it was 139 MiB of a 156 MiB pattern, allocated and never read. |
| `d16604c` | Kept `psi_col_idx` / `psi_roff_idx` readable via `__getattr__` after the above (the test suite reads them by name). |
| `d0e2b2f` | **Collapsed the backward's two neighbourhood walks into one.** See §5. |
| `3ce0d4e` | Routed bfloat16 back to the two-pass path — the collapse is not numerically safe there. See §6. |

### Measured, nside 64, fp32, 96 channels, 1 head, batch 1

| | forward | fwd+bwd |
|---|---|---|
| before any of this (job 3811782 baseline) | 3.965 ms | 13.543 ms |
| through `e9092eb` | 1.309 ms | 5.133 ms |
| through `3ce0d4e` (job 3837708) | 1.283 ms | **3.915 ms** |

**3.09× forward, 3.46× overall**, and the backward alone from 9.58 ms to 2.63 ms,
3.64×. Peak memory unchanged throughout. All four gates green in job 3837708.

The single-pass collapse was worth 1.31× of that (5.133 → 3.915). Less than the
halved traversals might suggest, for the reason ptxas gave in advance: it needs 96
registers against the two-pass 72, so 33% occupancy against 44%.

**Read the arms in job 3837708 with care.** Its baseline row says 10.475 ms, not
13.543, because `TORCH_HARMONICS_RAGGED_BWD_GENERIC=1` picks the generic *kernel*
while the one-pass/two-pass choice is a separate switch that defaults to one pass in
fp32. So the baseline arm now carries the collapse too and the four arms no longer
isolate what they did. The honest before-number is 13.543 ms from job 3811782.

**None of this is the training configuration.** The benchmark is fp32; training is
bf16, which routes to the two-pass path (§6), so training gets the forward's 3.09×
and none of the backward's. That is what makes the §6 trade worth pricing: 1.31× is
now a measured number to weigh against ~600 MB per layer.

Two things worth knowing about the shape of these results. At nside 32 nothing
improved at all; the fix targets a serial dependency chain, and at 95 neighbours per
point with one head there is too little work for it to matter. And the grouping in
`e9092eb` bought only 4.5% because it traded occupancy (56% → 44%) for memory
parallelism and the two cancelled — which is evidence that this kernel is **no longer
limited by either**, so further knobs of that kind are not worth pulling.

---

## 5. The single-pass backward (`d0e2b2f`)

The backward walked each neighbourhood twice. Three exact identities collapse that to
one. Writing `p_i = alpha_i / alpha_sum` and `gdotv_i = dy · v_i`:

```
integral = sum_i p_i gdotv_i = dy · out          (since out = sum_i p_i v_i)

dqy      = sum_i p_i (gdotv_i - integral) k_i

scatter multipliers are (gdotv_i - integral) * p_i  for dk,  and  p_i  for dv
```

The first is FlashAttention's `D = rowsum(dO * O)`: the scalar the second walk waited
for costs O(channels) given the forward output, instead of a traversal. The other two
then need the same two per-neighbour quantities, so everything happens in one pass.

This required the forward to **return** its softmax statistics rather than discard
them — as outputs, not state stashed aside, because `setup_context` only sees a custom
op's inputs and outputs. That changed the op signature, its registration, and the
autograd wrapper.

---

## 6. The open problem: bfloat16

The collapse is exact in exact arithmetic but **not in bfloat16**.

`integral = dy · out` reads the *stored* output. In bf16 that carries 8 mantissa bits.
It is then subtracted — `dqy` and `dk` both carry `(gdotv_i - integral)` — and
subtracting two nearly-equal quantities amplifies the rounding. Past the test suite's
3e-2 tolerance, in fact.

The evidence was clean: job `3835420` failed `dk` and `dq` in bf16, while `dv` — the
one gradient that does not involve `integral` — passed, and fp32 and fp16 passed
outright. The two-pass form was immune because it built `gdotv_i` and `integral` in
one fp32 accumulation from the same data.

`3ce0d4e` therefore defaults bf16 to two passes and everything else to one.

**This is the unhelpful way round.** Training runs under bf16, so the collapse
currently buys nothing where it matters.

The fix, if wanted: have the forward also emit an **fp32 copy of its output**, and
compute `integral` from that. Costs as much memory as the output itself — at nside 64
with 1536 channels and batch×time 2, roughly 600 MB per layer — in exchange for
halving the backward's traversals in the dtype training uses. That trade wants
measuring, not assuming.

Switches, all runtime, no rebuild needed:

- `TORCH_HARMONICS_RAGGED_GENERIC=1` — old forward kernel
- `TORCH_HARMONICS_RAGGED_BWD_GENERIC=1` — old backward kernel
- `TORCH_HARMONICS_RAGGED_BWD_TWO_PASS=0|1` — override the dtype default

Build-time, via `NVCC_APPEND_FLAGS`:
`TH_ATTENTION_RAGGED_NB`, `TH_ATTENTION_RAGGED_BWD_NB`,
`TH_ATTENTION_RAGGED_THREADS`, `TH_ATTENTION_RAGGED_BWD_THREADS`.

---

## 7. Immediate next step

Everything on this branch is validated and measured: job 3837708 passed all four
gates and gave the numbers in §4. There is no outstanding correctness question.

The next decision is the **bf16 trade in §6** — whether to spend an fp32 copy of the
forward output, roughly 600 MB per layer at nside 64 with 1536 channels, to get the
single-pass backward in the dtype training actually uses. The fp32 measurement puts
the prize at 1.31×, so the question is now properly priced rather than speculative.

If you want to re-measure after any change, the gate is:

```bash
cd ~/healda_project
sbatch sbatch_scripts/rebuild_and_validate_ragged.sh   # ~50 min, ~40 of it the rebuild
```

and the number to beat is **3.915 ms** for fwd+bwd at nside 64.

---

## 8. Where we stand against FlexAttention

Measured with `grid_vs_flex.py` at bf16, 1536 channels, 16 heads, batch×time 2, both
arms `torch.compile`d, both carrying q/k/v/output projections. Ratios are
flex/ours, so **above 1.00 means we win**.

| grid | radius | nbr/pt | ours | flex | ratio |
|---|---|---|---|---|---|
| level 5 | 2° | 4.4 | 2.36 | 2.43 | **1.03** |
| level 5 | 3° | 8.5 | 2.88 | 3.10 | **1.08** |
| level 5 | 10.46° | 101.4 | 12.33 | 3.17 | 0.26 |
| level 6 | 2° | 15.6 | 14.28 | 12.51 | 0.88 |
| level 6 | 10.46° | 410.6 | 175.22 | 16.80 | 0.10 |
| level 7 | any | 62–1633 | runs | **OOM** | — |

Read this carefully, because the intuitive strategy is backwards:

- **Flex's waste shrinks as the neighbourhood grows.** It works in fixed tiles, so
  more neighbours fill them better — 10.8× wasted pairs at level 5, 4.6× at level 6.
  Scaling up the grid at fixed angular radius walks *toward* its strength. We win only
  at very small neighbourhoods, and only by a few percent.
- **Flex's time is nearly flat in neighbour count.** At level 6 it reads 12.51 ms at
  15, 34 and 95 neighbours per point. It is bound by moving Q/K/V, not by the
  neighbourhood. Shrinking the radius does not hurt it; it just stops helping us.
- **Flex fails outright at level 7** (196k points) — `OutOfMemoryError` trying to
  allocate 4608 GiB, which is exactly `2 × 16 × 196608² × 4` bytes, i.e. it fell back
  off its block-sparse path to dense. Ours runs level 7 in 144 ms to 2.7 s. **This
  looks like a fallback bug, not a fundamental limit, and is worth ten minutes of
  investigation before any strategy is built on it.**
- **Flex is leaner on memory** (3034 vs 4930 MiB at level 6) and **more flexible than
  expected**: quadrature weights cost it 3%, a smooth radial falloff 59%, and
  cross-grid attention works fine. The one thing it cannot do is a trainable
  distance-dependent bias — 292× slower in the backward from atomic contention.

The structural reason we lose at the radii healda actually uses: **Flex runs on the
tensor cores and this kernel does not.** It computes 4.6× more pairs than necessary at
level 6 and still wins by 10×, implying roughly 130× more arithmetic throughput.
Matching it at level 6 while doing only the exact work would need ~95% of the
non-tensor-core peak.

---

## 9. Remaining work, ranked

1. **Validate and time `3ce0d4e`** (§7). Everything else depends on knowing where we
   are.
2. **Decide the bf16 question** (§6): spend ~600 MB per layer on an fp32 output copy
   to get the single-pass backward in the dtype training uses, or don't.
3. **Atomics.** The gradient scatter does one atomic add per channel per neighbour
   into rows that overlapping disks contend for (74% overlap, ~1.2×10¹¹ atomic adds
   per layer at nside 64). Because the neighbourhood is symmetric, this could become a
   gather with no atomics. Two cautions: it wants the *transposed* orientation while
   `dqy` wants the current one, so it may reintroduce a second pass; and the symmetry
   must be verified against the **computed** arc pattern, not assumed from the
   continuous geometry. Whether it is worth it depends on the §7 numbers — if the
   single-pass backward lands near 70-90 ms at nside 64 the atomics are not dominant.
4. **Smaller storage wins.** Arc records are three int32 that fit in int16 even at
   nside 256 (~2×). And along one output ring the ring index and length should be
   nearly constant with only the start offset varying per point (~3×) — speculative,
   check it against the real pattern first.
5. **Tensor cores.** The only route to parity at the radii that matter, and a rewrite.
   The idea with the best odds: tiles of 16 aligned to arcs. Arcs average 12.7
   neighbours at level 6 and are contiguous in memory, and adjacent queries share 74%
   of their neighbours, so a 16-wide tile would waste far less than Flex's 128-wide
   ones. This is essentially writing a FlashAttention specialised for HEALPix.

---

## 10. Traps

Things that cost time here and will cost it again.

**Benchmark fairness bugs bite in both directions and are invisible.** Three separate
ones occurred:
- Flex was given a pixel ordering that did nothing (2760 active tiles where NEST gives
  824), so it did 3.35× its proper work. `grid_vs_flex.py` now prints a `tiles` column
  computed independently — **at level 5 it must read 824**; check it before believing
  any row.
- Flex was run without the four projections the harmonics layer carries, so our side
  paid for eight matmuls its side did not.
- Flex was run uncompiled while ours was compiled, or vice versa. `flex_attention`
  without `torch.compile` falls back to a dense implementation — that fallback is the
  level-7 OOM above.

**`python setup.py build_ext -j N` races.** It builds the four extension modules
concurrently and a link step can run before its own compile finishes — job `3787269`
died with "cannot find disco_helpers.o". Use `MAX_JOBS`, which torch's ninja build
reads, and `rm -rf build` first.

**The image compiles for six GPU architectures** (`8.0 8.6 9.0 10.0 11.0 12.0+PTX`),
which is most of the 40-minute rebuild. Only `disco_helpers.cpp` reads the arch list
(for `BUILD_KPACKED_SM90/SM100`); nothing in `attention/` does. So narrowing
`TORCH_CUDA_ARCH_LIST` to the local GPU should cut the build ~6× and cannot affect the
attention kernels. Not yet tried — **verify the timings are unchanged the first time
you do it.** Note also that those flags look for `10.0a`/`10.3a` and the image's list
says plain `10.0`, so `BUILD_KPACKED_SM100` is currently 0 in every build.

**Tests can pass while testing nothing.** See `fde285b`. Before this branch, no ragged
test exceeded 32 channels, so the register-blocked path's unguarded loops never ran.
When adding a kernel variant, check that some test actually reaches it.

**An equiangular grid is not equal-area.** At matched point counts and a fixed angular
radius it holds ~1.9× the neighbours of HEALPix (193 vs 101 at level 5; 747 vs 411 at
level 6), because its longitudes crowd at the poles. HEALPix measuring faster than the
product grid is that, not a kernel difference. Also: the two grid families store the
neighbour pattern differently — per point when ragged, per *latitude* when not — so
dividing both by `npoints` undercounts the product grid by a factor of `nlon`. The
layer now exposes `neighbours_per_point` to avoid this.

**Slurm here was unreliable for days**, with `scontrol ping` reporting
`** RESTORE SLURMCTLD DAEMON TO SERVICE **`. If the new cluster is healthier, the
`autosubmit_ragged.sh` machinery is unnecessary.
