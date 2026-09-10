# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
What the ragged HEALPix neighbourhood costs, in memory and in time.

Two questions the standard harness cannot answer, hence a separate script:

Memory. A ragged grid has no rotational self-similarity -- rotating a ring by one
of its own points leaves the rings above and below misaligned, because they hold
a different number of points -- so the p-shift the product-grid kernels use to
slide one neighbour list along a latitude does not exist here. The pattern must be
keyed per output POINT instead of per output RING, which multiplies its size by
npoints/nrings ~ 3*nside. This measures that factor, and measures how much of it
the arc encoding takes back: arcs store (ring, start, len) per run of consecutive
neighbours rather than one index per neighbour, so the compression is the mean arc
length. The two encodings are reported side by side because the module currently
builds BOTH -- the kernels consume the arcs, the torch reference consumes the CSR
expansion -- and on a GPU run the CSR half is dead weight.

Time, against the torch reference. Whether the arc kernel is worth having at all at
healda's working resolution. This is the weak baseline -- the reference materialises
a gather over every neighbour column -- so a win here says the kernel is worth
compiling, not that it is fast.

Time, against the equiangular counterpart. The comparison that actually prices
raggedness. The per-neighbour arithmetic in the two CUDA kernels is the same; what
the ragged one gives up is pattern reuse, since the product-grid kernel reads one
list per latitude and reuses it across the ring while this one reads a distinct list
per point. Held at matched point count and matched cutoff, that bandwidth cost is
what the healpix/equiangular ratio measures.

Run under the container, on a GPU node:
    python -m benchmarks.healpix_ragged_report
"""

import argparse
import gc
import math

import torch

from torch_harmonics import EquiangularGrid, HealpixGrid, NeighborhoodAttentionS2
from torch_harmonics.neighborhood import precompute_neighborhood_arcs_s2, precompute_neighborhood_csr_s2

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

# Every buffer either setup path registers to hold the neighbourhood pattern. Summing
# whichever of these a layer actually has is how the two grids are compared on pattern
# memory without the report needing to know which precompute ran: the product grid
# keys by ring into (row, col, roff), the ragged grid keys by point into arcs plus a
# CSR expansion for the torch reference.
_PATTERN_BUFFERS = (
    "psi_row_idx",
    "psi_col_idx",
    "psi_roff_idx",
    "psi_seg",
    "psi_seg_off",
    "psi_ring_base",
    "psi_ring_size",
)


def _mib(nbytes: int) -> float:
    return nbytes / (1024.0 * 1024.0)


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def matched_equiangular(nside: int) -> EquiangularGrid:
    r"""
    The equiangular grid closest in point count to ``HealpixGrid(nside)``.

    HEALPix has :math:`12 n_{side}^2` points; an equiangular grid with the usual
    :math:`n_{lon} = 2 n_{lat}` has :math:`2 n_{lat}^2`. Equating the two gives
    :math:`n_{lat} = \sqrt{6} n_{side}`, which is not an integer, so the match is
    within rounding rather than exact -- the achieved point count is reported
    alongside the HEALPix one so the residual is visible rather than assumed.

    Matching total points is the right control here because the question is whether
    the ragged kernel pays for losing pattern reuse, and that cost scales with the
    number of output points each having its own neighbour list. It does mean the two
    grids differ in how those points are distributed: equiangular rings crowd
    together near the poles, so at a fixed geodesic radius its polar points have
    more neighbours than HEALPix's do.
    """
    nlat = max(2, int(round(math.sqrt(6.0) * nside)))
    return EquiangularGrid(nlat=nlat, nlon=2 * nlat)


def layer_pattern_mib(layer: NeighborhoodAttentionS2) -> float:
    """MiB of neighbourhood pattern a constructed layer holds, whichever encoding it uses."""
    return _mib(sum(_nbytes(b) for n, b in layer.named_buffers() if n in _PATTERN_BUFFERS))


def pattern_report(nside: int) -> dict:
    """Sizes of the two pattern encodings for one HEALPix grid."""
    grid = HealpixGrid(nside=nside)
    npix = grid.npoints
    nrings = grid.nlat

    # theta_cutoff is left at the module default so these numbers describe what a
    # layer actually allocates, rather than a radius chosen to flatter the report
    layer_cutoff = NeighborhoodAttentionS2(in_channels=8, num_heads=1, grid_in=grid, grid_out=grid).theta_cutoff

    arcs = precompute_neighborhood_arcs_s2(grid, grid, layer_cutoff)
    col_idx, row_off = precompute_neighborhood_csr_s2(grid, grid, layer_cutoff)

    nsegs = arcs.segments.shape[0]
    nnz = col_idx.numel()

    arc_bytes = _nbytes(arcs.segments) + _nbytes(arcs.offsets) + _nbytes(arcs.ring_base) + _nbytes(arcs.ring_size)
    csr_bytes = _nbytes(col_idx) + _nbytes(row_off)

    return {
        "nside": nside,
        "npix": npix,
        "nrings": nrings,
        # the structural penalty of raggedness: a product grid stores one neighbour
        # list per ring and slides it, a ragged grid needs one per point
        "point_vs_ring_keying": npix / nrings,
        "theta_cutoff": layer_cutoff,
        "neighbours_total": nnz,
        "neighbours_per_point": nnz / npix,
        "arcs_total": nsegs,
        "arcs_per_point": nsegs / npix,
        "mean_arc_len": nnz / max(nsegs, 1),
        "arc_mib": _mib(arc_bytes),
        "csr_mib": _mib(csr_bytes),
        "csr_over_arc": csr_bytes / max(arc_bytes, 1),
    }


def _time_ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def timing_report(grid, cutoff: float, batch: int, channels: int, num_heads: int, dtype, iters: int, warmup: int) -> dict:
    """
    Forward and backward, optimized kernel vs torch reference, on one grid.

    Takes the grid and the cutoff rather than an nside so the same body serves both
    the HEALPix arm and its equiangular counterpart. Passing the cutoff in is what
    makes the two comparable: left to default, each grid would size its own radius
    from its own spacing and the two arms would be attending over different
    neighbourhood areas.
    """
    device = torch.device("cuda")

    out = {
        "grid": grid.grid_type,
        "npoints": grid.npoints,
        "shape": "x".join(str(d) for d in grid.spatial_shape),
        "batch": batch,
        "channels": channels,
        "dtype": str(dtype).replace("torch.", ""),
        "cutoff": cutoff,
    }

    for label, optimized in (("kernel", True), ("reference", False)):
        layer = NeighborhoodAttentionS2(
            in_channels=channels,
            num_heads=num_heads,
            grid_in=grid,
            grid_out=grid,
            theta_cutoff=cutoff,
            optimized_kernel=optimized,
        ).to(device)

        # a module built with optimized_kernel=True still falls back when the
        # extension lacks the ops, so report what was actually exercised
        out[f"{label}_selected_optimized"] = bool(layer.optimized_kernel)
        out[f"{label}_pattern_mib"] = layer_pattern_mib(layer)

        # spatial_shape is (nlat, nlon) on a product grid and (npoints,) on a ragged
        # one, which is exactly the input layout each path expects
        x = torch.randn(batch, channels, *grid.spatial_shape, device=device, dtype=dtype, requires_grad=True)

        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()

            out[f"{label}_fwd_ms"] = _time_ms(lambda: layer(x), iters, warmup)

            # backward is timed on its own, so the graph is rebuilt untimed each pass
            def bwd():
                if x.grad is not None:
                    x.grad = None
                y = layer(x)
                y.backward(torch.ones_like(y))

            out[f"{label}_bwd_ms"] = _time_ms(bwd, iters, warmup)

            torch.cuda.synchronize()
            out[f"{label}_peak_mib"] = _mib(torch.cuda.max_memory_allocated() - base)

        except torch.cuda.OutOfMemoryError:
            # the reference materialises a gather over every neighbour column, so it
            # is the side expected to run out first; record it rather than abort, the
            # fact that it cannot run at this size is itself the result
            out[f"{label}_fwd_ms"] = float("nan")
            out[f"{label}_bwd_ms"] = float("nan")
            out[f"{label}_peak_mib"] = float("nan")
            out[f"{label}_oom"] = True

        del layer, x
        gc.collect()
        torch.cuda.empty_cache()

    for phase in ("fwd", "bwd"):
        ref, ker = out.get(f"reference_{phase}_ms"), out.get(f"kernel_{phase}_ms")
        out[f"{phase}_speedup"] = (ref / ker) if (ref and ker and ker == ker and ref == ref) else float("nan")

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsides", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--timing-nsides", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dtypes", type=str, nargs="+", default=["float32", "bfloat16"], choices=list(_DTYPES))
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--skip-equiangular",
        action="store_true",
        help="only time HEALPix, dropping the matched product-grid arm and the head-to-head table",
    )
    args = parser.parse_args()

    print("=" * 108)
    print("PATTERN SIZE  --  what raggedness costs, and what the arc encoding takes back")
    print("=" * 108)
    hdr = (
        f"{'nside':>6} {'npix':>8} {'rings':>6} {'pt/ring':>8} {'cutoff':>8} "
        f"{'nbr/pt':>8} {'arc/pt':>8} {'arclen':>7} {'arc MiB':>9} {'csr MiB':>9} {'csr/arc':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for nside in args.nsides:
        r = pattern_report(nside)
        print(
            f"{r['nside']:>6} {r['npix']:>8} {r['nrings']:>6} {r['point_vs_ring_keying']:>8.1f} "
            f"{r['theta_cutoff']:>8.4f} {r['neighbours_per_point']:>8.1f} {r['arcs_per_point']:>8.1f} "
            f"{r['mean_arc_len']:>7.2f} {r['arc_mib']:>9.2f} {r['csr_mib']:>9.2f} {r['csr_over_arc']:>8.2f}"
        )

    print()
    print("pt/ring is npoints/nrings: the factor by which keying the pattern per output point")
    print("rather than per output ring inflates it, which is the price of raggedness (~3*nside).")
    print("csr/arc is how much the arc encoding saves over the CSR column list; the module")
    print("currently allocates both, and only the CSR half is unused by the CUDA path.")

    if not torch.cuda.is_available():
        print("\nno CUDA device: skipping the timing section")
        return

    print()
    print("=" * 122)
    print(f"TIME AND PEAK MEMORY  --  optimized kernel vs torch reference on {torch.cuda.get_device_name(0)}")
    print("=" * 122)
    hdr = (
        f"{'nside':>6} {'grid':>12} {'pts':>8} {'dtype':>9} {'opt?':>5} {'ker fwd':>9} {'ref fwd':>9} {'fwd x':>7} "
        f"{'ker bwd':>9} {'ref bwd':>9} {'bwd x':>7} {'ker MiB':>9} {'ref MiB':>9} {'patt MiB':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    head_to_head = {}
    for dtype_name in args.dtypes:
        for nside in args.timing_nsides:
            # both arms attend over the radius a HEALPix layer would choose for itself,
            # so the HEALPix arm is the untouched default and the equiangular arm is the
            # one adapted to match it
            hpx = HealpixGrid(nside=nside)
            cutoff = NeighborhoodAttentionS2(in_channels=8, num_heads=1, grid_in=hpx, grid_out=hpx).theta_cutoff

            arms = [hpx] if args.skip_equiangular else [hpx, matched_equiangular(nside)]
            for grid in arms:
                r = timing_report(
                    grid, cutoff, args.batch, args.channels, args.num_heads, _DTYPES[dtype_name], args.iters, args.warmup
                )
                head_to_head[(dtype_name, nside, r["grid"])] = r
                print(
                    f"{nside:>6} {r['grid']:>12} {r['npoints']:>8} {r['dtype']:>9} "
                    f"{str(r['kernel_selected_optimized']):>5} "
                    f"{r['kernel_fwd_ms']:>9.3f} {r['reference_fwd_ms']:>9.3f} {r['fwd_speedup']:>7.2f} "
                    f"{r['kernel_bwd_ms']:>9.3f} {r['reference_bwd_ms']:>9.3f} {r['bwd_speedup']:>7.2f} "
                    f"{r['kernel_peak_mib']:>9.1f} {r['reference_peak_mib']:>9.1f} {r['kernel_pattern_mib']:>9.2f}"
                )

    print()
    print("times are ms/iter; 'x' columns are reference/kernel, so >1 means the kernel wins.")
    print("opt? is whether the layer actually selected the optimized handle -- if False the")
    print("two columns are the same code path and the ratio is meaningless.")
    print("patt MiB is the pattern the layer holds: arcs+CSR when ragged, row/col/roff when not.")

    if args.skip_equiangular:
        return

    print()
    print("=" * 122)
    print("RAGGED vs PRODUCT GRID  --  matched point count, same cutoff, kernel against kernel")
    print("=" * 122)
    hdr = (
        f"{'nside':>6} {'dtype':>9} {'hpx pts':>9} {'eq shape':>10} {'eq pts':>8} {'hpx fwd':>9} {'eq fwd':>9} "
        f"{'fwd h/e':>8} {'hpx bwd':>9} {'eq bwd':>9} {'bwd h/e':>8} {'hpx patt':>9} {'eq patt':>9} {'patt h/e':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for dtype_name in args.dtypes:
        for nside in args.timing_nsides:
            h = head_to_head.get((dtype_name, nside, "healpix"))
            e = head_to_head.get((dtype_name, nside, "equiangular"))
            if h is None or e is None:
                continue

            def ratio(a, b):
                return (a / b) if (a == a and b == b and b) else float("nan")

            print(
                f"{nside:>6} {dtype_name:>9} {h['npoints']:>9} {e['shape']:>10} {e['npoints']:>8} "
                f"{h['kernel_fwd_ms']:>9.3f} {e['kernel_fwd_ms']:>9.3f} "
                f"{ratio(h['kernel_fwd_ms'], e['kernel_fwd_ms']):>8.2f} "
                f"{h['kernel_bwd_ms']:>9.3f} {e['kernel_bwd_ms']:>9.3f} "
                f"{ratio(h['kernel_bwd_ms'], e['kernel_bwd_ms']):>8.2f} "
                f"{h['kernel_pattern_mib']:>9.2f} {e['kernel_pattern_mib']:>9.2f} "
                f"{ratio(h['kernel_pattern_mib'], e['kernel_pattern_mib']):>9.2f}"
            )

    print()
    print("'h/e' columns are healpix/equiangular, so >1 means the ragged kernel is the more")
    print("expensive of the two -- the opposite convention to the 'x' columns above, because")
    print("here neither side is a reference implementation and the question is what raggedness")
    print("costs against a product grid of the same size, not whether a kernel beats torch.")
    print("Point counts differ by rounding (sqrt(6)*nside is not an integer); both are shown.")


if __name__ == "__main__":
    main()
