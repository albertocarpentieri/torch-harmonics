# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

r"""
Measure the three properties of the computed ragged pattern that the remaining
storage and atomics work rests on, none of which needs a GPU.

Each one is a claim that is plausible from the continuous geometry and has to be
checked against the pattern the code actually produces:

1. **Arc records fit in int16.** They are stored as three int32 -- ``(ring, start,
   length)``. All three are bounded by the ring sizes, so the question is what the
   real maxima are and at what ``nside`` they would overflow a narrower type.

2. **Along one output ring, only ``start`` varies.** Output points on the same ring
   share a colatitude, so they should see the same input rings with the same arc
   lengths, and differ only in where each arc begins. If that holds, the ring and
   length columns are per-ring data rather than per-point data.

3. **The neighbourhood is symmetric.** ``j`` is a neighbour of ``i`` exactly when
   ``i`` is a neighbour of ``j``. The scatter in the backward could then become a
   gather with no atomics. The inequality being solved is symmetric in the two
   points, but the arc endpoints come from a ceil and a floor of it, so symmetry is
   a property of the discretization and not of the geometry.

Run inside the HEALPix image, or anywhere torch is importable:

    python -m benchmarks.ragged_pattern_report
    python -m benchmarks.ragged_pattern_report --nsides 16 32 64 --radius-deg 10.0
"""

import argparse
import importlib.util
import math
import pathlib
import sys
import types

import torch

INT16_MAX = 2**15 - 1


def _load_pattern_modules():
    r"""
    Import the pattern code without importing the ``torch_harmonics`` package.

    ``torch_harmonics/__init__.py`` pulls in ``attention_helpers``, a compiled
    extension, so a plain import needs the built tree. Everything this report
    touches -- the grid descriptors and the arc precompute -- is pure Python, so on
    an unbuilt checkout the modules are loaded directly under their real names with
    an empty parent package standing in for the real ``__init__``.

    Returns the ``(healpix, neighborhood)`` modules either way.
    """
    try:
        from torch_harmonics import healpix, neighborhood

        return healpix, neighborhood
    except ImportError:
        pass

    root = pathlib.Path(__file__).resolve().parent.parent / "torch_harmonics"
    if not root.is_dir():
        raise SystemExit(f"cannot find torch_harmonics source under {root}")

    parent = types.ModuleType("torch_harmonics")
    parent.__path__ = [str(root)]
    sys.modules["torch_harmonics"] = parent

    # in dependency order, so each module's imports are already registered
    loaded = {}
    for name in ("cache", "partition", "quadrature", "grid", "healpix", "neighborhood"):
        spec = importlib.util.spec_from_file_location(f"torch_harmonics.{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"torch_harmonics.{name}"] = module
        spec.loader.exec_module(module)
        setattr(parent, name, module)
        loaded[name] = module

    print("note: loaded the pattern modules in isolation (no compiled extension present)\n")
    return loaded["healpix"], loaded["neighborhood"]


def report_record_width(arcs, nside):
    """Maxima of the three arc columns, and the int16 headroom they leave."""
    segments = arcs.segments.to(torch.int64)
    maxima = {
        "ring": int(segments[:, 0].max()),
        "start": int(segments[:, 1].max()),
        "length": int(segments[:, 2].max()),
    }
    # every column is bounded by a ring size, and the widest HEALPix ring holds 4*nside
    bound = 4 * nside
    return maxima, bound, all(v <= INT16_MAX for v in maxima.values())


def report_along_ring(arcs, grid):
    r"""
    How much of an arc record is constant along an output ring.

    For each output ring, compare every point's arcs against the first point's:
    the number of arcs, the ring column, and the length column. Only ``start`` is
    expected to differ. Lengths are checked for exact equality and, separately, for
    equality to within one point, since the arc endpoints are a ceil and a floor of
    a longitude that shifts from point to point.
    """
    offsets = arcs.offsets.to(torch.int64)
    segments = arcs.segments.to(torch.int64)
    nlon_per_lat = grid.nlon_per_lat.to(torch.int64)

    rings_total = 0
    rings_same_count = 0
    rings_same_ring_col = 0
    rings_same_length_exact = 0
    rings_same_length_within_1 = 0
    worst_length_delta = 0

    point = 0
    for iring in range(grid.nlat):
        npoints_ring = int(nlon_per_lat[iring])
        first = point
        reference = None
        same_count = same_ring_col = same_len = within_1 = True

        for p in range(first, first + npoints_ring):
            block = segments[offsets[p] : offsets[p + 1]]
            if reference is None:
                reference = block
                continue
            if block.shape[0] != reference.shape[0]:
                same_count = same_ring_col = same_len = within_1 = False
                break
            if not torch.equal(block[:, 0], reference[:, 0]):
                same_ring_col = False
            delta = (block[:, 2] - reference[:, 2]).abs()
            worst = int(delta.max()) if delta.numel() else 0
            worst_length_delta = max(worst_length_delta, worst)
            if worst != 0:
                same_len = False
            if worst > 1:
                within_1 = False

        rings_total += 1
        rings_same_count += same_count
        rings_same_ring_col += same_count and same_ring_col
        rings_same_length_exact += same_count and same_len
        rings_same_length_within_1 += same_count and within_1
        point += npoints_ring

    return {
        "rings": rings_total,
        "same_arc_count": rings_same_count,
        "same_ring_column": rings_same_ring_col,
        "same_length_exact": rings_same_length_exact,
        "same_length_within_1": rings_same_length_within_1,
        "worst_length_delta": worst_length_delta,
    }


def report_symmetry(arcs, npoints):
    r"""
    Whether the computed relation equals its own transpose.

    Compares the multiset of ``(row, col)`` pairs with the multiset of ``(col,
    row)`` pairs by sorting a single composite key, which is exact and needs no
    sparse matrix. Returns the count of pairs present in one direction only.
    """
    col_idx, row_off = arcs.to_csr()
    counts = row_off[1:] - row_off[:-1]
    rows = torch.repeat_interleave(torch.arange(npoints, dtype=torch.int64), counts)
    cols = col_idx.to(torch.int64)

    forward = torch.sort(rows * npoints + cols).values
    transposed = torch.sort(cols * npoints + rows).values

    if forward.numel() != transposed.numel():
        asymmetric = abs(forward.numel() - transposed.numel())
    else:
        asymmetric = int((forward != transposed).sum())

    # a self-pair is symmetric trivially; report it so the nnz reconciles
    self_pairs = int((rows == cols).sum())
    return {
        "nnz": int(rows.numel()),
        "asymmetric_entries": asymmetric,
        "self_pairs": self_pairs,
        "neighbours_per_point": rows.numel() / npoints,
    }


def report_storage(arcs, grid):
    r"""
    Bytes the pattern costs now, and under the two narrowings, from real counts.

    Three schemes, in increasing order of how much they exploit:

    ``current``
        three int32 per arc per output point.
    ``int16``
        the same three columns narrowed, which needs only the maxima to fit.
    ``per_ring``
        ring and length promoted to a per-output-ring table, leaving ``start`` as
        the only per-point column, plus two bits per arc for the +-1 length
        correction that the measurement above shows is genuinely needed.
    """
    nsegs = int(arcs.segments.shape[0])
    offsets = arcs.offsets.to(torch.int64)
    arcs_per_point = offsets[1:] - offsets[:-1]
    nlon_per_lat = grid.nlon_per_lat.to(torch.int64)

    # one row of the per-ring table per (output ring, arc slot). The slot count is
    # constant along a ring -- section 2 is what establishes that -- so it can be
    # read off that ring's first point.
    point = 0
    table_rows = 0
    for iring in range(grid.nlat):
        table_rows += int(arcs_per_point[point])
        point += int(nlon_per_lat[iring])

    current = nsegs * 3 * 4
    narrowed = nsegs * 3 * 2
    per_ring = nsegs * 2 + (nsegs * 2 + 7) // 8 + table_rows * 2 * 2

    return {
        "nsegs": nsegs,
        "table_rows": table_rows,
        "current": current,
        "int16": narrowed,
        "per_ring": per_ring,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nsides", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--radius-deg", type=float, default=10.0)
    # symmetry expands the pattern to a column list, which is the memory-hungry part;
    # the storage questions only need the arcs, so the two have separate ceilings
    parser.add_argument("--symmetry-max-nside", type=int, default=64)
    args = parser.parse_args()

    healpix, neighborhood = _load_pattern_modules()
    radius = math.radians(args.radius_deg)

    print(f"radius {args.radius_deg} deg ({radius:.6f} rad)")
    print(f"int16 holds up to {INT16_MAX}\n")

    print("=" * 78)
    print("1. arc record width  --  are three int32 wider than they need to be?")
    print("=" * 78)
    print(f"{'nside':>6} {'nsegs':>12} {'max ring':>10} {'max start':>10} {'max length':>11} {'4*nside':>9}  int16?")
    widths = {}
    for nside in args.nsides:
        grid = healpix.HealpixGrid(nside=nside)
        arcs = neighborhood.precompute_neighborhood_arcs_s2(grid, grid, radius)
        maxima, bound, fits = report_record_width(arcs, nside)
        widths[nside] = (grid, arcs)
        print(
            f"{nside:>6} {arcs.segments.shape[0]:>12} {maxima['ring']:>10} {maxima['start']:>10} "
            f"{maxima['length']:>11} {bound:>9}  {'yes' if fits else 'NO'}"
        )
    print()
    print(f"  every column is bounded by the widest ring, 4*nside, so int16 suffices")
    print(f"  while 4*nside <= {INT16_MAX}, i.e. nside <= {INT16_MAX // 4}.")
    print(f"  At nside 256 the bound is {4 * 256}, which leaves {INT16_MAX // (4 * 256)}x headroom.")
    print()

    print("=" * 78)
    print("2. along one output ring  --  is only 'start' per-point?")
    print("=" * 78)
    for nside in args.nsides:
        grid, arcs = widths[nside]
        stats = report_along_ring(arcs, grid)
        rings = stats["rings"]
        print(f"  nside {nside}:  {rings} output rings")
        print(f"    same number of arcs per point : {stats['same_arc_count']}/{rings}")
        print(f"    same ring column              : {stats['same_ring_column']}/{rings}")
        print(f"    same length column, exactly   : {stats['same_length_exact']}/{rings}")
        print(f"    same length column, +-1       : {stats['same_length_within_1']}/{rings}")
        print(f"    worst length deviation        : {stats['worst_length_delta']}")
    print()

    print("=" * 78)
    print("3. symmetry  --  could the gradient scatter become a gather?")
    print("=" * 78)
    for nside in args.nsides:
        if nside > args.symmetry_max_nside:
            print(f"  nside {nside}: skipped (above --symmetry-max-nside)")
            continue
        grid, arcs = widths[nside]
        stats = report_symmetry(arcs, grid.npoints)
        verdict = "SYMMETRIC" if stats["asymmetric_entries"] == 0 else "NOT symmetric"
        print(
            f"  nside {nside}: nnz {stats['nnz']}, {stats['neighbours_per_point']:.1f} nbr/pt, "
            f"{stats['asymmetric_entries']} one-directional  ->  {verdict}"
        )
    print()

    print("=" * 78)
    print("4. what the two narrowings are worth, from the counts above")
    print("=" * 78)
    print(f"{'nside':>6} {'current':>12} {'int16':>12} {'per-ring':>12} {'int16 x':>9} {'per-ring x':>11}")
    for nside in args.nsides:
        grid, arcs = widths[nside]
        s = report_storage(arcs, grid)
        print(
            f"{nside:>6} {s['current'] / 2**20:>11.2f}M {s['int16'] / 2**20:>11.2f}M "
            f"{s['per_ring'] / 2**20:>11.2f}M {s['current'] / s['int16']:>8.2f}x "
            f"{s['current'] / s['per_ring']:>10.2f}x"
        )
    print()


if __name__ == "__main__":
    main()
