"""Two things the test suite does not currently answer.

1. Accuracy in bfloat16. Every numerical test in tests/test_attention_ragged.py
   runs in float32 with TF32 disabled, so the precision we actually train in has
   never been compared against anything. The fp32 kernel *is* validated there
   against an independent dense masked-softmax oracle to atol 1e-5, so it serves
   as the reference here and the question reduces to what autocast costs.

2. Whether a HEALPix grid can attend to a product grid. NeighborhoodAttentionS2
   refuses the mix, but the refusal is about tensor layout, not geometry:
   precompute_neighborhood_arcs_s2 documents itself as working on any isolatitude
   grid. If the arcs build across the mix, the gap is plumbing rather than maths.
"""

import math
import sys

import torch

from torch_harmonics import HealpixGrid, NeighborhoodAttentionS2, as_grid, precompute_neighborhood_arcs_s2


def _err(got, ref):
    """Relative error in the norm, plus the worst single element."""
    got, ref = got.double(), ref.double()
    denom = ref.norm().clamp_min(1e-30)
    return (got - ref).norm().item() / denom.item(), (got - ref).abs().max().item()


def precision_probe(device):
    print("=" * 100)
    print("BFLOAT16 ACCURACY  --  autocast against the float32 kernel the tests validate")
    print("=" * 100)
    hdr = f"{'nside':>6} {'cutoff deg':>11} {'nbr/pt':>8} {'chan':>5} {'heads':>6} {'out rel':>10} {'out max':>10} {'dq rel':>10} {'dk rel':>10} {'dv rel':>10}"
    print(hdr)
    print("-" * len(hdr))

    for nside, cutoff_deg in ((8, None), (16, None), (16, 10.0), (32, 10.0)):
        grid = HealpixGrid(nside=nside)
        cutoff = math.radians(cutoff_deg) if cutoff_deg is not None else None
        channels, heads, batch = 64, 4, 1

        torch.manual_seed(0)
        model = NeighborhoodAttentionS2(
            grid_in=grid, grid_out=grid, in_channels=channels, num_heads=heads, theta_cutoff=cutoff
        ).to(device)
        nbr = model.psi_col_idx.numel() / grid.npoints

        def run(enabled):
            torch.manual_seed(1)
            qkv = [
                torch.randn(batch, channels, grid.npoints, device=device, requires_grad=True) for _ in range(3)
            ]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
                out = model(*qkv)
            # the upstream gradient is the same draw for both passes, so the only
            # difference between them is the precision of the arithmetic
            torch.manual_seed(2)
            out.backward(torch.randn(out.shape, device=device, dtype=out.dtype))
            return out.detach().float(), [t.grad.detach().float() for t in qkv]

        ref_out, ref_grads = run(False)
        bf_out, bf_grads = run(True)

        out_rel, out_max = _err(bf_out, ref_out)
        grad_rel = [_err(g, r)[0] for g, r in zip(bf_grads, ref_grads)]
        label = f"{math.degrees(model.theta_cutoff):.3f}"
        print(
            f"{nside:>6} {label:>11} {nbr:>8.1f} {channels:>5} {heads:>6} "
            f"{out_rel:>10.2e} {out_max:>10.2e} {grad_rel[0]:>10.2e} {grad_rel[1]:>10.2e} {grad_rel[2]:>10.2e}"
        )

    print()
    print("'rel' is ||bf16 - fp32|| / ||fp32|| over the whole tensor; bf16 carries 8 mantissa")
    print("bits, so ~4e-3 is the floor a single rounding imposes and anything near it means")
    print("the accumulation is not compounding error.")


def crossgrid_probe():
    print()
    print("=" * 100)
    print("CROSS-GRID  --  can the neighbourhood be built between a ragged and a regular grid?")
    print("=" * 100)

    hpx = HealpixGrid(nside=8)
    eqa = as_grid("equiangular", (32, 64))
    cutoff = 2 * math.radians(10.0)

    for name, gin, gout in (
        ("healpix -> healpix", hpx, hpx),
        ("equiangular -> healpix", eqa, hpx),
        ("healpix -> equiangular", hpx, eqa),
        ("equiangular -> equiangular", eqa, eqa),
    ):
        try:
            arcs = precompute_neighborhood_arcs_s2(gin, gout, cutoff)
            nseg = int(arcs.segments.shape[0])
            print(
                f"  {name:<28} arcs OK   segments={nseg:>9}  "
                f"arcs/pt={nseg / gout.npoints:>6.1f}  npoints_out={gout.npoints:>7}"
            )
        except Exception as exc:
            print(f"  {name:<28} FAILED    {type(exc).__name__}: {exc}")

    print()
    print("  and the layer that would consume them:")
    for name, gin, gout in (("equiangular -> healpix", eqa, hpx), ("healpix -> equiangular", hpx, eqa)):
        try:
            NeighborhoodAttentionS2(grid_in=gin, grid_out=gout, in_channels=8, num_heads=1)
            print(f"  {name:<28} layer OK")
        except Exception as exc:
            print(f"  {name:<28} layer REFUSED  {type(exc).__name__}: {str(exc)[:90]}")


def main():
    if not torch.cuda.is_available():
        print("no CUDA device; the precision probe needs one")
        crossgrid_probe()
        return 0
    print(f"device: {torch.cuda.get_device_name(0)}")
    precision_probe("cuda")
    crossgrid_probe()
    return 0


if __name__ == "__main__":
    sys.exit(main())
