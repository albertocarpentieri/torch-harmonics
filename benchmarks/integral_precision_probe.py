# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

r"""
Does the single-pass backward need an fp32 stored output, or would fp16 do?

The collapse in the ragged backward reads ``integral = dy . out`` from the *stored*
forward output. In bf16 that output carries 8 mantissa bits, and ``integral`` is then
subtracted -- ``dqy`` and ``dk`` both carry ``(gdotv_i - integral)`` -- so a quantity
with a relative error of ~2^-9 is subtracted from one it is close to, and the error in
the difference is amplified by however tightly the ``gdotv_i`` are clustered. That is
why bf16 is routed to the two-pass form, which never forms ``integral`` independently:
it accumulates it in fp32 from the same ``gdotv_i`` it later subtracts, so the errors
are correlated and cancel.

Recovering the collapse for bf16 means storing the output more precisely, and the
question is how much more. fp32 costs 4 bytes per element on top of the bf16 the rest
of the network consumes; fp16 costs 2. Job 3835420 showed a full fp16 run passing the
gates while bf16 failed, which points at mantissa width -- 10 explicit bits against 7
-- but that is not the same experiment as bf16 activations with an fp16 stored output,
which is what this asks.

Two things could go wrong with fp16 and this separates them: too few mantissa bits, or
too little exponent range. bf16 carries fp32's exponent; fp16 saturates near 65504 and
loses normals below ~6e-5.

No CUDA kernel is involved. The math is written out directly against a random
neighbourhood so the only variable is the precision of the stored output.

    python -m benchmarks.integral_precision_probe
    python -m benchmarks.integral_precision_probe --points 512 --neighbours 411 --channels 96
"""

import argparse

import torch


def reference_and_variants(npoints, nnbr, nchan, dtype, device, seed=0, logit_scale=1.0):
    r"""
    Gradients from the two-pass form and from the collapse at several stored
    precisions, all against the same problem.

    Returns a dict of name -> (dq, dk_mult) where dk_mult is the per-neighbour
    multiplier the scatter would apply; comparing the multiplier rather than the
    scattered result keeps the atomics out of a question that is not about them.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32)

    # one query's neighbourhood, repeated over points; k/v/q/dy in the run dtype, as
    # they would be in memory, but widened at use exactly as the kernel does at load
    q = rnd(npoints, nchan).to(dtype).float()
    k = rnd(npoints, nnbr, nchan).to(dtype).float()
    v = rnd(npoints, nnbr, nchan).to(dtype).float()
    dy = rnd(npoints, nchan).to(dtype).float()

    # softmax over the neighbourhood, in fp32, as the kernel accumulates it.
    #
    # logit_scale sharpens it. This matters because the amount of cancellation is a
    # property of the data, not of the formulation: on random activations the softmax
    # is diffuse, the gdotv_i are spread out, and (gdotv_i - integral) is the same
    # order as its terms, so bf16 survives. A peaked softmax concentrates the weight,
    # out approaches the dominant v, and gdotv for that neighbour approaches integral
    # -- which is when the subtraction loses its leading digits. A trained attention
    # layer is peaked; random data is not, which is why this defaults to sweeping it.
    qdotk = logit_scale * torch.einsum("pc,pnc->pn", q, k)
    alpha = torch.exp(qdotk - qdotk.max(dim=1, keepdim=True).values)
    alpha_sum = alpha.sum(dim=1, keepdim=True)
    p = alpha / alpha_sum

    gdotv = torch.einsum("pc,pnc->pn", dy, v)          # fp32, from the same data
    out_exact = torch.einsum("pn,pnc->pc", p, v)       # the forward's result, fp32

    # Compare the gradients themselves, not the intermediate. (gdotv - integral) is a
    # deviation from a weighted mean, so it crosses zero by construction and a
    # pointwise relative error on it divides by ~0 and blows up even in exact
    # arithmetic. dqy is a sum over the neighbourhood and has a well-defined scale.
    def dq_from(integral):
        return torch.einsum("pn,pnc->pc", p * (gdotv - integral), k)

    results = {}

    # Two-pass: integral accumulated in fp32 from the same gdotv it is subtracted
    # from. This is the reason the two-pass form is immune, and the reference here.
    integral_2p = (p * gdotv).sum(dim=1, keepdim=True)
    results["two-pass"] = (integral_2p, dq_from(integral_2p))

    # Single-pass: integral = dy . out, with out read back at each candidate storage
    # precision. fp64 is the control -- if it does not match the two-pass form then
    # the difference is the formulation, not the storage, and nothing below means
    # what it appears to.
    for name, store in (
        ("single-pass, out fp64", torch.float64),
        ("single-pass, out fp32", torch.float32),
        ("single-pass, out fp16", torch.float16),
        ("single-pass, out bf16", torch.bfloat16),
    ):
        out_stored = out_exact.to(store).float()
        integral = (dy * out_stored).sum(dim=1, keepdim=True)
        results[name] = (integral, dq_from(integral))

    return results, out_exact


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--neighbours", type=int, default=411, help="410.6 is nside 64 at 10.46 degrees")
    p.add_argument("--channels", type=int, default=96)
    p.add_argument("--tol", type=float, default=3e-2, help="the suite's tolerance")
    p.add_argument(
        "--logit-scales",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 10.0, 30.0],
        help="sharpen the softmax. Random activations give a diffuse softmax and "
        "little cancellation; a trained layer is peaked. Sweeping this is what makes "
        "the probe predictive rather than merely ordinal",
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}, {args.points} points x {args.neighbours} neighbours x {args.channels} channels")
    print(f"suite tolerance {args.tol}\n")

    for dtype, label in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
        print(f"=== activations in {label} ===")
        print(f"  {'logit':>6s} {'out fp64':>11s} {'out fp32':>11s} {'out fp16':>11s} {'out bf16':>11s}   fp16/bf16")
        print(f"  {'scale':>6s} {'(control)':>11s}")
        for ls in args.logit_scales:
            results, out_exact = reference_and_variants(
                args.points, args.neighbours, args.channels, dtype, device, logit_scale=ls
            )
            ref_int, ref_dq = results["two-pass"]
            scale = ref_dq.abs().max()

            errs = {}
            for name, (integral, dq) in results.items():
                if name == "two-pass":
                    continue
                errs[name] = ((dq - ref_dq).abs().max() / scale).item()

            def mark(v):
                return f"{v:.2e}{'*' if v >= args.tol else ' '}"

            gain = errs["single-pass, out bf16"] / max(errs["single-pass, out fp16"], 1e-30)
            print(
                f"  {ls:6.0f} {mark(errs['single-pass, out fp64']):>11s} "
                f"{mark(errs['single-pass, out fp32']):>11s} "
                f"{mark(errs['single-pass, out fp16']):>11s} "
                f"{mark(errs['single-pass, out bf16']):>11s}   {gain:8.1f}x"
            )
        print("  (* exceeds the suite tolerance)")
        print()

    print("The dqy column is the one that matters: it is a real gradient, measured")
    print("against its own scale, which is how the suite compares tensors. fp64 is the")
    print("control -- if it is not clean, the difference is the formulation, not storage.")


if __name__ == "__main__":
    main()
