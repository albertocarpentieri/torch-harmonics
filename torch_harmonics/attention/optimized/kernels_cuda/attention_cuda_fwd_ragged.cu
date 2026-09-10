// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Forward neighborhood attention on a RAGGED isolatitude grid (HEALPix).
//
// Relation to the product-grid kernels in attention_cuda_fwd.cu
// -------------------------------------------------------------
// Those kernels store one neighbour list per output LATITUDE and slide it to the
// current longitude with an integer p-shift, wip = wi + pscale*wo. That works
// because rotating a product grid about the polar axis maps it onto itself. On a
// ragged grid it does not: rotating a ring by one of its own points maps that ring
// onto itself but not the rings above and below, whose point counts differ. So the
// pattern here is keyed per output POINT and the p-shift is gone.
//
// The trade is index reuse for footprint. The product-grid kernel reads one column
// list per latitude and reuses it across every longitude in the ring; this one reads
// a distinct list per point, so the pattern is npoints_out/nlat_out times larger --
// about 3*nside on HEALPix. In exchange the addressing gets strictly simpler: no
// pscale, no (ho, wo) decomposition, no wrap_lon on a shifted index.
//
// What carries over unchanged is the part that matters for speed. An arc is a run of
// consecutive points of one input ring, and RING ordering lays a ring out
// contiguously, so k/v reads along an arc are still stride-1 and the ring's
// quadrature weight is still an per-arc constant. The column advances by counting
// and wraps with a compare-and-subtract, so there is no integer division per
// neighbour -- which is the optimization the arc form exists for (see the commentary
// in attention_cuda_fwd.cu: emulated 64-bit division dominated the original
// col_idx-based inner loop).
//
// Only the generic (any channel count) variant is provided. The product-grid file
// also carries a register-blocked "special" variant for channel counts that fit in
// MAX_LOCAL_ARR_LEN; adding the ragged equivalent is a pure performance question and
// is deliberately left until the benchmark says it is worth the code.

#include "attention_cuda.cuh"
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include "c10/core/MemoryFormat.h"

#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>
#include <cfloat>

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"

#define THREADS (64)

namespace attention_kernels
{

    // One warp per output point: warp lanes span the channel dimension, so the
    // per-neighbour q.k reduction is a warp reduction and the running softmax state
    // lives in registers replicated across the warp.
    template <int BDIM_X, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X) void s2_attn_fwd_ragged_generic_vec_k(
        int nheads,    // no. of attention heads packed along the channel dim
        int nchan_in,  // no. of STORAGE_T elements along channel dim, per head
        int nchan_out, // no. of STORAGE_T elements along channel dim, per head
        int64_t npoints_in, int64_t npoints_out, const STORAGE_T *__restrict__ kx,
        const STORAGE_T *__restrict__ vx, const STORAGE_T *__restrict__ qy, const int32_t *__restrict__ seg,
        const int32_t *__restrict__ seg_off, const int64_t *__restrict__ ring_base,
        const int64_t *__restrict__ ring_size, const float *__restrict__ ring_weights, STORAGE_T *__restrict__ y)
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        extern __shared__ __align__(sizeof(float4)) float shext[];
        COMPUTE_T *shy = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * nchan_out;

        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        // leading dimensions: elements between adjacent spatial points
        const int64_t ldi = int64_t(nheads) * nchan_in;
        const int64_t ldo = int64_t(nheads) * nchan_out;

        const int64_t ipoint = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;

        if (ipoint >= npoints_out) { return; }

        const int tidx = threadIdx.x;

        // No row_idx indirection. The product-grid kernel sorts output rows by
        // neighbour count to keep long rows off the tail of the grid; on a geodesic
        // neighbourhood of a near-uniform grid the counts barely vary (HEALPix pixels
        // are equal-area by construction), so the sort would cost a device-side sort
        // per call to balance nothing.
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { shy[chan] = __vset<COMPUTE_T>(0.f); }

        kx += int64_t(batch) * npoints_in * ldi + int64_t(head) * nchan_in;
        vx += int64_t(batch) * npoints_in * ldo + int64_t(head) * nchan_out;

        qy += int64_t(batch) * npoints_out * ldi + int64_t(head) * nchan_in + ipoint * ldi;
        y += int64_t(batch) * npoints_out * ldo + int64_t(head) * nchan_out + ipoint * ldo;

        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        const int seg_beg = seg_off[ipoint];
        const int seg_end = seg_off[ipoint + 1];

        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int iring = seg[3 * sg + 0];
            const int lo = seg[3 * sg + 1];
            const int len = seg[3 * sg + 2];

            // constant along the arc: every point of a ring carries the same
            // quadrature weight, so this is hoisted exactly as quad_weights[hi] is
            // in the product-grid kernel
            const float qw = ring_weights[iring];

            // An arc is a run of consecutive points of one ring, and a ring is
            // contiguous in RING order, so the flat column is just the ring's base
            // plus an offset that counts up and wraps at the ring's end.
            const int64_t ring_lo = ring_base[iring];
            const int64_t ring_hi = ring_lo + ring_size[iring];

            int64_t col = ring_lo + lo;

            for (int j = 0; j < len; j++) {

                const STORAGE_T *_kx = kx + col * ldi;
                const STORAGE_T *_vx = vx + col * ldo;

                COMPUTE_T qdotkv = __vset<COMPUTE_T>(0.f);

                for (int chan = tidx; chan < nchan_in; chan += WARP_SIZE) {
                    qdotkv = __vadd(qdotkv, __vmul(vload(qy, chan), vload(_kx, chan)));
                }

                float qdotk = __warp_sum(__vred(qdotkv));

                // online (streaming) softmax, identical to the product-grid kernel:
                // rescale the running numerator and denominator whenever a new
                // neighbour raises the running maximum
                const float qdotk_max_tmp = max(qdotk_max, qdotk);
                const float alpha = expf(qdotk - qdotk_max_tmp) * qw;
                const float exp_save = expf(qdotk_max - qdotk_max_tmp);

                alpha_sum = alpha + alpha_sum * exp_save;

                for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) {
                    shy[chan] = __vadd(__vscale(exp_save, shy[chan]), __vscale(alpha, vload(_vx, chan)));
                }
                qdotk_max = qdotk_max_tmp;

                // next point in the arc; wraps at most once, so a compare-and-subtract
                // replaces the modulo the arc encoding is designed to avoid
                if (++col == ring_hi) { col = ring_lo; }
            }
        }

        alpha_sum = 1.0f / alpha_sum;
        for (int chan = tidx; chan < nchan_out; chan += WARP_SIZE) { vstore(y, chan, __vscale(alpha_sum, shy[chan])); }

        return;
    }

    template <typename STORAGE_T>
    static void launch_gen_attn_fwd_ragged(int batch_size, int nheads, int nchans_in, int nchans_out,
                                           int64_t npoints_in, int64_t npoints_out, const STORAGE_T *__restrict__ _kxp,
                                           const STORAGE_T *__restrict__ _vxp, const STORAGE_T *__restrict__ _qyp,
                                           const int32_t *_seg, const int32_t *_seg_off, const int64_t *_ring_base,
                                           const int64_t *_ring_size, const float *_ring_weights,
                                           STORAGE_T *__restrict__ _yp, cudaStream_t stream)
    {
        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        // one block row per (batch, head) pair
        dim3 grid(DIV_UP(npoints_out, block.y), batch_size * nheads);

        // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T.
        // sized from the per-head channel count, so it does not scale with nheads
        size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * nchans_out * block.y;

        s2_attn_fwd_ragged_generic_vec_k<THREADS><<<grid, block, shsize, stream>>>(
            nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _seg, _seg_off, _ring_base,
            _ring_size, _ring_weights, _yp);
        CHECK_ERROR("s2_attn_fwd_ragged_generic_vec_k");

        return;
    }

    // NHWC ABI, flattened: kx, vx, qy are physically (B, npoints, num_heads * nchan)
    // and contiguous, with the spatial axes of the product-grid ABI collapsed into
    // one. Layout is never inferred from strides -- the caller states it by
    // construction (see attention/_layout.py). Heads stay packed along the channel
    // dimension for the same reason as on the product grids: folding them into the
    // batch dimension is not free in a channel-innermost layout.
    torch::Tensor s2_attention_fwd_ragged_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor ring_weights,
                                               at::Tensor psi_seg, at::Tensor psi_seg_off, at::Tensor ring_base,
                                               at::Tensor ring_size, int64_t num_heads, int64_t npoints_out)
    {
        CHECK_CUDA_INPUT_TENSOR(kx);
        CHECK_CUDA_INPUT_TENSOR(vx);
        CHECK_CUDA_INPUT_TENSOR(qy);
        CHECK_CUDA_TENSOR(ring_weights);
        CHECK_CUDA_TENSOR(psi_seg);
        CHECK_CUDA_TENSOR(psi_seg_off);
        CHECK_CUDA_TENSOR(ring_base);
        CHECK_CUDA_TENSOR(ring_size);

        TORCH_CHECK(kx.dim() == 3, "kx must be (B, npoints_in, num_heads * C_k), got ", kx.dim(), " dims");
        TORCH_CHECK(vx.dim() == 3, "vx must be (B, npoints_in, num_heads * C_v), got ", vx.dim(), " dims");
        TORCH_CHECK(qy.dim() == 3, "qy must be (B, npoints_out, num_heads * C_k), got ", qy.dim(), " dims");

        TORCH_CHECK(num_heads >= 1, "num_heads must be positive, got ", num_heads);
        TORCH_CHECK(qy.size(2) % num_heads == 0, "q/k channel count (", qy.size(2),
                    ") must be divisible by num_heads (", num_heads, ")");
        TORCH_CHECK(vx.size(2) % num_heads == 0, "v channel count (", vx.size(2), ") must be divisible by num_heads (",
                    num_heads, ")");

        TORCH_CHECK(qy.size(1) == npoints_out, "qy has ", qy.size(1), " points but npoints_out is ", npoints_out);
        TORCH_CHECK(kx.size(1) == vx.size(1), "kx has ", kx.size(1), " points but vx has ", vx.size(1));
        TORCH_CHECK(psi_seg_off.size(0) == npoints_out + 1, "seg_off must have npoints_out + 1 = ", npoints_out + 1,
                    " entries, got ", psi_seg_off.size(0));
        TORCH_CHECK(psi_seg.dim() == 2 && psi_seg.size(1) == 3, "seg must be (nsegs, 3)");
        TORCH_CHECK(ring_base.size(0) == ring_size.size(0), "ring_base and ring_size must agree in length, got ",
                    ring_base.size(0), " and ", ring_size.size(0));
        TORCH_CHECK(ring_weights.size(0) == ring_base.size(0), "ring_weights must have one entry per input ring, got ",
                    ring_weights.size(0), " for ", ring_base.size(0), " rings");

        // Every activation must share one dtype: the dispatch below selects a single
        // scalar_t from qy and the launcher reinterpret_casts k/v/q to it, so a
        // mismatched input would be reinterpreted rather than converted.
        TORCH_CHECK(kx.scalar_type() == qy.scalar_type(), "k dtype (", kx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");
        TORCH_CHECK(vx.scalar_type() == qy.scalar_type(), "v dtype (", vx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");

        // per-head channel counts; the packed extent is num_heads times these
        const int nchans_in = qy.size(2) / num_heads; // or kx.size(2) / num_heads
        const int nchans_out = vx.size(2) / num_heads;

        const int batch_size = kx.size(0);
        const int64_t npoints_in = kx.size(1);

        auto qy_type = qy.dtype();
        const int64_t out_dims[] = {batch_size, npoints_out, int64_t(nchans_out) * num_heads};
        torch::Tensor y;

        // Activations stay in their native dtype and y is allocated in it, so there is
        // no whole-tensor fp32 copy and the read bandwidth for fp16/bf16 is halved.
        // The kernel widens to fp32 at load and narrows back at store; compute and
        // softmax accumulation are fp32 in-kernel either way.
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qy.scalar_type(), "s2_attention_fwd_ragged_cuda", [&] {
            using storage_t = scalar_t;

            auto stream = at::cuda::getCurrentCUDAStream().stream();

            torch::Tensor y_nhwc = torch::empty(out_dims, kx.options()); // native dtype

            launch_gen_attn_fwd_ragged<storage_t>(
                batch_size, num_heads, nchans_in, nchans_out, npoints_in, npoints_out,
                reinterpret_cast<const storage_t *>(kx.data_ptr()), reinterpret_cast<const storage_t *>(vx.data_ptr()),
                reinterpret_cast<const storage_t *>(qy.data_ptr()), reinterpret_cast<const int32_t *>(psi_seg.data_ptr()),
                reinterpret_cast<const int32_t *>(psi_seg_off.data_ptr()),
                reinterpret_cast<const int64_t *>(ring_base.data_ptr()),
                reinterpret_cast<const int64_t *>(ring_size.data_ptr()),
                reinterpret_cast<const float *>(ring_weights.data_ptr()),
                reinterpret_cast<storage_t *>(y_nhwc.data_ptr()), stream);

            y = y_nhwc;
        });

        y = y.to(qy_type);

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return y;
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m) { m.impl("forward_ragged", &s2_attention_fwd_ragged_cuda); }

} // namespace attention_kernels
