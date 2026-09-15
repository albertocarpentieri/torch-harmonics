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

// Backward neighborhood attention on a RAGGED isolatitude grid (HEALPix).
//
// Relation to the product-grid kernel in attention_cuda_bwd.cu
// ------------------------------------------------------------
// The mathematics and the scatter strategy are taken over verbatim; only the
// addressing changes, in exactly the same way as in attention_cuda_fwd_ragged.cu.
// The pattern is keyed per output POINT rather than per output LATITUDE, so there
// is no p-shift (wip = wi + pscale*wo) and no (ho, wo) decomposition -- a neighbour
// is located as ring_base[iring] + offset, counting along the arc.
//
// Why the scatter strategy needs no rethinking
// --------------------------------------------
// dk/dv are scatter-accumulated: neighbourhoods of distinct output points overlap,
// so several output points contribute to the same input point. The product-grid
// kernel resolves that with atomicAdd into fp32 buffers, and nothing about that
// choice depends on the pattern being a product grid -- it depends only on the
// pattern being many-to-many, which is equally true here. Raggedness changes which
// input point an output point lands on, not how many of them collide. So this is a
// port, not a redesign: no transposed (input-keyed) pattern is built, and no second
// precompute is needed.
//
// Two passes over the same arcs
// -----------------------------
// Pass 1 accumulates the running-softmax statistics (alpha_sum, qdotk_max, the two
// per-channel reductions and the scalar alpha_vw_) and writes dqy. Pass 2 replays the arcs with the final
// qdotk_max to scatter dk/dv. The replay is what avoids materialising the per-
// neighbour alphas, which on a HEALPix neighbourhood would be a far larger array
// than on a product grid, since the pattern here is npoints_out/nlat_out ~ 3*nside
// times bigger. Recomputing q.k is cheaper than storing it.
//
// Both variants of the product-grid file are now provided, for the reason the
// forward gave: the generic kernel keeps one neighbour in flight at a time and
// re-reads qy and dy from shared memory on every one of them, and it measures like
// it -- the ragged backward is 24 ms of the 33 ms fwd+bwd at HEALPix level 5, at 1.6
// TFLOP/s, so bound by the latency of a dependency chain 102 neighbours long rather
// than by arithmetic or bandwidth. s2_attn_bwd_ragged_special_vec_k is the port of
// s2_attn_bwd_special_vec_k, which moves the four per-channel reductions and the
// staged qy/dy into registers; shared memory survives only for the pre-9.0 atomic
// epilogue. It applies when the per-head channel count fits NLOC registers per lane
// and q/k and v agree on it, and the generic kernel remains for everything else and
// as an escape hatch, since only the last of NLOC registers is bounds-checked and
// that argument depends on NLOC being exactly DIV_UP(nchan, BDIM_X).

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
#include <cstdlib>

#include "cudamacro.h"
#include "attention_cuda_utils.cuh"

// Threads per block, and so warps per block, since BDIM_X is a warp. Overridable for
// the reason given in attention_cuda_fwd_ragged.cu, and more so here: warps in a
// block take consecutive output points, adjacent points share 74% of their
// neighbours by measurement, and the backward walks each neighbourhood twice, so it
// has twice as much to gain from L1 holding that overlap. Kept separate from the
// forward's knob because the two kernels have different register pressure.
#ifndef TH_ATTENTION_RAGGED_BWD_THREADS
#define TH_ATTENTION_RAGGED_BWD_THREADS (64)
#endif
#define THREADS (TH_ATTENTION_RAGGED_BWD_THREADS)

// Largest number of COMPUTE_T registers per lane the special kernel will hold for
// one accumulator. 16 matches MAX_LOCAL_ARR_LEN in attention_cuda_bwd.cu, so with a
// 32-lane BDIM_X it covers up to 512 channels per head; HEALDA's dit-5B runs 96.
//
// The kernel holds four accumulators of that length, so it is the register-hungry
// half of the pair, and occupancy here is register-limited. ptxas -v for the real
// workload, nchan 96 (NLOC 3): 56 registers, no spill, 36 warps/SM on both sm_90a
// and sm_100a, which is the same register count the generic kernel needs while doing
// strictly less work per neighbour. The one instantiation that spills is nchan 64 in
// fp32 on sm_90a, by 8 bytes, where ptxas prefers 48 registers and 42 warps/SM to 56
// and 36. See benchmarks/ptxas_register_report.py.
//
// There is deliberately no neighbours-per-group constant to go with it, as there is
// in the forward: grouping would move the online softmax's rescale from per
// neighbour to per group, and that changes the arithmetic of a gradient, so it is a
// change to make against a numerical check rather than alongside a port.
#define MAX_LOCAL_ARR_LEN_RAGGED_BWD (16)

namespace attention_kernels
{

    // One warp per output point, as in the forward. STORAGE_T is the global-memory
    // element type of the inputs (kx/vx/qy/dy); COMPUTE_T is the arithmetic type and
    // the type of the gradient OUTPUTS dkx/dvx/dqy. The gradients stay fp32 because
    // dkx/dvx are atomically scatter-accumulated and reduced-precision atomics would
    // lose precision; the wrapper narrows them back at the end.
    template <int BDIM_X, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X) void s2_attn_bwd_ragged_generic_vec_k(
        int nheads,     // no. of attention heads packed along the channel dim
        int nchans_in,  // no. of STORAGE_T elements along channel dim, per head
        int nchans_out, // no. of STORAGE_T elements along channel dim, per head
        int64_t npoints_in, int64_t npoints_out,
        const STORAGE_T *__restrict__ kx, // [batch][npoints_in][nheads * nchan_in]
        const STORAGE_T *__restrict__ vx, // [batch][npoints_in][nheads * nchan_out]
        const STORAGE_T *__restrict__ qy, // [batch][npoints_out][nheads * nchan_in]
        const STORAGE_T *__restrict__ dy, // [batch][npoints_out][nheads * nchan_out]
        const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off,
        const int64_t *__restrict__ ring_base, const int64_t *__restrict__ ring_size,
        const float *__restrict__ ring_weights,
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dkx, // [batch][npoints_in][nheads * nchan_in]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dvx, // [batch][npoints_in][nheads * nchan_out]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dqy) // [batch][npoints_out][nheads * nchan_in]
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        extern __shared__ __align__(sizeof(float4)) float shext[];

        // Four per-warp arrays: 3 * nchans_in + nchans_out.
        //
        // The product-grid kernels carry a fifth, alpha_vw_, one entry per channel.
        // Its recurrence is alpha_vw_ = alpha_vw_ * max_correction + ainz_gdotv and
        // both of those are warp-uniform scalars, so every entry holds the same
        // number and always has: it is a scalar accumulator stored nchans_in times.
        // It is only ever consumed as a multiplier of sh_alpha_k__ when dqy is
        // written, so a single register does the same work. Dropping it removes a
        // third of pass 1's shared-memory read-modify-writes and a quarter of the
        // per-warp footprint. (attention_cuda_bwd.cu replicates it in both its
        // generic and its register-blocked kernel; the redundancy is not ragged's.)
        COMPUTE_T *sh_alpha_k__ = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * (nchans_in * 3 + nchans_out);
        COMPUTE_T *sh_alpha_kvw = sh_alpha_k__ + nchans_in;

        COMPUTE_T *sh_dy = sh_alpha_kvw + nchans_in;
        COMPUTE_T *sh_qy = sh_dy + nchans_out;

        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        // leading dimensions: elements between adjacent spatial points
        const int64_t ldi = int64_t(nheads) * nchans_in;
        const int64_t ldo = int64_t(nheads) * nchans_out;

        const int64_t ipoint = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;

        if (ipoint >= npoints_out) { return; }

        const int tidx = threadIdx.x;

        // No row_idx indirection, for the reason given in the forward: the geodesic
        // neighbourhood of an equal-area grid has near-uniform neighbour counts, so
        // sorting output rows by length would cost a device-side sort per call to
        // balance nothing.

        // offset input tensors
        kx += int64_t(batch) * npoints_in * ldi + int64_t(head) * nchans_in;
        vx += int64_t(batch) * npoints_in * ldo + int64_t(head) * nchans_out;

        qy += int64_t(batch) * npoints_out * ldi + int64_t(head) * nchans_in + ipoint * ldi;
        dy += int64_t(batch) * npoints_out * ldo + int64_t(head) * nchans_out + ipoint * ldo;

        // offset output tensors (same packed layout as their inputs)
        dkx += int64_t(batch) * npoints_in * ldi + int64_t(head) * nchans_in;
        dvx += int64_t(batch) * npoints_in * ldo + int64_t(head) * nchans_out;
        dqy += int64_t(batch) * npoints_out * ldi + int64_t(head) * nchans_in + ipoint * ldi;

        // zero/init shared memory
        for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
            sh_alpha_k__[chan] = __vset<COMPUTE_T>(0.0f);
            sh_alpha_kvw[chan] = __vset<COMPUTE_T>(0.0f);

            sh_qy[chan] = vload(qy, chan);
        }
        for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) { sh_dy[chan] = vload(dy, chan); }

#if __CUDA_ARCH__ < 900
        // for architectures < 9.0, sh_dy and sh_qy will be read as individual floats
        // at the end of the kernel, which breaks the assumption that each COMPUTE_T
        // location is written to and read by the same thread throughout the kernel,
        // in the case COMPUTE_T==float4
        if constexpr (std::is_same<COMPUTE_T, float4>::value) { __syncwarp(); }
#endif

        // for dkx, dvx, dqy
        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        // for dkx
        float integral = 0.0f;

        // the scalar that replaces the replicated sh_alpha_vw_ array
        float alpha_vw_ = 0.0f;

        const int seg_beg = seg_off[ipoint];
        const int seg_end = seg_off[ipoint + 1];

        // Pass 1: accumulate alpha_sum, integral and the shared reductions, along
        // with a progressively computed qdotk_max.
        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int iring = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];

            // constant along the arc: every point of a ring carries the same
            // quadrature weight
            const float qw_seg = ring_weights[iring];

            // a ring is contiguous in RING order, so the flat column is the ring's
            // base plus an offset that counts up and wraps at the ring's end
            const int64_t ring_lo = ring_base[iring];
            const int64_t ring_hi = ring_lo + ring_size[iring];

            int64_t col = ring_lo + seg_lo;

            for (int j = 0; j < seg_len; j++) {

                const STORAGE_T *_kx = kx + col * ldi;
                const STORAGE_T *_vx = vx + col * ldo;

                COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
                COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

                for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
                    qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], vload(_kx, chan)));
                }
                for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) {
                    gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], vload(_vx, chan)));
                }

                const float qdotk = __warp_sum(__vred(qdotk_v));
                const float gdotv = __warp_sum(__vred(gdotv_v));

                const float qdotk_max_tmp = max(qdotk_max, qdotk);
                const float alpha_inz = expf(qdotk - qdotk_max_tmp) * qw_seg;
                const float max_correction = expf(qdotk_max - qdotk_max_tmp);
                alpha_sum = alpha_sum * max_correction + alpha_inz;

                integral = integral * max_correction + alpha_inz * gdotv;

                const float ainz_gdotv = alpha_inz * gdotv;

                // same recurrence the per-channel array used to run, once
                alpha_vw_ = alpha_vw_ * max_correction + ainz_gdotv;

                for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {

                    const COMPUTE_T kxval = vload(_kx, chan);

                    sh_alpha_k__[chan] = __vadd(__vscale(max_correction, sh_alpha_k__[chan]), __vscale(alpha_inz, kxval));
                    sh_alpha_kvw[chan]
                        = __vadd(__vscale(max_correction, sh_alpha_kvw[chan]), __vscale(ainz_gdotv, kxval));
                }
                qdotk_max = qdotk_max_tmp;

                // next point in the arc; wraps at most once
                if (++col == ring_hi) { col = ring_lo; }
            }
        }

        const float alpha_sum_inv = 1.0f / alpha_sum;

        integral *= alpha_sum_inv;

        // Write dqy (fp32 output)
        for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {

            // __vscale by the scalar, where this used to __vmul by a vector every
            // entry of which held that scalar
            dqy[chan] = __vscale(
                alpha_sum_inv * alpha_sum_inv,
                __vsub(__vscale(alpha_sum, sh_alpha_kvw[chan]), __vscale(alpha_vw_, sh_alpha_k__[chan])));
        }

        // Pass 2: replay the arcs with the final qdotk_max to scatter dk/dv.
        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int iring = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];
            const float qw_seg = ring_weights[iring];

            const int64_t ring_lo = ring_base[iring];
            const int64_t ring_hi = ring_lo + ring_size[iring];

            int64_t col = ring_lo + seg_lo;

            for (int j = 0; j < seg_len; j++) {

                const STORAGE_T *_kx = kx + col * ldi;
                const STORAGE_T *_vx = vx + col * ldo;

                COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
                COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

                for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
                    qdotk_v = __vadd(qdotk_v, __vmul(sh_qy[chan], vload(_kx, chan)));
                }
                for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) {
                    gdotv_v = __vadd(gdotv_v, __vmul(sh_dy[chan], vload(_vx, chan)));
                }

                const float qdotk = __warp_sum(__vred(qdotk_v));
                const float gdotv = __warp_sum(__vred(gdotv_v));

                const float alpha_inz = expf(qdotk - qdotk_max) * qw_seg;

                // _dkx / _dvx are COMPUTE_T (fp32) gradient buffers, accumulated
                // atomically: neighbourhoods overlap, so several output points hit
                // the same input point.
                COMPUTE_T *_dkx = dkx + col * ldi;
                COMPUTE_T *_dvx = dvx + col * ldo;

                const float alpha_mul = alpha_inz * alpha_sum_inv;

                const float scale_fact_qy = (gdotv - integral) * alpha_mul;
                const float scale_fact_dy = alpha_mul;

                // float4, 128-bit atomics are only supported by devices of compute
                // capability 9.x+, so on older devices we resort to 32-bit atomics

#if __CUDA_ARCH__ < 900
                // to use 32-bit operations on consecutive addresses
                float *sh_qy_scl = reinterpret_cast<float *>(sh_qy);
                float *sh_dy_scl = reinterpret_cast<float *>(sh_dy);

                float *_dkx_scl = reinterpret_cast<float *>(_dkx);
                float *_dvx_scl = reinterpret_cast<float *>(_dvx);

                constexpr int VEC_SIZE = sizeof(COMPUTE_T) / sizeof(float);

                // 32-bit, consecutive atomics to glmem;
                // strided atomics results in a severe slowdown
                for (int chan = tidx; chan < nchans_in * VEC_SIZE; chan += WARP_SIZE) {
                    atomicAdd(_dkx_scl + chan, scale_fact_qy * sh_qy_scl[chan]);
                }
                for (int chan = tidx; chan < nchans_out * VEC_SIZE; chan += WARP_SIZE) {
                    atomicAdd(_dvx_scl + chan, scale_fact_dy * sh_dy_scl[chan]);
                }
#else
                // 128-bit, consecutive atomics to glmem
                for (int chan = tidx; chan < nchans_in; chan += WARP_SIZE) {
                    atomicAdd(_dkx + chan, __vscale(scale_fact_qy, sh_qy[chan]));
                }
                for (int chan = tidx; chan < nchans_out; chan += WARP_SIZE) {
                    atomicAdd(_dvx + chan, __vscale(scale_fact_dy, sh_dy[chan]));
                }
#endif
                if (++col == ring_hi) { col = ring_lo; }
            }
        }

        return;
    }

    // Register-blocked counterpart of the kernel above, and the port of
    // s2_attn_bwd_special_vec_k in attention_cuda_bwd.cu. Everything that differs is
    // addressing, and it gets simpler: the neighbour list is keyed per output point,
    // and an arc is a contiguous run of one ring, so there is no (ho, wo)
    // decomposition, no pscale, no wrap_lon and no row_idx indirection -- just a
    // column that counts and wraps at the ring end.
    //
    // NLOC must be exactly DIV_UP(nchan, BDIM_X). The unrolled loops below leave every
    // register but the last unguarded, which is sound only under that equality: for
    // i <= NLOC-2, i*BDIM_X + tidx <= (NLOC-1)*BDIM_X - 1 < nchan. The launcher
    // enforces it, and requires nchan_in == nchan_out so one NLOC serves both. That
    // equality is also why there is no CHOUT_AS_IN template parameter here: it selects
    // between taking the dy side through the same unrolled loops as the qy side or
    // through a runtime-bounded loop over shared memory, and only the former is
    // reachable once the channel counts are required to agree.
    template <int BDIM_X, int BDIM_Y, int NLOC, typename STORAGE_T>
    __global__ __launch_bounds__(BDIM_X *BDIM_Y) void s2_attn_bwd_ragged_special_vec_k(
        int nheads,    // no. of attention heads packed along the channel dim
        int nchan_in,  // no. of STORAGE_T elements along channel dim, per head
        int nchan_out, // no. of STORAGE_T elements along channel dim, per head
        int64_t npoints_in, int64_t npoints_out,
        const STORAGE_T *__restrict__ kx, // [batch][npoints_in][nheads * nchan_in]
        const STORAGE_T *__restrict__ vx, // [batch][npoints_in][nheads * nchan_out]
        const STORAGE_T *__restrict__ qy, // [batch][npoints_out][nheads * nchan_in]
        const STORAGE_T *__restrict__ dy, // [batch][npoints_out][nheads * nchan_out]
        const int32_t *__restrict__ seg, const int32_t *__restrict__ seg_off, const int64_t *__restrict__ ring_base,
        const int64_t *__restrict__ ring_size, const float *__restrict__ ring_weights,
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dkx, // [batch][npoints_in][nheads * nchan_in]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dvx, // [batch][npoints_in][nheads * nchan_out]
        typename vec_traits<STORAGE_T>::compute_t *__restrict__ dqy) // [batch][npoints_out][nheads * nchan_in]
    {
        using COMPUTE_T = typename vec_traits<STORAGE_T>::compute_t;

        static_assert(BDIM_X == WARP_SIZE, "the ragged special kernel reduces with __warp_sum");
        static_assert(NLOC >= 1);

        constexpr int NLOC_M1 = NLOC - 1;

        const int tidx = threadIdx.x;

        const int bh = blockIdx.y;
        const int batch = bh / nheads;
        const int head = bh - (batch * nheads);

        const int64_t ldi = int64_t(nheads) * nchan_in;
        const int64_t ldo = int64_t(nheads) * nchan_out;

        const int64_t ipoint = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;

        if (ipoint >= npoints_out) { return; }

        extern __shared__ __align__(sizeof(float4)) float shext[];

        // sh_dy[nchan_out], sh_qy[nchan_in]. Two arrays per warp where the generic
        // kernel needs four: the per-channel reductions are in registers now, and
        // these two survive only because the pre-9.0 epilogue undoes the tidx offset
        // and reads them as individual floats, which registers cannot serve.
        COMPUTE_T *sh_dy = reinterpret_cast<COMPUTE_T *>(shext) + threadIdx.y * (nchan_in + nchan_out) + tidx;
        COMPUTE_T *sh_qy = sh_dy + nchan_out;

        // for dqy. The product grid carries a third array, loc_vw_[NLOC], every entry
        // of which always holds the same number; it is a scalar here for the reason
        // spelled out in the generic kernel above.
        COMPUTE_T loc_k__[NLOC];
        COMPUTE_T loc_kvw[NLOC];

        // Register copies of this thread's slice of qy / dy. Both are loop-invariant
        // across neighbours, and each thread only ever touches its own
        // (tidx + i*BDIM_X) slots, so re-reading them from shared once per neighbour
        // was pure overhead -- and this kernel reads them twice per neighbour, once in
        // each pass.
        COMPUTE_T loc_qy[NLOC];
        COMPUTE_T loc_dy[NLOC];
#pragma unroll
        for (int i = 0; i < NLOC; i++) {
            loc_qy[i] = __vset<COMPUTE_T>(0.0f);
            loc_dy[i] = __vset<COMPUTE_T>(0.0f);
        }

        // the lane's channel offset folded into the base pointers, as on the product
        // grid, so the inner loops index by register rather than by channel
        kx += int64_t(batch) * npoints_in * ldi + int64_t(head) * nchan_in + tidx;
        vx += int64_t(batch) * npoints_in * ldo + int64_t(head) * nchan_out + tidx;

        qy += int64_t(batch) * npoints_out * ldi + int64_t(head) * nchan_in + ipoint * ldi + tidx;
        dy += int64_t(batch) * npoints_out * ldo + int64_t(head) * nchan_out + ipoint * ldo + tidx;

        dkx += int64_t(batch) * npoints_in * ldi + int64_t(head) * nchan_in + tidx;
        dvx += int64_t(batch) * npoints_in * ldo + int64_t(head) * nchan_out + tidx;
        dqy += int64_t(batch) * npoints_out * ldi + int64_t(head) * nchan_in + ipoint * ldi + tidx;

#pragma unroll
        for (int i = 0; i < NLOC; i++) {
            loc_k__[i] = __vset<COMPUTE_T>(0.0f);
            loc_kvw[i] = __vset<COMPUTE_T>(0.0f);
        }

#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) {
            loc_qy[i] = vload(qy, i * BDIM_X);
            sh_qy[i * BDIM_X] = loc_qy[i];
        }
        if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
            loc_qy[NLOC_M1] = vload(qy, NLOC_M1 * BDIM_X);
            sh_qy[NLOC_M1 * BDIM_X] = loc_qy[NLOC_M1];
        }

#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) {
            loc_dy[i] = vload(dy, i * BDIM_X);
            sh_dy[i * BDIM_X] = loc_dy[i];
        }
        if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
            loc_dy[NLOC_M1] = vload(dy, NLOC_M1 * BDIM_X);
            sh_dy[NLOC_M1 * BDIM_X] = loc_dy[NLOC_M1];
        }

#if __CUDA_ARCH__ < 900
        // for architectures < 9.0, sh_dy and sh_qy will be read as individual floats
        // at the end of the kernel, which breaks the assumption that each COMPUTE_T
        // location is written to and read by the same thread throughout the kernel,
        // in the case COMPUTE_T==float4
        if constexpr (std::is_same<COMPUTE_T, float4>::value) { __syncwarp(); }
#endif

        // for dkx, dvx, dqy
        float alpha_sum = 0.0f;
        float qdotk_max = -FLT_MAX;

        // for dkx
        float integral = 0.0f;

        // the scalar that replaces the replicated loc_vw_ array
        float alpha_vw_ = 0.0f;

        const int seg_beg = seg_off[ipoint];
        const int seg_end = seg_off[ipoint + 1];

        // Pass 1: accumulate alpha_sum, integral and the register reductions, along
        // with a progressively computed qdotk_max.
        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int iring = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];

            // constant along the arc: every point of a ring carries the same
            // quadrature weight
            const float qw_seg = ring_weights[iring];

            // a ring is contiguous in RING order, so the flat column is the ring's
            // base plus an offset that counts up and wraps at the ring's end
            const int64_t ring_lo = ring_base[iring];
            const int64_t ring_hi = ring_lo + ring_size[iring];

            int64_t col = ring_lo + seg_lo;

            for (int j = 0; j < seg_len; j++) {

                const STORAGE_T *_kx = kx + col * ldi;
                const STORAGE_T *_vx = vx + col * ldo;

                COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
                COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[i], vload(_kx, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[NLOC_M1], vload(_kx, NLOC_M1 * BDIM_X)));
                }
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    gdotv_v = __vadd(gdotv_v, __vmul(loc_dy[i], vload(_vx, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    gdotv_v = __vadd(gdotv_v, __vmul(loc_dy[NLOC_M1], vload(_vx, NLOC_M1 * BDIM_X)));
                }

                const float qdotk = __warp_sum(__vred(qdotk_v));
                const float gdotv = __warp_sum(__vred(gdotv_v));

                const float qdotk_max_tmp = max(qdotk_max, qdotk);
                const float alpha_inz = expf(qdotk - qdotk_max_tmp) * qw_seg;
                const float max_correction = expf(qdotk_max - qdotk_max_tmp);

                alpha_sum = alpha_sum * max_correction + alpha_inz;
                integral = integral * max_correction + alpha_inz * gdotv;

                const float ainz_gdotv = alpha_inz * gdotv;

                // same recurrence the per-channel array used to run, once
                alpha_vw_ = alpha_vw_ * max_correction + ainz_gdotv;

#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    const COMPUTE_T kxval = vload(_kx, i * BDIM_X);
                    loc_k__[i] = __vadd(__vscale(max_correction, loc_k__[i]), __vscale(alpha_inz, kxval));
                    loc_kvw[i] = __vadd(__vscale(max_correction, loc_kvw[i]), __vscale(ainz_gdotv, kxval));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    const COMPUTE_T kxval = vload(_kx, NLOC_M1 * BDIM_X);
                    loc_k__[NLOC_M1] = __vadd(__vscale(max_correction, loc_k__[NLOC_M1]), __vscale(alpha_inz, kxval));
                    loc_kvw[NLOC_M1] = __vadd(__vscale(max_correction, loc_kvw[NLOC_M1]), __vscale(ainz_gdotv, kxval));
                }

                qdotk_max = qdotk_max_tmp;

                // next point in the arc; wraps at most once
                if (++col == ring_hi) { col = ring_lo; }
            }
        }

        const float alpha_sum_inv = 1.0f / alpha_sum;

        integral *= alpha_sum_inv;

        // Write dqy (fp32 output)
        const float alpha_sum_inv_sq = alpha_sum_inv * alpha_sum_inv;

#pragma unroll
        for (int i = 0; i < NLOC_M1; i++) {
            // __vscale by the scalar, where the product grid __vmul-s by a vector every
            // entry of which holds that scalar
            dqy[i * BDIM_X]
                = __vscale(alpha_sum_inv_sq, __vsub(__vscale(alpha_sum, loc_kvw[i]), __vscale(alpha_vw_, loc_k__[i])));
        }
        if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
            dqy[NLOC_M1 * BDIM_X] = __vscale(
                alpha_sum_inv_sq, __vsub(__vscale(alpha_sum, loc_kvw[NLOC_M1]), __vscale(alpha_vw_, loc_k__[NLOC_M1])));
        }

        // Pass 2: replay the arcs with the final qdotk_max to scatter dk/dv.
        for (int sg = seg_beg; sg < seg_end; sg++) {

            const int iring = seg[3 * sg + 0];
            const int seg_lo = seg[3 * sg + 1];
            const int seg_len = seg[3 * sg + 2];
            const float qw_seg = ring_weights[iring];

            const int64_t ring_lo = ring_base[iring];
            const int64_t ring_hi = ring_lo + ring_size[iring];

            int64_t col = ring_lo + seg_lo;

            for (int j = 0; j < seg_len; j++) {

                const STORAGE_T *_kx = kx + col * ldi;
                const STORAGE_T *_vx = vx + col * ldo;

                COMPUTE_T qdotk_v = __vset<COMPUTE_T>(0.0f);
                COMPUTE_T gdotv_v = __vset<COMPUTE_T>(0.0f);

#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[i], vload(_kx, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    qdotk_v = __vadd(qdotk_v, __vmul(loc_qy[NLOC_M1], vload(_kx, NLOC_M1 * BDIM_X)));
                }
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) {
                    gdotv_v = __vadd(gdotv_v, __vmul(loc_dy[i], vload(_vx, i * BDIM_X)));
                }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    gdotv_v = __vadd(gdotv_v, __vmul(loc_dy[NLOC_M1], vload(_vx, NLOC_M1 * BDIM_X)));
                }

                const float qdotk = __warp_sum(__vred(qdotk_v));
                const float gdotv = __warp_sum(__vred(gdotv_v));

                const float alpha_inz = expf(qdotk - qdotk_max) * qw_seg;

                // _dkx / _dvx are COMPUTE_T (fp32) gradient buffers, accumulated
                // atomically: neighbourhoods overlap, so several output points hit
                // the same input point.
                COMPUTE_T *_dkx = dkx + col * ldi;
                COMPUTE_T *_dvx = dvx + col * ldo;

                const float alpha_mul = alpha_inz * alpha_sum_inv;

                const float scale_fact_qy = (gdotv - integral) * alpha_mul;
                const float scale_fact_dy = alpha_mul;

                // float4, 128-bit atomics are only supported by devices of compute
                // capability 9.x+, so on older devices we resort to 32-bit atomics

#if __CUDA_ARCH__ < 900
                constexpr int VEC_SIZE = sizeof(COMPUTE_T) / sizeof(float);

                // to use 32-bit operations on consecutive addresses; the registers
                // cannot serve this, which is why sh_qy / sh_dy exist
                float *sh_qy_scl = reinterpret_cast<float *>(sh_qy);
                float *sh_dy_scl = reinterpret_cast<float *>(sh_dy);

                float *_dkx_scl = reinterpret_cast<float *>(_dkx);
                float *_dvx_scl = reinterpret_cast<float *>(_dvx);

                sh_qy_scl -= tidx * VEC_SIZE;
                sh_dy_scl -= tidx * VEC_SIZE;
                _dkx_scl -= tidx * VEC_SIZE;
                _dvx_scl -= tidx * VEC_SIZE;

                // 32-bit, consecutive atomics to glmem;
                // strided atomics results in a severe slowdown
                for (int chan = tidx; chan < nchan_in * VEC_SIZE; chan += BDIM_X) {
                    atomicAdd(_dkx_scl + chan, scale_fact_qy * sh_qy_scl[chan]);
                }
                for (int chan = tidx; chan < nchan_out * VEC_SIZE; chan += BDIM_X) {
                    atomicAdd(_dvx_scl + chan, scale_fact_dy * sh_dy_scl[chan]);
                }
#else
                // 128-bit, consecutive atomics to glmem, straight out of the registers
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) { atomicAdd(_dkx + i * BDIM_X, __vscale(scale_fact_qy, loc_qy[i])); }
                if (NLOC_M1 * BDIM_X + tidx < nchan_in) {
                    atomicAdd(_dkx + NLOC_M1 * BDIM_X, __vscale(scale_fact_qy, loc_qy[NLOC_M1]));
                }
#pragma unroll
                for (int i = 0; i < NLOC_M1; i++) { atomicAdd(_dvx + i * BDIM_X, __vscale(scale_fact_dy, loc_dy[i])); }
                if (NLOC_M1 * BDIM_X + tidx < nchan_out) {
                    atomicAdd(_dvx + NLOC_M1 * BDIM_X, __vscale(scale_fact_dy, loc_dy[NLOC_M1]));
                }
#endif
                if (++col == ring_hi) { col = ring_lo; }
            }
        }

        return;
    }

    // Resolve NLOC, which has to be a compile-time constant, from the runtime channel
    // count by walking the supported range. Mirrors launch_spc_attn_fwd_ragged, and so
    // launch_spc_attn_bwd, minus its CHOUT_AS_IN branch (see the kernel).
    template <int BDIM_X, int BDIM_Y, int CUR_LOC, int MAX_LOC, typename STORAGE_T>
    static void launch_spc_attn_bwd_ragged(int nloc, int batch_size, int nheads, int nchans_in, int nchans_out,
                                           int64_t npoints_in, int64_t npoints_out, const STORAGE_T *_kxp,
                                           const STORAGE_T *_vxp, const STORAGE_T *_qyp, const STORAGE_T *_dyp,
                                           const int32_t *_seg, const int32_t *_seg_off, const int64_t *_ring_base,
                                           const int64_t *_ring_size, const float *_ring_weights,
                                           typename vec_traits<STORAGE_T>::compute_t *_dkxp,
                                           typename vec_traits<STORAGE_T>::compute_t *_dvxp,
                                           typename vec_traits<STORAGE_T>::compute_t *_dqyp, cudaStream_t stream)
    {
        if constexpr (CUR_LOC > MAX_LOC) {
            TORCH_CHECK(false, "ragged special attention kernel reached nloc ", nloc, " above its bound ", MAX_LOC);
            return;
        } else {
            if (CUR_LOC == nloc) {
                dim3 block(BDIM_X, BDIM_Y);
                dim3 grid(DIV_UP(npoints_out, block.y), batch_size * nheads);

                // 2 arrays per warp, against the generic kernel's 4: only qy and dy are
                // staged, and only for the pre-9.0 epilogue
                size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * (nchans_in + nchans_out) * block.y;

                s2_attn_bwd_ragged_special_vec_k<BDIM_X, BDIM_Y, CUR_LOC><<<grid, block, shsize, stream>>>(
                    nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _dyp, _seg, _seg_off,
                    _ring_base, _ring_size, _ring_weights, _dkxp, _dvxp, _dqyp);
                CHECK_ERROR("s2_attn_bwd_ragged_special_vec_k");
                return;
            }
            launch_spc_attn_bwd_ragged<BDIM_X, BDIM_Y, CUR_LOC + 1, MAX_LOC, STORAGE_T>(
                nloc, batch_size, nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _dyp, _seg,
                _seg_off, _ring_base, _ring_size, _ring_weights, _dkxp, _dvxp, _dqyp, stream);
        }
    }

    // Set TORCH_HARMONICS_RAGGED_BWD_GENERIC=1 to force the original kernel. Separate
    // from the forward's TORCH_HARMONICS_RAGGED_GENERIC on purpose: the two ports are
    // independent, so a wrong gradient can be attributed to one or the other at
    // runtime without a rebuild.
    static bool ragged_bwd_force_generic()
    {
        static const bool forced = []() {
            const char *env = std::getenv("TORCH_HARMONICS_RAGGED_BWD_GENERIC");
            return env != nullptr && env[0] == '1';
        }();
        return forced;
    }

    template <typename STORAGE_T>
    static void launch_gen_attn_bwd_ragged(int batch_size, int nheads, int nchans_in, int nchans_out,
                                          int64_t npoints_in, int64_t npoints_out, const STORAGE_T *_kxp,
                                          const STORAGE_T *_vxp, const STORAGE_T *_qyp, const STORAGE_T *_dyp,
                                          const int32_t *_seg, const int32_t *_seg_off, const int64_t *_ring_base,
                                          const int64_t *_ring_size, const float *_ring_weights,
                                          typename vec_traits<STORAGE_T>::compute_t *_dkxp,
                                          typename vec_traits<STORAGE_T>::compute_t *_dvxp,
                                          typename vec_traits<STORAGE_T>::compute_t *_dqyp, cudaStream_t stream)
    {
        // The register-blocked kernel needs NLOC == DIV_UP(nchan, WARP_SIZE) to hold for
        // both channel counts at once, which is why the equality is required rather
        // than taking the larger: NLOC also decides which registers go unguarded.
        const int nloc = DIV_UP(nchans_in, WARP_SIZE);
        if (!ragged_bwd_force_generic() && nchans_in == nchans_out && nloc <= MAX_LOCAL_ARR_LEN_RAGGED_BWD) {
            launch_spc_attn_bwd_ragged<WARP_SIZE, THREADS / WARP_SIZE, 1, MAX_LOCAL_ARR_LEN_RAGGED_BWD, STORAGE_T>(
                nloc, batch_size, nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _dyp, _seg,
                _seg_off, _ring_base, _ring_size, _ring_weights, _dkxp, _dvxp, _dqyp, stream);
            return;
        }

        dim3 block(WARP_SIZE, THREADS / WARP_SIZE);
        // one block row per (batch, head) pair
        dim3 grid(DIV_UP(npoints_out, block.y), batch_size * nheads);

        // shared memory holds compute-type (COMPUTE_T) data, not STORAGE_T. 4 arrays
        // per warp; the fifth was alpha_vw_, which is a scalar (see the kernel).
        size_t shsize = sizeof(typename vec_traits<STORAGE_T>::compute_t) * (nchans_in * 3 + nchans_out) * block.y;

        s2_attn_bwd_ragged_generic_vec_k<THREADS><<<grid, block, shsize, stream>>>(
            nheads, nchans_in, nchans_out, npoints_in, npoints_out, _kxp, _vxp, _qyp, _dyp, _seg, _seg_off, _ring_base,
            _ring_size, _ring_weights, _dkxp, _dvxp, _dqyp);
        CHECK_ERROR("s2_attn_bwd_ragged_generic_vec_k");

        return;
    }

    // NHWC ABI, flattened: see s2_attention_fwd_ragged_cuda. Argument order mirrors
    // `forward_ragged` with dy inserted after qy, matching how `backward` mirrors
    // `forward` on the product grids.
    std::tuple<at::Tensor, at::Tensor, at::Tensor>
    s2_attention_bwd_ragged_cuda(at::Tensor kx, at::Tensor vx, at::Tensor qy, at::Tensor dy, at::Tensor ring_weights,
                                 at::Tensor psi_seg, at::Tensor psi_seg_off, at::Tensor ring_base,
                                 at::Tensor ring_size, int64_t num_heads, int64_t npoints_out)
    {
        CHECK_CUDA_INPUT_TENSOR(kx);
        CHECK_CUDA_INPUT_TENSOR(vx);
        CHECK_CUDA_INPUT_TENSOR(qy);
        CHECK_CUDA_INPUT_TENSOR(dy);
        CHECK_CUDA_TENSOR(ring_weights);
        CHECK_CUDA_TENSOR(psi_seg);
        CHECK_CUDA_TENSOR(psi_seg_off);
        CHECK_CUDA_TENSOR(ring_base);
        CHECK_CUDA_TENSOR(ring_size);

        TORCH_CHECK(kx.dim() == 3, "kx must be (B, npoints_in, num_heads * C_k), got ", kx.dim(), " dims");
        TORCH_CHECK(vx.dim() == 3, "vx must be (B, npoints_in, num_heads * C_v), got ", vx.dim(), " dims");
        TORCH_CHECK(qy.dim() == 3, "qy must be (B, npoints_out, num_heads * C_k), got ", qy.dim(), " dims");
        TORCH_CHECK(dy.dim() == 3, "dy must be (B, npoints_out, num_heads * C_v), got ", dy.dim(), " dims");

        TORCH_CHECK(num_heads >= 1, "num_heads must be positive, got ", num_heads);
        TORCH_CHECK(qy.size(2) % num_heads == 0, "q/k channel count (", qy.size(2),
                    ") must be divisible by num_heads (", num_heads, ")");
        TORCH_CHECK(vx.size(2) % num_heads == 0, "v channel count (", vx.size(2), ") must be divisible by num_heads (",
                    num_heads, ")");

        TORCH_CHECK(qy.size(1) == npoints_out, "qy has ", qy.size(1), " points but npoints_out is ", npoints_out);
        TORCH_CHECK(dy.size(1) == npoints_out, "dy has ", dy.size(1), " points but npoints_out is ", npoints_out);
        TORCH_CHECK(dy.size(2) == vx.size(2), "dy has ", dy.size(2), " channels but vx has ", vx.size(2));
        TORCH_CHECK(kx.size(1) == vx.size(1), "kx has ", kx.size(1), " points but vx has ", vx.size(1));
        TORCH_CHECK(psi_seg_off.size(0) == npoints_out + 1, "seg_off must have npoints_out + 1 = ", npoints_out + 1,
                    " entries, got ", psi_seg_off.size(0));
        TORCH_CHECK(psi_seg.dim() == 2 && psi_seg.size(1) == 3, "seg must be (nsegs, 3)");
        TORCH_CHECK(ring_base.size(0) == ring_size.size(0), "ring_base and ring_size must agree in length, got ",
                    ring_base.size(0), " and ", ring_size.size(0));
        TORCH_CHECK(ring_weights.size(0) == ring_base.size(0), "ring_weights must have one entry per input ring, got ",
                    ring_weights.size(0), " for ", ring_base.size(0), " rings");

        // Every activation must share one dtype: the dispatch below selects a single
        // scalar_t from qy and the launcher reinterpret_casts k/v/q/dy to it, so a
        // mismatched input would be reinterpreted rather than converted.
        TORCH_CHECK(kx.scalar_type() == qy.scalar_type(), "k dtype (", kx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");
        TORCH_CHECK(vx.scalar_type() == qy.scalar_type(), "v dtype (", vx.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");
        TORCH_CHECK(dy.scalar_type() == qy.scalar_type(), "dy dtype (", dy.scalar_type(), ") must match q dtype (",
                    qy.scalar_type(), ")");

        // ring_weights is read as float32 whatever scalar_t the dispatch picks, so it
        // is the one tensor that must not follow the activations. Casting a whole
        // module to bf16 would sweep it along, and reinterpreting those bytes as float
        // computes silent garbage instead of failing.
        TORCH_CHECK(ring_weights.scalar_type() == at::kFloat, "ring_weights must be float32, got ",
                    ring_weights.scalar_type());

        // per-head channel counts; the packed extent is num_heads times these
        const int nchans_in = qy.size(2) / num_heads; // or kx.size(2) / num_heads
        const int nchans_out = vx.size(2) / num_heads;

        const int batch_size = kx.size(0);
        const int64_t npoints_in = kx.size(1);

        auto kx_type = kx.dtype();
        auto vx_type = vx.dtype();
        auto qy_type = qy.dtype();

        torch::Tensor dkx, dvx, dqy;

        // Activations stay in their native dtype and are widened to fp32 at load.
        // Gradient buffers are allocated fp32 because dkx/dvx are atomically
        // scatter-accumulated; they are narrowed back to the input dtype at the end.
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qy.scalar_type(), "s2_attention_bwd_ragged_cuda", [&] {
            using storage_t = scalar_t;

            auto stream = at::cuda::getCurrentCUDAStream().stream();

            // zeros, not empty: dkx/dvx are accumulated into with atomicAdd, so the
            // buffers must start at zero. dqy is written outright, but is allocated
            // the same way to keep the three identical.
            const auto f32_like
                = [](const torch::Tensor &t) { return torch::zeros_like(t, t.options().dtype(torch::kFloat32)); };

            torch::Tensor dkxP = f32_like(kx);
            torch::Tensor dvxP = f32_like(vx);
            torch::Tensor dqyP = f32_like(qy);

            using compute_t = typename vec_traits<storage_t>::compute_t;

            launch_gen_attn_bwd_ragged<storage_t>(
                batch_size, num_heads, nchans_in, nchans_out, npoints_in, npoints_out,
                reinterpret_cast<const storage_t *>(kx.data_ptr()), reinterpret_cast<const storage_t *>(vx.data_ptr()),
                reinterpret_cast<const storage_t *>(qy.data_ptr()), reinterpret_cast<const storage_t *>(dy.data_ptr()),
                reinterpret_cast<const int32_t *>(psi_seg.data_ptr()),
                reinterpret_cast<const int32_t *>(psi_seg_off.data_ptr()),
                reinterpret_cast<const int64_t *>(ring_base.data_ptr()),
                reinterpret_cast<const int64_t *>(ring_size.data_ptr()),
                reinterpret_cast<const float *>(ring_weights.data_ptr()),
                reinterpret_cast<compute_t *>(dkxP.data_ptr()), reinterpret_cast<compute_t *>(dvxP.data_ptr()),
                reinterpret_cast<compute_t *>(dqyP.data_ptr()), stream);

            dkx = dkxP;
            dvx = dvxP;
            dqy = dqyP;
        });

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // convert precision back to starting dtype (no-op for fp32; narrows for fp16/bf16)
        dkx = dkx.to(kx_type);
        dvx = dvx.to(vx_type);
        dqy = dqy.to(qy_type);

        return std::make_tuple(dkx, dvx, dqy);
    }

    TORCH_LIBRARY_IMPL(attention_kernels, CUDA, m) { m.impl("backward_ragged", &s2_attention_bwd_ragged_cuda); }

} // namespace attention_kernels
