# coding=utf-8

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
#

from typing import List, Tuple, Union, Optional
from warnings import warn

import math

import torch
import torch.nn as nn
import numpy as np

from torch_harmonics.quadrature import _precompute_latitudes
from torch_harmonics.convolution import _precompute_convolution_tensor_s2
from torch_harmonics._neighborhood_attention import _neighborhood_attention_s2_torch, _neighborhood_attention_s2_cuda
from torch_harmonics.filter_basis import get_filter_basis

# import custom C++/CUDA extensions
try:
    import attention_cuda_extension

    _cuda_extension_available = True
except ImportError as err:
    attention_cuda_extension = None
    _cuda_extension_available = False

class AttentionS2(nn.Module):
    """
    (Global) attention on the 2-sphere.
    Parameters
    -----------
    in_channels: int
        number of channels of the input signal (corresponds to embed_dim in MHA in PyTorch)
    num_heads: int
        number of attention heads
    in_shape: tuple
        shape of the input grid
    out_shape: tuple
        shape of the output grid
    grid_in: str, optional
        input grid type, "equiangular" by default
    grid_out: str, optional
        output grid type, "equiangular" by default
    bias: bool, optional
        if specified, adds bias to input / output projection layers
    k_channels: int
        number of dimensions for interior inner product in the attention matrix (corresponds to kdim in MHA in PyTorch)
    out_channels: int, optional
        number of dimensions for interior inner product in the attention matrix (corresponds to vdim in MHA in PyTorch)
    """

    def __init__(
            self,
            in_channels: int,
            num_heads: int,
            in_shape: Tuple[int],
            out_shape: Tuple[int],
            grid_in: Optional[str] = "equiangular",
            grid_out: Optional[str] = "equiangular",
            scale: Optional[Union[torch.Tensor, float]] = None,
            bias: Optional[bool] = True,
            k_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            drop_rate: Optional[float]=0.0,
    ):
        super().__init__()

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.k_channels = in_channels if k_channels is None else k_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.drop_rate = drop_rate
        self.scale = scale

        # integration weights
        _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in)
        quad_weights = 2.0 * torch.pi * wgl.to(dtype=torch.float32) / self.nlon_in
        # we need to tile and flatten them accordingly
        quad_weights = torch.tile(quad_weights.reshape(-1, 1), (1, self.nlon_in)).flatten()

        # compute log because they are applied as an addition prior to the softmax ('attn_mask'), which includes an exponential.
        # see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        # for info on how 'attn_mask' is applied to the attention weights
        log_quad_weights = torch.log(quad_weights).reshape(1,1,-1)
        self.register_buffer("log_quad_weights", log_quad_weights, persistent=False)

        # learnable parameters
        # TODO: double-check that this gives us the correct initialization magnitudes
        # the standard MHA uses xavier uniform, NATTEN uses kaiming. Let's use that for now
        if self.k_channels % self.num_heads != 0:
            raise ValueError(f"Please make sure that number of heads {self.num_heads} divides k_channels {self.k_channels} evenly.")
        if self.out_channels % self.num_heads != 0:
            raise ValueError(f"Please make sure that number of heads {self.num_heads} divides out_channels {self.out_channels} evenly.")
        scale_qkv = math.sqrt(3.0 / self.in_channels)
        self.q_weights = nn.Parameter(scale_qkv * (2 * torch.rand(self.k_channels, self.in_channels, 1, 1) - 1))
        self.k_weights = nn.Parameter(scale_qkv * (2 * torch.rand(self.k_channels, self.in_channels, 1, 1) - 1))
        self.v_weights = nn.Parameter(scale_qkv * (2 * torch.rand(self.out_channels, self.in_channels, 1, 1) - 1))
        scale_proj = math.sqrt(3.0 / self.out_channels)
        self.proj_weights = nn.Parameter(scale_proj * (2 * torch.rand(self.out_channels, self.out_channels, 1, 1) - 1))

        if bias:
            self.q_bias = nn.Parameter(torch.zeros(self.k_channels))
            self.k_bias = nn.Parameter(torch.zeros(self.k_channels))
            self.v_bias = nn.Parameter(torch.zeros(self.out_channels))
            self.proj_bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
            self.proj_bias = None


    def extra_repr(self):
        r"""
            Pretty print module
         """
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_channels={self.in_channels}, out_channels={self.out_channels}, k_channels={self.k_channels}"

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None) -> torch.Tensor:

        # self attention simplification
        if key is None:
            key = query

        if value is None:
            value = query

        # change this later to allow arbitrary number of batch dims
        assert (query.dim() == key.dim()) and (key.dim() == value.dim()) and (value.dim() == 4)

        # perform MLP
        query = nn.functional.conv2d(query, self.q_weights, bias=self.q_bias)
        key = nn.functional.conv2d(key, self.k_weights, bias=self.k_bias)
        value = nn.functional.conv2d(value, self.v_weights, bias=self.v_bias)

        # reshape
        B, _, H, W = query.shape
        query = query.reshape(B, self.num_heads, -1, H, W)
        B, _, H, W = key.shape
        key = key.reshape(B, self.num_heads, -1, H, W)
        B, _, H, W = value.shape
        value = value.reshape(B, self.num_heads, -1, H, W)

        # reshape to the right dimensions
        B, _, C, H, W = query.shape
        query = query.permute(0,1,3,4,2).reshape(B, self.num_heads, H*W, C)
        B, _, C, H, W = key.shape
        key = key.permute(0,1,3,4,2).reshape(B, self.num_heads, H*W, C)
        B, _, C, H, W = value.shape
        value = value.permute(0,1,3,4,2).reshape(B, self.num_heads, H*W, C)

        # multiply the query, key and value tensors
        out = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=self.log_quad_weights, dropout_p=self.drop_rate, scale=self.scale)

        # reshape
        B, _, _, C = out.shape
        # (B, heads, H*W, C)
        out = out.permute(0,1,3,2)
        # (B, heads, C, H*W)
        out = out.reshape(B, self.num_heads*C, self.nlat_out, self.nlon_out)
        # (B, heads*C, H, W)
        out = nn.functional.conv2d(out, self.proj_weights, bias=self.proj_bias)

        return out

def compute_lon_padding_columns(
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    nlon: int,
    radius_rad: float
) -> int:
    """
    Compute how many longitude columns to add on each side of an equiangular grid slice
    so that the grid point at the middle longitude and the latitude closest to a pole
    has a complete spherical cap neighborhood of angular radius `radius_rad`.

    The domain is a “slice” of the sphere in colatitude-longitude coordinates:
      • Longitude runs from lon_min to lon_max (in radians), with 0 <= lon_min < lon_max <= 2pi.
      • Colatitude theta runs from lat_min to lat_max (in radians), with 0 <= lat_min < lat_max <= pi.
        (theta = 0 at North Pole, theta = pi/2 at Equator, theta = pi at South Pole.)

    Parameters
    ----------
    lon_range : tuple[float, float]
        (lon_min, lon_max) in radians, specifying the slice's longitude bounds.
        Assumes 0 <= lon_min < lon_max <= 2pi and no wrap-around across 0/2pi.
    lat_range : tuple[float, float]
        (lat_min, lat_max) in radians, specifying the slice's colatitude bounds.
        Assumes 0 <= lat_min < lat_max <= pi.  If lat_min = 0 or lat_max = pi, a pole is included.
    nlon : int
        Number of equally spaced longitude grid points between lon_min and lon_max.
    nlat : int
        Number of equally spaced colatitude grid points between lat_min and lat_max.
    radius_rad : float
        Desired neighborhood angular radius (in radians).

    Returns
    -------
    pad_cols_per_side : int
        The number of extra longitude-columns to add on each side so that the point
        at “middle longitude” and the boundary latitude nearest a pole can still
        have a full spherical-cap of radius 'radius_rad'.  Guaranteed to lie in [0, nlon//2].

    Method
    ------
    1.  Let current_width = lon_max - lon_min.  Each longitude column spans
          deltalambda_grid = current_width / (nlon - 1).
    2.  Compute distances to the poles at the two colatitude boundaries:
          d_north = lat_min,   # distance to North Pole (theta = 0)
          d_south = pi - lat_max  # distance to South Pole (theta = pi)
        Let d_min = min(d_north, d_south).  If radius_rad >= d_min, a cap at that boundary
        already reaches a pole, so the required longitude-width is 2pi.
    3.  Otherwise, set phi_edge = lat_min if d_north <= d_south else lat_max.  That phi_edge
        is the boundary colatitude closest to a pole.
    4.  Solve for deltalambda_max at phi_edge from the spherical law of cosines (same latitude):
          cos(radius_rad) = cos**2(phi_edge) + sin**2(phi_edge) * cos(deltalambda_max),
        so
          deltalambda_max = arccos( (cos(radius_rad) - cos**2phi_edge)) / sin**2(phi_edge) ).
        The total required width at that latitude is 2·Δλ_max.
    5.  Let required_width = max(2pi, 2deltalambda_max).  Compute extension_width = max(0, required_width - current_width).
    6.  The naive number of extra columns per side = ceil((extension_width/2) / Δλ_grid).
    7.  As a secondary condition, also require at least
          ceil(radius_rad * (lon_max - lon_min) / nlon)
        columns per side (ensures a minimum based on the cap's angular radius).
    8.  Take the maximum of steps 6 and 7, then cap at nlon//2:
          pad = min( nlon//2, max( ceil((extension_width/2)/deltalambda_grid),
                                     ceil(radius_rad * (lon_max - lon_min)/nlon ) ) ).

    This guarantees 0 <= pad_cols_per_side <= nlon//2.
    """

    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range

    # Step 1: current total width and grid spacing in longitude
    current_width = lon_max - lon_min
    delta_lambda_grid = current_width / (nlon - 1)

    # Step 2: distance from each boundary to nearest pole
    distance_north = lat_min            # colatitude lat_min is distance to North Pole
    distance_south = math.pi - lat_max  # colatitude lat_max is distance to South Pole
    d_min = min(distance_north, distance_south)

    # Step 3: if radius >= d_min, cap hits a pole => required_width = 2pi
    if radius_rad >= d_min:
        required_width = 2 * math.pi
    else:
        # Step 4: pick the boundary colatitude nearest a pole
        phi_edge = lat_min if (distance_north <= distance_south) else lat_max

        # use spherical law of cosines at constant colatitude to find deltalambda_max
        cos_r = math.cos(radius_rad)
        cos_phi = math.cos(phi_edge)
        sin_phi = math.sin(phi_edge)

        # Compute the argument for arccos, clamped to [-1, +1]
        numerator = cos_r - (cos_phi * cos_phi)
        denominator = sin_phi * sin_phi
        arg = numerator / denominator
        arg = max(-1.0, min(1.0, arg))

        delta_lambda_max = math.acos(arg)
        required_width = 2 * delta_lambda_max

    # Step 5: how much wider than current_width?
    extension_width = max(0.0, required_width - current_width)

    # Step 6: naive padding = ceil((extension_width/2) / deltalambda_grid)
    naive_pad = math.ceil((extension_width / 2) / delta_lambda_grid)

    # Step 7: secondary minimum padding = ceil(radius_rad * (lon_max − lon_min) / nlon)
    #         (This term enforces a baseline based on the cap’s angular radius.)
    baseline_pad = math.ceil(radius_rad * nlon / current_width)

    # Step 8: final pad = min(nlon//2, max(naive_pad, baseline_pad))
    pad_cols_per_side = min(nlon // 2, max(naive_pad, baseline_pad))

    return pad_cols_per_side

class NeighborhoodAttentionS2(nn.Module):
    """
    Neighborhood attention on the 2-sphere.

    Parameters
    -----------
    in_channels: int
        number of channels of the input signal (corresponds to embed_dim in MHA in PyTorch)
    in_shape: tuple
        shape of the input grid
    out_shape: tuple
        shape of the output grid
    grid_in: str, optional
        input grid type, "equiangular" by default
    grid_out: str, optional
        output grid type, "equiangular" by default
    bias: bool, optional
        if specified, adds bias to input / output projection layers
    theta_cutoff: float, optional
        neighborhood size
    k_channels: int
        number of dimensions for interior inner product in the attention matrix (corresponds to kdim in MHA in PyTorch)
    out_channels: int, optional
        number of dimensions for interior inner product in the attention matrix (corresponds to vdim in MHA in PyTorch)
    """
    def __init__(
        self,
        in_channels: int,
        in_shape: Tuple[int,int],
        out_shape: Tuple[int,int],
        grid_in: str = "cosine",
        grid_out: str = "cosine",
        lon_range: Tuple[float,float] = (0, 2*math.pi),
        lat_range: Tuple[float,float] = (0, math.pi),
        num_heads: int = 1,
        scale=None,
        bias: bool = True,
        theta_cutoff: Optional[float] = None,
        k_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
    ):
        
        super().__init__()
        
        # validate shapes/ranges omitted for brevity…
        self.k_channels = in_channels if k_channels is None else k_channels
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.out_channels = in_channels if out_channels is None else out_channels
        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.grid_in = grid_in
        self.grid_out = grid_out

        if theta_cutoff is None:
            theta_cutoff = math.pi / float(self.nlat_out - 1)

        # need ghost grid if we do NOT cover full 0→2π
        self.use_padding = not (abs(lon_range[0]) < 1e-6 and abs(lon_range[1] - 2*math.pi) < 1e-6)

        if self.use_padding:
            padding_points = compute_lon_padding_columns(
                lon_range=lon_range,
                lat_range=lat_range,
                nlon=self.nlon_in,
                nlat=self.nlat_in,
                radius_rad=theta_cutoff
            )
            domain_size = lon_range[1] - lon_range[0]
            cell_width = domain_size / self.nlon_in
            padded_lon_min = max(0, lon_range[0] - padding_points * cell_width)
            padded_lon_max = lon_range[1] + padding_points * cell_width
            
            # Recompute padded_nlon_in
            self.padded_nlon_in = int(round((padded_lon_max - padded_lon_min) / cell_width))
            self.in_lon_range = (padded_lon_min, padded_lon_max)
            self.start_idx = int(round((lon_range[0] - padded_lon_min) / cell_width))
            self.end_idx   = self.start_idx + self.nlon_in
            self.orig_nlon_in = self.nlon_in
            self.nlon_in      = self.padded_nlon_in
        else:
            self.in_lon_range = lon_range
            self.orig_nlon_in = self.nlon_in
            self.padded_nlon_in = self.orig_nlon_in
            self.start_idx = 0
            self.end_idx = self.nlon_in

        # integration weights (on the padded grid if ghost)
        _, wgl = _precompute_latitudes(self.nlat_in, grid=grid_in,
                                       a=math.cos(lat_range[1]), b=math.cos(lat_range[0]))
        quad_weights = 2.0 * math.pi * wgl.to(torch.float32) / self.nlon_in
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        # dummy basis for neighborhood shape only
        fb = get_filter_basis(kernel_shape=1, basis_type="zernike")

        # **use padded size in precompute**
        idx, _ = _precompute_convolution_tensor_s2(
            (self.nlat_in, self.padded_nlon_in),
            out_shape,
            fb,
            grid_in=grid_in,
            grid_out=grid_out,
            in_lat_range=lat_range,
            out_lat_range=lat_range,
            in_lon_range=self.in_lon_range,
            out_lon_range=self.lon_range,
            theta_cutoff=theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode="none",
            merge_quadrature=True,
        )

        # this is kept for legacy resons in case we want to resuse sorting of these entries
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()

        # compute row offsets for more structured traversal.
        # only works if rows are sorted but they are by construction
        row_offset = np.empty(self.nlat_out + 1, dtype=np.int64)
        row_offset[0] = 0
        row = row_idx[0]
        for idz, z in enumerate(range(col_idx.shape[0])):
            if row_idx[z] != row:
                row_offset[row + 1] = idz
                row = row_idx[z]

        # set the last value
        row_offset[row + 1] = idz + 1
        row_offset = torch.from_numpy(row_offset)
        self.max_psi_nnz = col_idx.max().item() + 1

        self.register_buffer("psi_row_idx", row_idx, persistent=False)
        self.register_buffer("psi_col_idx", col_idx, persistent=False)
        self.register_buffer("psi_roff_idx", row_offset, persistent=False)

        # learnable parameters
        if self.k_channels % self.num_heads != 0:
            raise ValueError(f"Please make sure that number of heads {self.num_heads} divides k_channels {self.k_channels} evenly.")
        if self.out_channels % self.num_heads != 0:
            raise ValueError(f"Please make sure that number of heads {self.num_heads} divides out_channels {self.out_channels} evenly.")
        scale_qkv = math.sqrt(3.0 / self.in_channels)
        self.q_weights = nn.Parameter(scale_qkv * (2 * torch.rand(self.k_channels, self.in_channels, 1, 1) - 1))
        self.k_weights = nn.Parameter(scale_qkv * (2 * torch.rand(self.k_channels, self.in_channels, 1, 1) - 1))
        self.v_weights = nn.Parameter(scale_qkv * (2 * torch.rand(self.out_channels, self.in_channels, 1, 1) - 1))
        scale_proj = math.sqrt(3.0 / self.out_channels)
        self.proj_weights = nn.Parameter(scale_proj * (2 * torch.rand(self.out_channels, self.out_channels, 1, 1) - 1))

        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1 / math.sqrt(self.k_channels)

        if bias:
            self.q_bias = nn.Parameter(torch.zeros(self.k_channels))
            self.k_bias = nn.Parameter(torch.zeros(self.k_channels))
            self.v_bias = nn.Parameter(torch.zeros(self.out_channels))
            self.proj_bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
            self.proj_bias = None

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"in_shape={(self.nlat_in, self.nlon_in)}, out_shape={(self.nlat_out, self.nlon_out)}, in_channels={self.in_channels}, out_channels={self.out_channels}, k_channels={self.k_channels}, wrap={self.wrap}, use_ghost_grid={self.use_ghost_grid}"

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input validation
        if query.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {query.dim()}D")
        
        # self attention simplification
        if key is None:
            key = query

        if value is None:
            value = query

        # Validate input shapes
        if key.shape != value.shape:
            raise ValueError(f"Key and value shapes must match, got {key.shape} and {value.shape}")
        if key.shape[2:] != (self.nlat_in, self.nlon_in):
            raise ValueError(f"Expected input shape {(self.nlat_in, self.nlon_in)}, got {key.shape[2:]}")

        # do the scaling
        query_scaled = query * self.scale

        # TODO: insert dimension checks for input
        if query.is_cuda and _cuda_extension_available:
            out = _neighborhood_attention_s2_cuda(
                key,
                value,
                query_scaled,
                self.k_weights,
                self.v_weights,
                self.q_weights,
                self.k_bias,
                self.v_bias,
                self.q_bias,
                self.quad_weights,
                self.psi_col_idx,
                self.psi_roff_idx,
                self.max_psi_nnz,
                self.num_heads,
                self.orig_nlon_in,  # Use original size for input
                self.nlat_out,
                self.nlon_out,
            )
        else:
            if query.is_cuda:
                warn("couldn't find CUDA extension, falling back to slow PyTorch implementation")

            # call attention
            out = _neighborhood_attention_s2_torch(
                key,
                value,
                query_scaled,
                self.k_weights,
                self.v_weights,
                self.q_weights,
                self.k_bias,
                self.v_bias,
                self.q_bias,
                self.quad_weights,
                self.psi_col_idx,
                self.psi_roff_idx,
                self.num_heads,
                self.orig_nlon_in,  # Use original size for input
                self.nlat_out,
                self.nlon_out,
                self.start_idx,
                self.end_idx
            )

        out = nn.functional.conv2d(out, self.proj_weights, bias=self.proj_bias)

        return out