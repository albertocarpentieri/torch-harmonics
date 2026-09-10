# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
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

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, Optional, Set, Tuple, Type, Union

import torch

from torch_harmonics.partition import compute_split_shapes
from torch_harmonics.quadrature import precompute_latitudes, precompute_longitudes

__all__ = [
    "GridS2",
    "GridShardS2",
    "RegularGridS2",
    "EquiangularGrid",
    "LegendreGaussGrid",
    "LobattoGrid",
    "EquiangularTrapezoidalGrid",
    "as_grid",
    "grid_types",
    "require_grid",
]

# populated by __init_subclass__; maps the historical grid string to its class
_GRID_REGISTRY: Dict[str, Type["GridS2"]] = {}

# intermediate bases declared with `abstract=True`; they carry shared behaviour but
# describe no grid, so they neither register a grid_type nor allow instantiation
_ABSTRACT_GRIDS: Set[Type["GridS2"]] = set()


@dataclass(frozen=True, eq=False)
class GridS2:
    r"""
    Descriptor for an isolatitude grid on :math:`S^2`.

    This is the abstract base; instantiate one of the concrete subclasses, or use
    :func:`as_grid` to coerce a legacy ``(grid_string, shape)`` pair.

    Each concrete subclass carries a class-level ``grid_type`` holding the
    historical grid string it corresponds to, e.g. ``"equiangular"``. That string
    is used for serialization and for the registry behind :func:`as_grid`.

    The base deliberately declares **no fields**. A grid's parameters are whatever
    identifies it: a latitude/longitude product grid is fixed by ``(nlat, nlon)``,
    which is what :class:`RegularGridS2` adds, whereas HEALPix is fixed by a single
    ``nside`` and has ``npoints != nlat * nlon``. Pushing the resolution down to the
    subclass is what lets a ragged grid be an additive change rather than a second
    API break.

    Concrete subclasses must provide :attr:`params`, :attr:`nlat`, :attr:`nlon`,
    :attr:`lats`, :attr:`quad_weights` and :meth:`lons`; everything else here is
    derived from those.

    Notes
    -----
    Three properties of this type are load-bearing rather than incidental.

    Identity is a canonical tuple of scalars, and :attr:`key` backs both hashing
    and equality. It is derived from :attr:`params` rather than restated, so a
    parameter cannot be added to a grid and forgotten in its key -- a descriptor
    whose key omitted a distinguishing parameter would silently serve one grid's
    cached psi to another. Node and weight tensors are deliberately not fields: a
    descriptor carrying tensors would fall back to identity hashing and silently
    defeat every cache keyed on the grid.

    :attr:`nlon_per_lat` and :attr:`lon_offsets` exist on the regular grids too,
    where they are trivial. Consumers that cannot handle a ragged grid should assert
    :attr:`is_regular` rather than assume a uniform ``nlon`` stride, and should read
    :attr:`spatial_shape` rather than :attr:`shape`, which is a product grid's
    notion and raises on a ragged one.

    Descriptors stop at the Python layer: compiled kernels keep taking plain ints
    and index tensors, and modules unpack the descriptor before calling into them.
    """

    #: historical grid string; set by each concrete subclass
    grid_type: ClassVar[str]

    def __init_subclass__(cls, abstract: Optional[bool] = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if abstract:
            _ABSTRACT_GRIDS.add(cls)
            return
        grid_type = getattr(cls, "grid_type", None)
        if grid_type is None:
            raise TypeError(f"{cls.__name__} must define a 'grid_type' class attribute")
        if grid_type in _GRID_REGISTRY:
            raise ValueError(f"grid_type '{grid_type}' is already registered to {_GRID_REGISTRY[grid_type].__name__}")
        _GRID_REGISTRY[grid_type] = cls

    def __post_init__(self):
        if type(self) is GridS2 or type(self) in _ABSTRACT_GRIDS:
            raise TypeError(f"{type(self).__name__} is abstract; instantiate a concrete grid or use as_grid()")
        self._validate()

    def _validate(self):
        """Check this grid's own parameters. Called from ``__post_init__``."""
        raise NotImplementedError(f"{type(self).__name__} does not define _validate")

    # -- identity ------------------------------------------------------------

    @property
    def params(self) -> Dict[str, Any]:
        """
        The constructor arguments that identify this grid, as plain scalars.

        Single source of truth for :attr:`key`, :meth:`to_dict` and ``__repr__``,
        so those three cannot drift apart as grids are added.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define params")

    @property
    def key(self) -> Tuple[Any, ...]:
        """
        Canonical, hashable identity of this grid.

        Contains only scalars. Everything that distinguishes two grids must appear
        here, and nothing that does not; this tuple backs both ``__hash__`` and
        ``__eq__``, and therefore every cache keyed on a descriptor. Built from
        :attr:`params` so that it stays exhaustive by construction.
        """
        params = self.params
        return (self.grid_type,) + tuple(params[name] for name in sorted(params))

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridS2):
            return NotImplemented
        return self.key == other.key

    def __repr__(self) -> str:
        args = ", ".join(f"{name}={value}" for name, value in self.params.items())
        return f"{type(self).__name__}({args})"

    # -- resolution ----------------------------------------------------------
    #
    # nlat and nlon are deliberately *not* declared here, as properties or otherwise,
    # even though everything below reads them and every concrete grid has them.
    #
    # A subclass supplies them either as dataclass fields (RegularGridS2, where the
    # resolution *is* the parameterization) or as derived properties (HealpixGrid,
    # where both follow from nside). Those two are mutually exclusive with a stub on
    # the base: a property is a data descriptor, so it wins over the instance
    # dictionary, and the frozen dataclass __init__ -- which assigns through
    # object.__setattr__, and that honours data descriptors -- would fail with
    # "property 'nlat' has no setter". Worse, @dataclass reads class attributes to
    # find field defaults, so the stub would also silently become the default value
    # of the field that shadows it.
    #
    # nlat is the number of latitude rings. nlon is the number of longitudes on the
    # *widest* ring: on a regular grid that is every ring, on a ragged one it is the
    # bound a dense azimuthal representation must accommodate, and never a stride --
    # see nlon_per_lat for the per-ring counts.

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Spatial shape ``(nlat, nlon)`` of a field sampled on this grid.

        Only meaningful on a regular grid, and raises otherwise: a ragged grid
        stores its points flat, and returning ``(nlat, nlon)`` for it would name a
        rectangle that is larger than the grid. Ragged-aware consumers read
        :attr:`spatial_shape`.
        """
        raise TypeError(
            f"{type(self).__name__} is ragged, so it has no (nlat, nlon) shape: it carries "
            f"{self.npoints} points flat, not {self.nlat} * {self.nlon}. Use spatial_shape for the "
            "storage shape, nlon_per_lat and lon_offsets for the ring structure, and check is_regular "
            "before assuming a uniform nlon stride."
        )

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        """
        Shape of the spatial dimensions of a field on this grid.

        ``(nlat, nlon)`` on a regular grid and ``(npoints,)`` on a ragged one, so a
        layer can allocate and validate without knowing which it has.
        """
        return (self.nlat, self.nlon) if self.is_regular else (self.npoints,)

    # -- geometry ------------------------------------------------------------

    @property
    def lats(self) -> torch.Tensor:
        r"""
        Colatitudes :math:`\theta_k \in [0, \pi]`, ascending (north pole first), shape ``(nlat,)``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define lats")

    @property
    def quad_weights(self) -> torch.Tensor:
        r"""
        Latitudinal quadrature weights, shape ``(nlat,)``, paired with :attr:`lats`.

        Formulated in the :math:`\cos\theta` domain, so they already absorb the
        :math:`\sin\theta` Jacobian and sum to 2. The weight of a single point on
        ring ``k`` is ``2 * pi * quad_weights[k] / nlon_per_lat[k]``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define quad_weights")

    def lons(self, ilat: Optional[int] = None) -> torch.Tensor:
        r"""
        Longitudes :math:`\lambda_j \in [0, 2\pi)` of a latitude ring.

        Parameters
        ----------
        ilat : int, optional
            Index of the latitude ring. Ignored on regular grids, where every ring
            carries the same longitudes; required on ragged ones, where both the
            count and the offset of the ring's longitudes depend on it.

        Returns
        -------
        torch.Tensor
            Longitudes in radians, shape ``(nlon_per_lat[ilat],)``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define lons")

    # -- raggedness ----------------------------------------------------------

    @property
    def is_regular(self) -> bool:
        """
        Whether every latitude ring carries the same number of longitudes.

        ``True`` for the latitude/longitude product grids, ``False`` for HEALPix.
        Consumers backed by compiled kernels that index with a uniform ``nlon``
        stride should assert this.
        """
        return True

    @property
    def nlon_per_lat(self) -> torch.Tensor:
        """Number of longitudes on each latitude ring, shape ``(nlat,)``."""
        return torch.full((self.nlat,), self.nlon, dtype=torch.int64)

    @property
    def lon_offsets(self) -> torch.Tensor:
        """
        Exclusive prefix sum of :attr:`nlon_per_lat`, shape ``(nlat + 1,)``.

        A point ``(ilat, ilon)`` sits at flat index ``lon_offsets[ilat] + ilon``.
        On a regular grid this is just ``ilat * nlon``, but writing the flattening
        this way keeps consumers valid on ragged grids.
        """
        return torch.arange(self.nlat + 1, dtype=torch.int64) * self.nlon

    @property
    def npoints(self) -> int:
        """Total number of grid points."""
        return int(self.lon_offsets[-1].item())

    # -- derived quantities --------------------------------------------------

    @property
    def latitude_spacing(self) -> torch.Tensor:
        r"""Gaps :math:`\theta_{k+1} - \theta_k` between adjacent latitudes, shape ``(nlat - 1,)``."""
        lats = self.lats
        return lats[1:] - lats[:-1]

    @property
    def max_latitude_spacing(self) -> float:
        r"""
        Largest gap between adjacent latitudes, :math:`\max_k (\theta_{k+1} - \theta_k)`.

        This is the grid's own notion of "one latitudinal grid spacing". Only
        :class:`EquiangularGrid` is uniform in :math:`\theta`, where it reduces to
        :math:`\pi / (N_\theta - 1)`.

        Taken from the node distribution rather than from a per-grid formula, which
        is what :func:`torch_harmonics.quadrature.compute_latitude_spacing` does for
        the quadrature grids; the two agree by construction.
        """
        return self.latitude_spacing.max().item()

    @property
    def max_longitude_spacing(self) -> float:
        r"""
        Largest great-circle gap between neighbouring points *within* a latitude ring,
        :math:`\max_k \frac{2\pi}{N_{\lambda,k}} \sin\theta_k`.

        The azimuthal counterpart of :attr:`max_latitude_spacing`. On a product grid
        with :math:`N_\lambda \approx 2 N_\theta` the two are within a few percent of
        each other, which is why the localized operators have been able to describe
        their support with the latitudinal one alone. A grid can be strongly
        anisotropic though -- on HEALPix this is about 1.8x the latitudinal spacing --
        and there the distinction decides whether an output point's stencil reaches
        its neighbours at all.
        """
        return ((2.0 * torch.pi / self.nlon_per_lat.to(torch.float64)) * torch.sin(self.lats)).max().item()

    @property
    def is_uniform_in_theta(self) -> bool:
        r"""Whether the latitude nodes are equispaced in :math:`\theta`."""
        return False

    # -- spectral bounds -----------------------------------------------------
    #
    # These are facts about what the grid can represent, not decisions about
    # what an SHT should keep. The policy -- applying user overrides, enforcing
    # triangular truncation, warning about changed defaults -- lives in
    # :mod:`torch_harmonics.truncation`, so these properties stay silent.

    @property
    def max_exact_degree(self) -> int:
        r"""
        Highest spherical harmonic degree the quadrature rule integrates exactly.

        Non-inclusive, i.e. degrees :math:`0 \le l < l_{\max}`. Determined by the
        exactness of the latitudinal rule, so each grid type answers differently.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define max_exact_degree")

    @property
    def is_spectrally_accurate(self) -> bool:
        r"""
        Whether the latitudinal rule converges spectrally.

        An SHT relies on the associated Legendre polynomials being *discretely*
        orthogonal under the grid's quadrature. Interpolatory rules -- Gauss--Legendre,
        Gauss--Lobatto, Clenshaw--Curtis -- integrate the required polynomial degrees
        exactly, so orthogonality holds to machine precision. A rule that converges
        only algebraically does not, and refining the grid buys back accuracy far more
        slowly than raising the truncation loses it.
        """
        return True

    @property
    def max_azimuthal_order(self) -> int:
        r"""
        Nyquist limit of the longitudinal sampling, :math:`\lfloor N_\lambda / 2 \rfloor + 1`.

        Non-inclusive. On a ragged grid each latitude ring has its own limit; this
        returns the bound for the widest ring, which is the one a dense spectral
        representation has to accommodate.
        """
        return self.nlon // 2 + 1

    def theta_cutoff(self, scale: Optional[float] = 1.0) -> float:
        r"""
        Angular support radius of one latitudinal grid spacing.

        A restatement of :attr:`max_latitude_spacing` in the units localized
        operators ask for, and like it a fact about the node distribution: it
        neither applies a user override nor warns that the default moved. That
        policy lives in :func:`torch_harmonics.truncate_support`,
        which is what the layers call.

        Parameters
        ----------
        scale : float, optional
            Multiplier on the grid spacing, by default 1.0.

        Returns
        -------
        float
            Cutoff angle in radians.
        """
        return scale * self.max_latitude_spacing

    # -- decomposition -------------------------------------------------------

    def shard(self, polar: Optional[Tuple[int, int]] = (0, 1), azimuth: Optional[Tuple[int, int]] = (0, 1)) -> "GridShardS2":
        """
        Return the piece of this grid held by one rank of a 2D decomposition.

        Deliberately takes plain integers rather than process groups, so that the
        descriptors stay free of any dependency on :mod:`torch.distributed` and
        remain testable without a process group. The distributed layers translate
        their groups into these.

        How a grid decomposes is a property of the grid: a regular
        latitude--longitude grid splits as a product of a latitude range and a
        longitude range, but a reduced Gaussian grid has no single ``nlon`` to split,
        and an unstructured grid has no axes at all. Consumers should therefore ask
        the grid for a shard rather than compute ranges themselves.

        Parameters
        ----------
        polar : tuple of int, optional
            ``(rank, size)`` along the polar (latitude) direction, by default ``(0, 1)``.
        azimuth : tuple of int, optional
            ``(rank, size)`` along the azimuthal (longitude) direction, by default ``(0, 1)``.

        Returns
        -------
        GridShardS2
            The local piece, which knows the global grid it came from.
        """
        return GridShardS2(grid=self, polar_rank=polar[0], polar_size=polar[1], azimuth_rank=azimuth[0], azimuth_size=azimuth[1])

    def lat_shapes(self, num_chunks: int) -> Tuple[int, ...]:
        """Latitude counts held by each rank of a ``num_chunks``-way polar split."""
        return tuple(compute_split_shapes(self.nlat, num_chunks))

    def lon_shapes(self, num_chunks: int) -> Tuple[int, ...]:
        """Longitude counts held by each rank of a ``num_chunks``-way azimuthal split."""
        return tuple(compute_split_shapes(self.nlon, num_chunks))

    # -- construction and serialization --------------------------------------

    @classmethod
    def from_shape(cls, shape: Tuple[int, ...]) -> "GridS2":
        """
        Build this grid from a spatial shape, as :func:`as_grid` does.

        The hook that lets ``as_grid`` stay one function across grid families whose
        parameters differ: a product grid reads ``(nlat, nlon)`` off the shape, while
        HEALPix recovers its ``nside`` from a point count.
        """
        raise NotImplementedError(f"{cls.__name__} does not define from_shape")

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data representation, suitable for a config file or a checkpoint."""
        return {"grid": self.grid_type, **self.params}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GridS2":
        """Inverse of :meth:`to_dict`."""
        if "grid" not in data:
            raise ValueError("grid dict is missing ['grid']")
        grid_type = data["grid"]
        if grid_type not in _GRID_REGISTRY:
            raise ValueError(f"Unknown grid type {grid_type}, expected one of {list(_GRID_REGISTRY)}")

        cls = _GRID_REGISTRY[grid_type]
        expected = {field.name for field in fields(cls)}
        provided = set(data) - {"grid"}
        missing = expected - provided
        if missing:
            raise ValueError(f"grid dict is missing {sorted(missing)}")
        unexpected = provided - expected
        if unexpected:
            raise ValueError(f"grid dict for '{grid_type}' has unexpected keys {sorted(unexpected)}, expected {sorted(expected)}")

        return cls(**{name: data[name] for name in expected})


@dataclass(frozen=True, eq=False)
class GridShardS2:
    r"""
    One rank's piece of a decomposed :class:`GridS2`.

    A shard is deliberately **not** a :class:`GridS2`, because it is not a grid on the
    sphere. A band of latitudes does not cover :math:`S^2`, so:

    * its :attr:`quad_weights` do not sum to 2 -- they are the local contribution to
      an integral that a collective reduction completes;
    * quantities that describe the quadrature *rule* rather than this piece of it --
      the spectral bounds, the angular support radius -- are global, and a shard does
      not define them at all. Ask :attr:`global_grid` for them. Absent is a stronger
      guarantee than forwarded: a support radius derived from a shard's own node
      spacing would differ between ranks, and ranks disagreeing about the support of
      an operator is a correctness bug rather than an inefficiency.

    Making this a separate type keeps that distinction enforceable: a shard cannot be
    passed where a global grid is required, and :func:`require_grid` says so.

    Parameters
    ----------
    grid : GridS2
        The global grid this is a piece of.
    polar_rank, polar_size : int
        Position and extent of the decomposition along latitude.
    azimuth_rank, azimuth_size : int
        Position and extent of the decomposition along longitude.
    """

    grid: GridS2
    polar_rank: int = 0
    polar_size: int = 1
    azimuth_rank: int = 0
    azimuth_size: int = 1

    def __post_init__(self):
        if not isinstance(self.grid, GridS2):
            raise ValueError(f"grid must be a GridS2, got {type(self.grid).__name__}")
        for rank, size, name in [(self.polar_rank, self.polar_size, "polar"), (self.azimuth_rank, self.azimuth_size, "azimuth")]:
            if size < 1:
                raise ValueError(f"{name}_size must be at least 1, got {size}")
            if not 0 <= rank < size:
                raise ValueError(f"{name}_rank must lie in [0, {size}), got {rank}")

    # -- identity ------------------------------------------------------------

    @property
    def key(self) -> Tuple[Any, ...]:
        """Canonical identity, including the global grid's own key."""
        return (self.grid.key, self.polar_rank, self.polar_size, self.azimuth_rank, self.azimuth_size)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridShardS2):
            return NotImplemented
        return self.key == other.key

    def __repr__(self) -> str:
        return f"GridShardS2({self.grid!r}, polar={self.polar_rank}/{self.polar_size}, azimuth={self.azimuth_rank}/{self.azimuth_size})"

    # -- the global grid this came from --------------------------------------

    @property
    def global_grid(self) -> GridS2:
        """The undecomposed grid. Pass this wherever a global quantity is needed."""
        return self.grid

    @property
    def is_global(self) -> bool:
        """``False``; see :attr:`global_grid`."""
        return False

    # -- local extent --------------------------------------------------------

    @property
    def lat_shapes(self) -> Tuple[int, ...]:
        """Latitude counts held by every polar rank, ordered by rank."""
        return self.grid.lat_shapes(self.polar_size)

    @property
    def lon_shapes(self) -> Tuple[int, ...]:
        """Longitude counts held by every azimuthal rank, ordered by rank."""
        return self.grid.lon_shapes(self.azimuth_size)

    @property
    def nlat(self) -> int:
        """Number of latitudes on this rank."""
        return self.lat_shapes[self.polar_rank]

    @property
    def nlon(self) -> int:
        """Number of longitudes on this rank."""
        return self.lon_shapes[self.azimuth_rank]

    @property
    def lat_offset(self) -> int:
        """Index of this rank's first latitude within the global grid."""
        return sum(self.lat_shapes[: self.polar_rank])

    @property
    def lon_offset(self) -> int:
        """Index of this rank's first longitude within the global grid."""
        return sum(self.lon_shapes[: self.azimuth_rank])

    @property
    def shape(self) -> Tuple[int, int]:
        """Local spatial shape ``(nlat, nlon)``."""
        return (self.nlat, self.nlon)

    @property
    def npoints(self) -> int:
        """Number of grid points on this rank."""
        return self.nlat * self.nlon

    # -- local geometry ------------------------------------------------------

    @property
    def lats(self) -> torch.Tensor:
        """This rank's slice of the global colatitudes, shape ``(nlat,)``."""
        return self.grid.lats[self.lat_offset : self.lat_offset + self.nlat]

    @property
    def quad_weights(self) -> torch.Tensor:
        """
        This rank's slice of the global latitudinal weights, shape ``(nlat,)``.

        These sum to 2 only across all polar ranks; locally they are a partial sum.
        """
        return self.grid.quad_weights[self.lat_offset : self.lat_offset + self.nlat]

    def lons(self, ilat: Optional[int] = None) -> torch.Tensor:
        """This rank's slice of the longitudes of a latitude ring."""
        return self.grid.lons(ilat)[self.lon_offset : self.lon_offset + self.nlon]

    @property
    def is_regular(self) -> bool:
        """Whether every local latitude ring carries the same number of longitudes."""
        return self.grid.is_regular

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data representation, carrying the global grid with it."""
        return {"grid": self.grid.to_dict(), "polar_rank": self.polar_rank, "polar_size": self.polar_size, "azimuth_rank": self.azimuth_rank, "azimuth_size": self.azimuth_size}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GridShardS2":
        """Inverse of :meth:`to_dict`."""
        missing = {"grid", "polar_rank", "polar_size", "azimuth_rank", "azimuth_size"} - set(data)
        if missing:
            raise ValueError(f"grid shard dict is missing {sorted(missing)}")
        return GridShardS2(
            grid=GridS2.from_dict(data["grid"]),
            polar_rank=data["polar_rank"],
            polar_size=data["polar_size"],
            azimuth_rank=data["azimuth_rank"],
            azimuth_size=data["azimuth_size"],
        )


@dataclass(frozen=True, eq=False)
class RegularGridS2(GridS2, abstract=True):
    r"""
    A latitude/longitude product grid: ``nlon`` equispaced longitudes on each of
    ``nlat`` latitude rings.

    Abstract in itself -- the latitude nodes and weights come from a quadrature rule,
    which is what the concrete subclasses select via their ``grid_type`` -- but it
    holds everything that follows from the product structure: a rectangular
    ``(nlat, nlon)`` storage shape, a uniform ``2 * pi / nlon`` longitude spacing on
    every ring, and a decomposition that is the product of a latitude range and a
    longitude range.

    Parameters
    ----------
    nlat : int
        Number of latitudinal nodes. Must be at least 2.
    nlon : int
        Number of longitudinal nodes. Must be at least 1.
    """

    nlat: int
    nlon: int

    def _validate(self):
        if not isinstance(self.nlat, int) or isinstance(self.nlat, bool):
            raise ValueError(f"nlat must be an int, got {type(self.nlat).__name__}")
        if not isinstance(self.nlon, int) or isinstance(self.nlon, bool):
            raise ValueError(f"nlon must be an int, got {type(self.nlon).__name__}")
        if self.nlat < 2:
            raise ValueError(f"nlat must be at least 2, got {self.nlat}")
        if self.nlon < 1:
            raise ValueError(f"nlon must be at least 1, got {self.nlon}")

    @property
    def params(self) -> Dict[str, Any]:
        return {"nlat": self.nlat, "nlon": self.nlon}

    @property
    def shape(self) -> Tuple[int, int]:
        """Spatial shape ``(nlat, nlon)`` of a field sampled on this grid."""
        return (self.nlat, self.nlon)

    @property
    def npoints(self) -> int:
        """Total number of grid points."""
        return self.nlat * self.nlon

    # -- geometry ------------------------------------------------------------

    @property
    def lats(self) -> torch.Tensor:
        lats, _ = precompute_latitudes(self.nlat, grid=self.grid_type)
        return lats

    @property
    def quad_weights(self) -> torch.Tensor:
        _, w = precompute_latitudes(self.nlat, grid=self.grid_type)
        return w

    def lons(self, ilat: Optional[int] = None) -> torch.Tensor:
        return precompute_longitudes(self.nlon)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_shape(cls, shape: Tuple[int, ...]) -> "RegularGridS2":
        if len(shape) != 2:
            raise ValueError(f"shape must be a 2-tuple (nlat, nlon) for grid '{cls.grid_type}', got length {len(shape)}")
        nlat, nlon = shape
        return cls(nlat=int(nlat), nlon=int(nlon))


@dataclass(frozen=True, eq=False)
class EquiangularGrid(RegularGridS2):
    r"""
    Equiangular grid with Clenshaw--Curtis quadrature.

    Nodes are equally spaced in :math:`\theta` and include both poles, so the
    latitudinal spacing is exactly :math:`\pi / (N_\theta - 1)` everywhere. This is
    the default grid throughout torch-harmonics.
    """

    grid_type: ClassVar[str] = "equiangular"

    @property
    def is_uniform_in_theta(self) -> bool:
        return True

    @property
    def max_exact_degree(self) -> int:
        r"""Clenshaw--Curtis is exact to roughly degree :math:`N_\theta - 1`, giving :math:`\lfloor (N_\theta + 1) / 2 \rfloor`."""
        return (self.nlat + 1) // 2


@dataclass(frozen=True, eq=False)
class LegendreGaussGrid(RegularGridS2):
    r"""
    Gauss--Legendre grid; nodes are the roots of :math:`P_N(\cos\theta)`.

    Optimal quadrature accuracy per node, exact for polynomials up to degree
    :math:`2N - 1`, but the nodes exclude the poles and are not uniform in
    :math:`\theta`.
    """

    grid_type: ClassVar[str] = "legendre-gauss"

    @property
    def max_exact_degree(self) -> int:
        r"""Gauss--Legendre is exact to degree :math:`2N_\theta - 1`, giving :math:`N_\theta`."""
        return self.nlat


@dataclass(frozen=True, eq=False)
class LobattoGrid(RegularGridS2):
    r"""
    Gauss--Lobatto grid; nodes are the roots of :math:`P'_{N-1}(\cos\theta)` plus both poles.

    Nodes cluster towards the equator, so the polar spacing is noticeably coarser
    than :math:`\pi / (N_\theta - 1)`.
    """

    grid_type: ClassVar[str] = "lobatto"

    @property
    def max_exact_degree(self) -> int:
        r"""Gauss--Lobatto is exact to degree :math:`2N_\theta - 3`, giving :math:`N_\theta - 1`."""
        return self.nlat - 1


@dataclass(frozen=True, eq=False)
class EquiangularTrapezoidalGrid(RegularGridS2):
    r"""
    Trapezoidal rule applied on the :math:`\cos\theta` interval :math:`[-1, 1]`.

    Despite the name, the nodes are **not** equiangular in :math:`\theta`: they are
    equispaced in :math:`\cos\theta`, which makes the spacing in :math:`\theta`
    strongly non-uniform. The polar spacing is a factor :math:`\sqrt{N_\theta - 1}`
    coarser than the equatorial one, so the disparity grows with resolution instead
    of staying fixed.
    """

    grid_type: ClassVar[str] = "equiangular-trapezoidal"

    @property
    def max_exact_degree(self) -> int:
        r"""
        Matches the equiangular grid, :math:`\lfloor (N_\theta + 1) / 2 \rfloor`.

        Retained for backwards compatibility, but see
        :attr:`is_spectrally_accurate`: the trapezoidal rule is not accurate
        enough to reach this degree, so the value is optimistic.
        """
        return (self.nlat + 1) // 2

    @property
    def is_spectrally_accurate(self) -> bool:
        """
        ``False``. The trapezoidal rule converges only algebraically, as :math:`O(h^2)`.

        The consequence for an SHT is severe, because the default truncation grows
        with resolution faster than the accuracy does. Measured round-trip relative
        error at ``nlat = 64``, against ~1e-15 for the interpolatory rules:

        ========  ========
        ``lmax``  rel. err
        ========  ========
        1         9e-16
        2         2.2e-4
        4         1.6e-3
        8         2.4e-2
        16        1.3e-1
        32        6.7e-1
        ========  ========

        Only ``lmax = 1`` -- the constant mode -- is exact; the rule integrates
        functions linear in :math:`\\cos\theta` without error, and nothing beyond.
        ``lmax = 32`` is the default this grid is assigned at ``nlat = 64``. Refining
        the grid helps only as :math:`n^{-2}`, so this grid is usable for a transform
        at very low truncation and not otherwise. It remains perfectly
        serviceable for plain quadrature and for the localized operators.
        """
        return False


def require_grid(grid: Any, name: Optional[str] = "grid") -> GridS2:
    """
    Validate that a layer received a grid descriptor, with a migration-friendly error.

    Layers used to take a shape plus a grid name; they now take a single
    :class:`GridS2`. Passing either of the old arguments would otherwise fail deep
    inside the constructor with an opaque ``AttributeError``, so intercept it here
    and say what to write instead.

    Parameters
    ----------
    grid : Any
        The value supplied by the caller.
    name : str, optional
        Name of the parameter, used in the error message, by default ``"grid"``.

    Returns
    -------
    GridS2
        ``grid`` unchanged, once validated.

    Raises
    ------
    TypeError
        If ``grid`` is not a :class:`GridS2`.
    """
    if isinstance(grid, GridS2):
        return grid
    if isinstance(grid, GridShardS2):
        raise TypeError(
            f"{name} must be the global GridS2, not a shard of one. Quantities such as the spectral bounds and the angular cutoff are global; " f"pass {name}.global_grid."
        )
    if isinstance(grid, str):
        raise TypeError(f"{name} must be a GridS2, not the grid name {grid!r}. The descriptor carries the resolution too, so build one with " f"as_grid({grid!r}, (nlat, nlon)).")
    if isinstance(grid, (tuple, list)):
        shape = tuple(grid)
        raise TypeError(f"{name} must be a GridS2, not a shape {shape!r}. The descriptor carries the shape, so pass as_grid(<grid name>, {shape!r}) instead.")
    raise TypeError(f"{name} must be a GridS2, got {type(grid).__name__}. Build one with as_grid(<grid name>, (nlat, nlon)).")


def grid_types(regular: Optional[bool] = None) -> Tuple[str, ...]:
    """
    Names of the registered grid types, in registration order.

    Parameters
    ----------
    regular : bool, optional
        If given, keep only the grids whose points do (``True``) or do not
        (``False``) form a latitude/longitude product. Callers that build a grid
        from a ``(nlat, nlon)`` shape, or that index with a uniform ``nlon`` stride,
        want ``regular=True``; by default every registered type is returned.

    Returns
    -------
    tuple of str
        The matching grid type names.

    Examples
    --------
    >>> from torch_harmonics import grid_types
    >>> grid_types(regular=True)
    ('equiangular', 'legendre-gauss', 'lobatto', 'equiangular-trapezoidal')
    """
    names = tuple(_GRID_REGISTRY)
    if regular is None:
        return names
    return tuple(name for name in names if issubclass(_GRID_REGISTRY[name], RegularGridS2) == regular)


def as_grid(spec: Union[GridS2, str], shape: Optional[Tuple[int, int]] = None) -> GridS2:
    """
    Coerce a grid specification into a :class:`GridS2`.

    Lets a layer accept either a descriptor or the historical
    ``grid="equiangular"`` string plus a shape, so the descriptor API can be
    introduced without breaking existing call sites.

    Parameters
    ----------
    spec : GridS2 or str
        A descriptor, which is returned unchanged, or a grid type name.
    shape : tuple of int, optional
        The spatial shape the named grid should have: ``(nlat, nlon)`` for a product
        grid and ``(npoints,)`` for a ragged one. Required when ``spec`` is a string,
        and must agree with the descriptor when one is passed.

    Returns
    -------
    GridS2
        The corresponding descriptor.

    Raises
    ------
    ValueError
        If the grid type is unknown, if ``shape`` is missing for a string spec, if it
        does not describe a valid resolution for that grid, or if it contradicts a
        descriptor spec.

    Examples
    --------
    >>> from torch_harmonics import as_grid
    >>> as_grid("equiangular", (128, 256))
    EquiangularGrid(nlat=128, nlon=256)
    """
    if isinstance(spec, GridS2):
        if shape is not None and tuple(shape) != spec.spatial_shape:
            raise ValueError(f"shape {tuple(shape)} contradicts the grid descriptor {spec}")
        return spec

    if not isinstance(spec, str):
        raise ValueError(f"expected a GridS2 or a grid type name, got {type(spec).__name__}")

    if spec not in _GRID_REGISTRY:
        raise ValueError(f"Unknown grid type {spec}, expected one of {list(_GRID_REGISTRY)}")

    if shape is None:
        raise ValueError(f"shape is required when specifying a grid by name (got grid='{spec}')")

    # each family reads its own parameters off the shape, since a ragged grid has no
    # (nlat, nlon) to unpack
    return _GRID_REGISTRY[spec].from_shape(tuple(shape))
