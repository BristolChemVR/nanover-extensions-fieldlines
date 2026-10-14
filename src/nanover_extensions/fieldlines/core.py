"""Electric field of point charges and field line tracing.

Units: positions in angstrom, charges in elementary charges, fields in MV/cm and
potentials in volts. The field of fixed point charges, optionally with Debye-Hueckel
screening and a dielectric constant, comes either from a direct Coulomb sum
(`CoulombField`, which can also apply a distance cutoff) or from a grid (`GridField`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt
from scipy import fft
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from collections.abc import Iterator

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.intp]

VOLT_ANGSTROM = 14.399645
"""Potential in volts at 1 angstrom from one elementary charge in vacuum."""

MV_PER_CM = 1439.9645
"""Conversion from e/A^2 to MV/cm (14.399645 V*A/e, and 1 V/A is 100 MV/cm)."""

_MAX_DENSE_PAIRS = 2_000_000
"""Upper bound on point-charge pairs held in memory at once when no cutoff is set."""

_SAME_POINT = 1e-6
"""Charges closer than this (angstrom) to a bond atom are taken to be that atom."""


def _as_vectors(values: npt.ArrayLike, name: str) -> FloatArray:
    """Convert to a float (N, 3) array, rejecting any other shape."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:  # noqa: PLR2004
        raise ValueError(f"`{name}` must have shape (N, 3), not {array.shape}.")
    return array


def _validate(
    positions: npt.ArrayLike,
    charges: npt.ArrayLike,
    debye_length: float | None,
    dielectric: float,
) -> tuple[FloatArray, FloatArray]:
    """Validate the arguments shared by the field classes, returning positions and charges."""
    points = _as_vectors(positions, "positions")
    values = np.asarray(charges, dtype=np.float64)
    if values.shape != (len(points),):
        raise ValueError(f"`charges` must have shape ({len(points)},), not {values.shape}.")
    if len(points) == 0:
        msg = "At least one charge is required."
        raise ValueError(msg)
    if debye_length is not None and debye_length <= 0:
        raise ValueError(f"`debye_length` must be positive, not {debye_length}.")
    if dielectric <= 0:
        raise ValueError(f"`dielectric` must be positive, not {dielectric}.")
    return points, values


def _unit_potential(
    r: FloatArray,
    debye_length: float | None,
    dielectric: float,
    distance_dielectric: bool,  # noqa: FBT001
) -> FloatArray:
    """Potential in volts of one elementary charge at distances `r` > 0 (angstrom)."""
    permittivity = dielectric * r if distance_dielectric else dielectric
    potential = VOLT_ANGSTROM / (permittivity * r)
    if debye_length is not None:
        potential = potential * np.exp(-r / debye_length)
    return np.asarray(potential, dtype=np.float64)


class Field(Protocol):
    """The electric field and potential of a set of point charges."""

    @property
    def positions(self) -> FloatArray:
        """Charge positions in angstrom, shape (N, 3)."""

    def evaluate(self, points: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Return the field in MV/cm (P, 3) and the distance to the nearest charge (P,)."""

    def potential(self, points: npt.ArrayLike) -> FloatArray:
        """Return the potential in volts, shape (P,)."""


class CoulombField:
    """Electric field of a fixed set of point charges.

    Args:
        positions: Charge positions in angstrom, shape (N, 3).
        charges: Charges in elementary charges, shape (N,).
        debye_length: Debye-Hueckel screening length in angstrom. No screening if None.
        cutoff: Charges further than this from a point (angstrom) do not contribute to
            the field there. All charges contribute if None.
        dielectric: Relative permittivity of the medium. With `distance_dielectric`,
            the slope `k` of a permittivity `k * r` with `r` in angstrom.
        distance_dielectric: Use the distance-dependent permittivity `dielectric * r`.
            The field is then minus the gradient of the potential `q / (k r^2)`, which
            is twice what dividing the bare field by `k r` would give.
    """

    def __init__(
        self,
        positions: npt.ArrayLike,
        charges: npt.ArrayLike,
        *,
        debye_length: float | None = None,
        cutoff: float | None = None,
        dielectric: float = 1.0,
        distance_dielectric: bool = False,
    ) -> None:
        self._positions, self._charges = _validate(positions, charges, debye_length, dielectric)
        if cutoff is not None and cutoff <= 0:
            raise ValueError(f"`cutoff` must be positive, not {cutoff}.")

        self.debye_length = debye_length
        self.cutoff = cutoff
        self.dielectric = dielectric
        self.distance_dielectric = distance_dielectric
        self._tree = None if cutoff is None else cKDTree(self._positions)

    @property
    def positions(self) -> FloatArray:
        """Charge positions in angstrom, shape (N, 3)."""
        return self._positions

    @property
    def charges(self) -> FloatArray:
        """Charges in elementary charges, shape (N,)."""
        return self._charges

    def without(self, indices: npt.ArrayLike) -> CoulombField:
        """Return the same field with some charges removed.

        Args:
            indices: Indices of the charges to remove.

        Returns:
            A new field with the same settings.
        """
        keep = np.ones(len(self._positions), dtype=bool)
        keep[np.asarray(indices, dtype=np.intp)] = False
        return CoulombField(
            self._positions[keep],
            self._charges[keep],
            debye_length=self.debye_length,
            cutoff=self.cutoff,
            dielectric=self.dielectric,
            distance_dielectric=self.distance_dielectric,
        )

    def evaluate(self, points: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Evaluate the field at a set of points.

        A charge sitting exactly on a point contributes nothing to the field there.

        Args:
            points: Positions in angstrom, shape (P, 3).

        Returns:
            The field in MV/cm, shape (P, 3), and the distance from each point to its
            nearest charge in angstrom, shape (P,). The distance ignores the cutoff.
        """
        points = _as_vectors(points, "points")
        if len(points) == 0:
            return np.zeros((0, 3)), np.zeros(0)
        if self._tree is None:
            return self._evaluate_dense(points)
        return self._evaluate_cutoff(points, self._tree)

    def _weights(self, r2: FloatArray, charges: FloatArray) -> FloatArray:
        """Scalar that multiplies the charge-to-point vector to give the field in e/A^2."""
        # For a potential q exp(-x) / (eps r) with x = r / debye_length, the field is
        # q exp(-x) (1 + x) / (eps r^2) for constant eps, and q exp(-x) (2 + x) / (k r^3)
        # for eps = k r. Dividing by r once more turns the vector to the point into the field.
        r = np.sqrt(r2)
        x = 0.0 if self.debye_length is None else r / self.debye_length
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.distance_dielectric:
                weights = charges * (2.0 + x) / (self.dielectric * r2 * r2)
            else:
                weights = charges * (1.0 + x) / (self.dielectric * r2 * r)
            if self.debye_length is not None:
                weights = weights * np.exp(-x)
        return np.where(r2 > 0, weights, 0.0)

    def _evaluate_dense(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        field = np.empty_like(points)
        nearest = np.empty(len(points))
        chunk = max(1, _MAX_DENSE_PAIRS // len(self._positions))
        for start in range(0, len(points), chunk):
            block = slice(start, start + chunk)
            delta = points[block, None, :] - self._positions[None, :, :]
            r2 = np.einsum("pnk,pnk->pn", delta, delta)
            weights = self._weights(r2, self._charges[None, :])
            field[block] = np.einsum("pnk,pn->pk", delta, weights)
            nearest[block] = np.sqrt(r2.min(axis=1))
        return MV_PER_CM * field, nearest

    def _evaluate_cutoff(self, points: FloatArray, tree: cKDTree) -> tuple[FloatArray, FloatArray]:
        pairs = cKDTree(points).sparse_distance_matrix(tree, self.cutoff, output_type="ndarray")
        point_index, charge_index = pairs["i"], pairs["j"]
        delta = points[point_index] - self._positions[charge_index]
        r2 = np.einsum("ik,ik->i", delta, delta)
        terms = delta * self._weights(r2, self._charges[charge_index])[:, None]
        field = np.empty_like(points)
        for axis in range(3):
            field[:, axis] = np.bincount(point_index, weights=terms[:, axis], minlength=len(points))
        nearest, _ = tree.query(points)
        return MV_PER_CM * field, np.asarray(nearest, dtype=np.float64)

    def potential(self, points: npt.ArrayLike) -> FloatArray:
        """Evaluate the potential at a set of points.

        A charge sitting exactly on a point contributes nothing to the potential there.

        Args:
            points: Positions in angstrom, shape (P, 3).

        Returns:
            The potential in volts, shape (P,).
        """
        points = _as_vectors(points, "points")
        if self._tree is None:
            potential = np.zeros(len(points))
            chunk = max(1, _MAX_DENSE_PAIRS // len(self._positions))
            for start in range(0, len(points), chunk):
                block = slice(start, start + chunk)
                r = np.linalg.norm(points[block, None, :] - self._positions[None, :, :], axis=2)
                potential[block] = self._pair_potentials(r, self._charges[None, :]).sum(axis=1)
            return potential
        pairs = cKDTree(points).sparse_distance_matrix(
            self._tree,
            self.cutoff,
            output_type="ndarray",
        )
        terms = self._pair_potentials(pairs["v"], self._charges[pairs["j"]])
        return np.bincount(pairs["i"], weights=terms, minlength=len(points)).astype(np.float64)

    def _pair_potentials(self, r: FloatArray, charges: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore", invalid="ignore"):
            unit = _unit_potential(r, self.debye_length, self.dielectric, self.distance_dielectric)
        return np.where(r > 0, charges * unit, 0.0)


@lru_cache(maxsize=8)
def _kernel_spectrum(
    shape: tuple[int, int, int],
    spacing: float,
    debye_length: float | None,
    dielectric: float,
    distance_dielectric: bool,  # noqa: FBT001
) -> npt.NDArray[np.complex128]:
    """Fourier transform of the potential of a unit charge at the grid origin.

    Distances wrap around the grid (minimum image), so that a convolution with it is
    periodic in `shape`. The potential within half a cell of the charge is capped at
    its value at half a cell.
    """
    axes = [spacing * np.minimum(np.arange(n), n - np.arange(n)) for n in shape]
    r = np.sqrt(axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2 + axes[2] ** 2)
    kernel = _unit_potential(
        np.maximum(r, 0.5 * spacing),
        debye_length,
        dielectric,
        distance_dielectric,
    )
    spectrum = np.asarray(fft.rfftn(kernel), dtype=np.complex128)
    spectrum.flags.writeable = False  # Shared between all grids of this shape.
    return spectrum


def _spread_charges(
    coordinates: FloatArray,
    charges: FloatArray,
    shape: tuple[int, int, int],
) -> FloatArray:
    """Share each charge between its eight surrounding grid points (cloud in cell).

    `coordinates` are in grid units and inside the grid.
    """
    base = np.floor(coordinates).astype(np.intp)
    fraction = coordinates - base
    indices, weights = [], []
    for corner in itertools.product((0, 1), repeat=3):
        offset = np.array(corner)
        indices.append(np.ravel_multi_index(tuple((base + offset).T), shape))
        weights.append(charges * np.prod(np.where(offset, fraction, 1.0 - fraction), axis=1))
    total = np.bincount(np.concatenate(indices), np.concatenate(weights), minlength=np.prod(shape))
    return total.astype(np.float64).reshape(shape)


class GridField:
    """Electric field of a fixed set of point charges, computed on a grid.

    The charges are spread onto a grid and convolved with the potential of one charge by
    FFT, zero-padded so it does not wrap around (Hockney and Eastwood). The field is minus
    the gradient of the potential, and both are interpolated linearly and zero outside
    the grid. More than three spacings from any charge, the field matches `CoulombField`
    to about 2 % (median); closer in it is smeared out. Halving the spacing cuts the
    error by about four.

    Args:
        positions: Charge positions in angstrom, shape (N, 3).
        charges: Charges in elementary charges, shape (N,).
        debye_length: Debye-Hueckel screening length in angstrom. No screening if None.
        dielectric: Relative permittivity, or its slope per angstrom with
            `distance_dielectric`, as for `CoulombField`.
        distance_dielectric: Use the distance-dependent permittivity `dielectric * r`.
        spacing: Grid spacing in angstrom.
        margin: How far the grid reaches past the charges and `cover`, in angstrom.
        cover: More points the grid must reach, shape (C, 3).
    """

    def __init__(
        self,
        positions: npt.ArrayLike,
        charges: npt.ArrayLike,
        *,
        debye_length: float | None = None,
        dielectric: float = 1.0,
        distance_dielectric: bool = False,
        spacing: float = 1.0,
        margin: float = 12.0,
        cover: npt.ArrayLike | None = None,
    ) -> None:
        self._positions, charges = _validate(positions, charges, debye_length, dielectric)
        if spacing <= 0:
            raise ValueError(f"`spacing` must be positive, not {spacing}.")
        if margin < 0:
            raise ValueError(f"`margin` must not be negative, not {margin}.")

        reach = self._positions
        if cover is not None:
            reach = np.concatenate([reach, _as_vectors(cover, "cover")])
        self._origin = reach.min(axis=0) - margin
        extent = reach.max(axis=0) + margin - self._origin
        # Two extra points so that every charge has grid points on both sides.
        nx, ny, nz = (int(n) + 2 for n in np.ceil(extent / spacing))
        self._shape = (nx, ny, nz)
        self.spacing = spacing

        padded = (
            fft.next_fast_len(2 * nx - 1),
            fft.next_fast_len(2 * ny - 1),
            fft.next_fast_len(2 * nz - 1, real=True),
        )
        rho = _spread_charges((self._positions - self._origin) / spacing, charges, self._shape)
        kernel = _kernel_spectrum(padded, spacing, debye_length, dielectric, distance_dielectric)
        potential = fft.irfftn(fft.rfftn(rho, padded) * kernel, padded)[:nx, :ny, :nz]
        self._potential = np.ascontiguousarray(potential, dtype=np.float64)
        # 1 V/A is 100 MV/cm.
        self._field = -100.0 * np.stack(list(np.gradient(self._potential, spacing)))
        self._tree = cKDTree(self._positions)

    @property
    def positions(self) -> FloatArray:
        """Charge positions in angstrom, shape (N, 3)."""
        return self._positions

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of grid points along each axis."""
        return self._shape

    def evaluate(self, points: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Evaluate the field at a set of points.

        Args:
            points: Positions in angstrom, shape (P, 3).

        Returns:
            The field in MV/cm, shape (P, 3), and the distance from each point to its
            nearest charge in angstrom, shape (P,).
        """
        points = _as_vectors(points, "points")
        field = np.stack([self._sample(self._field[axis], points) for axis in range(3)], axis=1)
        nearest, _ = self._tree.query(points)
        return field, np.asarray(nearest, dtype=np.float64).reshape(len(points))

    def potential(self, points: npt.ArrayLike) -> FloatArray:
        """Evaluate the potential at a set of points.

        Args:
            points: Positions in angstrom, shape (P, 3).

        Returns:
            The potential in volts, shape (P,).
        """
        return self._sample(self._potential, _as_vectors(points, "points"))

    def _sample(self, values: FloatArray, points: FloatArray) -> FloatArray:
        """Interpolate grid values linearly at `points`, with zero outside the grid."""
        coordinates = ((points - self._origin) / self.spacing).T
        last = np.array(self._shape, dtype=np.float64)[:, None] - 1.0
        inside = np.all((coordinates >= 0.0) & (coordinates <= last), axis=0)
        sampled = map_coordinates(values, coordinates, order=1, mode="nearest")
        return np.where(inside, sampled, 0.0)


_ANCHOR_CANDIDATES = 8
"""Nearest atoms looked at when choosing the three atoms a seed follows."""

_FLAT = 0.1
"""Anchor atoms closer than this (angstrom) to each other, or to a line through two of
them, cannot fix a direction, so the seed only follows the first atom."""


def _frames(first: FloatArray, second: FloatArray, third: FloatArray) -> FloatArray:
    """Build an orthonormal frame from each triple of points, shape (S, 3, 3).

    The rows are the axes: from `first` towards `second`, then towards `third` at right
    angles to that, then their cross product. Triples that cannot fix a direction get
    the identity.
    """
    along = second - first
    along_length = np.linalg.norm(along, axis=1, keepdims=True)
    along = along / np.maximum(along_length, _FLAT)
    across = third - first
    across = across - np.sum(across * along, axis=1, keepdims=True) * along
    across_length = np.linalg.norm(across, axis=1, keepdims=True)
    across = across / np.maximum(across_length, _FLAT)
    frames = np.stack([along, across, np.cross(along, across)], axis=1)
    frames[(along_length[:, 0] < _FLAT) | (across_length[:, 0] < _FLAT)] = np.eye(3)
    return frames


@dataclass(frozen=True, eq=False)
class Seeds:
    """Starting points of field lines, each tied to three nearby atoms so it can follow them.

    A seed is stored in a frame built from its atoms, so when those atoms move or turn
    together, the seed moves and turns with them (see `follow`).

    Attributes:
        points: Seed positions in angstrom, shape (S, 3).
        ids: A number for each seed's line, kept for as long as the seed is, shape (S,).
        atoms: Indices of the three atoms each seed follows, shape (S, 3).
        local: Each seed in the frame of its atoms, shape (S, 3).
        next_id: The number for the next new seed.
    """

    points: FloatArray
    ids: IndexArray
    atoms: IndexArray
    local: FloatArray
    next_id: int

    def __len__(self) -> int:
        return len(self.points)

    @classmethod
    def anchor(
        cls,
        points: npt.ArrayLike,
        ids: npt.ArrayLike,
        positions: FloatArray,
        atoms: IndexArray,
        next_id: int,
    ) -> Seeds:
        """Tie each seed (shape (S, 3)) to the nearest of `atoms` and two more.

        The second atom is the next nearest, and the third whichever of the nearest
        few lies furthest from the line through the first two.

        Raises:
            ValueError: If there are no atoms to follow.
        """
        points = _as_vectors(points, "points")
        pool = np.asarray(atoms, dtype=np.intp)
        if len(pool) == 0:
            msg = "Seeds need at least one atom to follow."
            raise ValueError(msg)
        count = min(_ANCHOR_CANDIDATES, len(pool))
        _, near = cKDTree(positions[pool]).query(points, k=count)
        near = np.asarray(near, dtype=np.intp).reshape(len(points), count)
        first, second = near[:, 0], near[:, min(1, count - 1)]
        third = second
        if count > 2:  # noqa: PLR2004
            start = positions[pool[first]]
            axis = positions[pool[second]] - start
            axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), _FLAT)
            # Distance from the axis is the length of the cross product with it.
            cross = np.cross(positions[pool[near[:, 2:]]] - start[:, None], axis[:, None])
            furthest = np.linalg.norm(cross, axis=2).argmax(axis=1)
            third = near[np.arange(len(points)), 2 + furthest]
        chosen = pool[np.stack([first, second, third], axis=1)]
        corners = [positions[chosen[:, i]] for i in range(3)]
        local = np.einsum("sij,sj->si", _frames(*corners), points - corners[0])
        return cls(points, np.asarray(ids, dtype=np.intp), chosen, local, next_id)

    def follow(self, positions: FloatArray) -> FloatArray:
        """Return where the seeds are with their atoms at `positions`, shape (S, 3)."""
        corners = [positions[self.atoms[:, i]] for i in range(3)]
        offsets: FloatArray = np.einsum("sij,si->sj", _frames(*corners), self.local)
        return corners[0] + offsets


@dataclass(frozen=True, eq=False)
class FieldLines:
    """A set of field lines stored as flat arrays.

    Line `k` is `points[offsets[k]:offsets[k + 1]]`, ordered along the field direction
    (from positive towards negative charge).

    Attributes:
        points: Line vertices in angstrom, shape (M, 3).
        strengths: Field magnitude at each vertex in MV/cm, shape (M,).
        offsets: Start of each line in `points`, plus the total, shape (L + 1,).
        potentials: Potential at each vertex in volts, shape (M,), if it was evaluated.
        seeds: The seeds the lines were traced from, if recorded. Lines that were
            dropped afterwards (for being too short) keep their seeds here.
        ids: The `seeds.ids` number of each line, shape (L,), if recorded.
    """

    points: FloatArray
    strengths: FloatArray
    offsets: IndexArray
    potentials: FloatArray | None = None
    seeds: Seeds | None = None
    ids: IndexArray | None = None

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __iter__(self) -> Iterator[tuple[FloatArray, FloatArray]]:
        for start, end in zip(self.offsets[:-1], self.offsets[1:], strict=True):
            yield self.points[start:end], self.strengths[start:end]


def _march(
    field: Field,
    seeds: FloatArray,
    direction: float,
    *,
    step: float,
    max_steps: int,
    stop_distance: float,
    min_strength: float,
) -> tuple[FloatArray, FloatArray, IndexArray]:
    """Step every seed along (or against) the field until it stops.

    Returns the visited points (max_steps + 1, n, 3), the field magnitude at each
    (max_steps + 1, n), and how many leading entries are valid for each seed (n,).
    """
    n = len(seeds)
    current = seeds.copy()
    path = np.zeros((max_steps + 1, n, 3))
    strengths = np.zeros((max_steps + 1, n))
    counts = np.zeros(n, dtype=np.intp)
    alive = np.ones(n, dtype=bool)
    for i in range(max_steps + 1):
        index = np.flatnonzero(alive)
        if len(index) == 0:
            break
        vectors, nearest = field.evaluate(current[index])
        magnitude = np.linalg.norm(vectors, axis=1)
        path[i, index] = current[index]
        strengths[i, index] = magnitude
        counts[index] += 1
        ok = (nearest > stop_distance) & (magnitude > min_strength)
        alive[index[~ok]] = False
        current[index[ok]] += direction * step * vectors[ok] / magnitude[ok, None]
    return path, strengths, counts


def trace(
    field: Field,
    seeds: npt.ArrayLike,
    *,
    step: float = 0.25,
    max_steps: int = 100,
    stop_distance: float = 0.9,
    min_strength: float = 1e-6,
) -> FieldLines:
    """Trace a field line forwards and backwards through each seed.

    A line stops in a given direction after `max_steps` steps, when it comes within
    `stop_distance` of a charge, or when the field drops to `min_strength`. The point
    at which it stops is kept.

    Args:
        field: The field to trace.
        seeds: Starting points in angstrom, shape (S, 3).
        step: Step length in angstrom.
        max_steps: Maximum number of steps in each direction.
        stop_distance: Distance to the nearest charge (angstrom) at which a line ends.
        min_strength: Field magnitude (MV/cm) at or below which a line ends.

    Returns:
        One line per seed, in seed order.
    """
    seeds = _as_vectors(seeds, "seeds")
    if step <= 0:
        raise ValueError(f"`step` must be positive, not {step}.")
    if max_steps < 0:
        raise ValueError(f"`max_steps` must not be negative, not {max_steps}.")
    if len(seeds) == 0:
        return FieldLines(np.zeros((0, 3)), np.zeros(0), np.zeros(1, dtype=np.intp))

    halves = [
        _march(
            field,
            seeds,
            direction,
            step=step,
            max_steps=max_steps,
            stop_distance=stop_distance,
            min_strength=min_strength,
        )
        for direction in (1.0, -1.0)
    ]
    (fwd_path, fwd_strength, fwd_count), (bwd_path, bwd_strength, bwd_count) = halves

    points, strengths = [], []
    for k in range(len(seeds)):
        # The seed itself is the first entry of both halves; take it from the forward one.
        backward = slice(bwd_count[k] - 1, 0, -1)
        forward = slice(0, fwd_count[k])
        points += [bwd_path[backward, k], fwd_path[forward, k]]
        strengths += [bwd_strength[backward, k], fwd_strength[forward, k]]
    offsets = np.concatenate([[0], np.cumsum(bwd_count - 1 + fwd_count)]).astype(np.intp)
    return FieldLines(np.concatenate(points), np.concatenate(strengths), offsets)


def bond_field(
    field: CoulombField,
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    *,
    include_self: bool = False,
) -> float:
    """Field at the midpoint of a bond, projected on the bond direction.

    By default the charges of the two bond atoms are left out, so that the result is
    the field the environment puts on the bond (as in vibrational Stark analysis).
    A charge counts as a bond atom when it sits on `start` or `end`.

    Args:
        field: The field to evaluate.
        start: Position of the first atom in angstrom, shape (3,).
        end: Position of the second atom in angstrom, shape (3,).
        include_self: Keep the charges of the bond atoms.

    Returns:
        The component of the field along `start` -> `end` in MV/cm.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if not include_self:
        on_bond = (
            np.minimum(
                np.linalg.norm(field.positions - start, axis=1),
                np.linalg.norm(field.positions - end, axis=1),
            )
            < _SAME_POINT
        )
        if on_bond.all():
            return 0.0
        if on_bond.any():
            field = field.without(np.flatnonzero(on_bond))
    vectors, _ = field.evaluate((0.5 * (start + end))[None, :])
    axis = (end - start) / np.linalg.norm(end - start)
    return float(vectors[0] @ axis)
