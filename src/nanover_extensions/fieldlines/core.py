"""Electric field of point charges and field line tracing.

Units: positions in angstrom, charges in elementary charges, fields in MV/cm.
The field is a plain Coulomb sum over fixed point charges, optionally with
Debye-Hueckel screening, a dielectric constant and a distance cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from collections.abc import Iterator

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.intp]

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
        self._positions = _as_vectors(positions, "positions")
        self._charges = np.asarray(charges, dtype=np.float64)
        if self._charges.shape != (len(self._positions),):
            raise ValueError(
                f"`charges` must have shape ({len(self._positions)},), not {self._charges.shape}.",
            )
        if len(self._positions) == 0:
            msg = "At least one charge is required."
            raise ValueError(msg)
        if debye_length is not None and debye_length <= 0:
            raise ValueError(f"`debye_length` must be positive, not {debye_length}.")
        if cutoff is not None and cutoff <= 0:
            raise ValueError(f"`cutoff` must be positive, not {cutoff}.")
        if dielectric <= 0:
            raise ValueError(f"`dielectric` must be positive, not {dielectric}.")

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


@dataclass(frozen=True, eq=False)
class FieldLines:
    """A set of field lines stored as flat arrays.

    Line `k` is `points[offsets[k]:offsets[k + 1]]`, ordered along the field direction
    (from positive towards negative charge).

    Attributes:
        points: Line vertices in angstrom, shape (M, 3).
        strengths: Field magnitude at each vertex in MV/cm, shape (M,).
        offsets: Start of each line in `points`, plus the total, shape (L + 1,).
    """

    points: FloatArray
    strengths: FloatArray
    offsets: IndexArray

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __iter__(self) -> Iterator[tuple[FloatArray, FloatArray]]:
        for start, end in zip(self.offsets[:-1], self.offsets[1:], strict=True):
            yield self.points[start:end], self.strengths[start:end]


def _march(
    field: CoulombField,
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
    field: CoulombField,
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
