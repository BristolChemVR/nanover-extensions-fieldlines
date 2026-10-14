"""Tests for the Coulomb field and field line tracer."""

from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nanover_extensions.fieldlines import core
from nanover_extensions.fieldlines.core import (
    MV_PER_CM,
    VOLT_ANGSTROM,
    CoulombField,
    GridField,
    Seeds,
    bond_field,
    trace,
)

ORIGIN = np.zeros((1, 3))


def dipole(separation: float = 4.0, **kwargs: Any) -> CoulombField:  # noqa: ANN401
    """Return +1 at +z and -1 at -z, `separation` apart."""
    half = separation / 2
    return CoulombField([[0, 0, half], [0, 0, -half]], [1.0, -1.0], **kwargs)


def random_cloud(count: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return random charge positions, charges and probe points."""
    rng = np.random.default_rng(1)
    positions = rng.uniform(-10, 10, size=(count, 3))
    charges = rng.uniform(-1, 1, size=count)
    points = rng.uniform(-10, 10, size=(50, 3))
    return positions, charges, points


# Point charge


@pytest.mark.parametrize("charge", [1.0, -2.0])
@pytest.mark.parametrize("distance", [1.0, 2.5, 10.0])
def test_point_charge_field_is_radial_inverse_square(charge: float, distance: float) -> None:
    field = CoulombField(ORIGIN, [charge])
    direction = np.array([1.0, 2.0, -2.0]) / 3.0
    vectors, nearest = field.evaluate([distance * direction])
    assert vectors[0] == pytest.approx(MV_PER_CM * charge / distance**2 * direction)
    assert nearest[0] == pytest.approx(distance)


def test_unit_charge_at_one_angstrom() -> None:
    vectors, _ = CoulombField(ORIGIN, [1.0]).evaluate([[1.0, 0.0, 0.0]])
    assert vectors[0] == pytest.approx([1439.9645, 0.0, 0.0])


def test_charge_on_the_point_is_ignored() -> None:
    field = CoulombField([[0, 0, 0], [2, 0, 0]], [1.0, 1.0])
    vectors, nearest = field.evaluate(ORIGIN)
    assert vectors[0] == pytest.approx([-MV_PER_CM / 4, 0.0, 0.0])
    assert nearest[0] == 0.0


def test_point_charge_line_is_straight_and_ends_at_the_charge() -> None:
    lines = trace(CoulombField(ORIGIN, [1.0]), [[3.0, 0.0, 0.0]], max_steps=10)
    assert len(lines) == 1
    points, strengths = next(iter(lines))
    # Backwards from x = 3 in steps of 0.25 until within 0.9 of the charge, then 10 steps out.
    assert points[:, 0] == pytest.approx(np.arange(0.75, 5.5 + 0.125, 0.25))
    assert points[:, 1:] == pytest.approx(0.0)
    assert strengths == pytest.approx(MV_PER_CM / points[:, 0] ** 2)
    assert lines.offsets.tolist() == [0, 20]


def test_negative_charge_reverses_the_line() -> None:
    lines = trace(CoulombField(ORIGIN, [-1.0]), [[3.0, 0.0, 0.0]], max_steps=10)
    assert lines.points[:, 0] == pytest.approx(np.arange(5.5, 0.75 - 0.125, -0.25))


# Dipole


def test_dipole_field_on_axis() -> None:
    z = 5.0
    vectors, _ = dipole().evaluate([[0, 0, z]])
    expected = MV_PER_CM * (1 / (z - 2) ** 2 - 1 / (z + 2) ** 2)
    assert vectors[0] == pytest.approx([0.0, 0.0, expected])


def test_dipole_field_on_bisector() -> None:
    x = 3.0
    vectors, nearest = dipole().evaluate([[x, 0, 0]])
    expected = -MV_PER_CM * 4.0 / (x**2 + 4.0) ** 1.5
    assert vectors[0] == pytest.approx([0.0, 0.0, expected])
    assert nearest[0] == pytest.approx(np.sqrt(13.0))


def test_dipole_line_runs_from_positive_to_negative() -> None:
    lines = trace(dipole(), [[1.5, 0.0, 0.0]], step=0.05, max_steps=400)
    points, strengths = next(iter(lines))
    assert np.linalg.norm(points[0] - [0, 0, 2]) <= 0.9
    assert np.linalg.norm(points[-1] - [0, 0, -2]) <= 0.9
    assert np.all(np.diff(points[:, 2]) < 0)
    assert np.all(strengths > 0)
    # The line is mirror symmetric about the plane between the charges.
    assert points[::-1] * [1, 1, -1] == pytest.approx(points, abs=1e-6)


def test_lines_are_returned_per_seed_in_order() -> None:
    seeds = np.array([[1.5, 0.0, 0.0], [0.0, 3.0, 0.0], [-2.5, 0.0, 0.5]])
    lines = trace(dipole(), seeds, max_steps=20)
    assert len(lines) == 3
    assert lines.offsets[0] == 0
    assert lines.offsets[-1] == len(lines.points) == len(lines.strengths)
    for seed, (points, _) in zip(seeds, lines, strict=True):
        assert np.any(np.all(np.isclose(points, seed), axis=1))
        assert np.linalg.norm(np.diff(points, axis=0), axis=1) == pytest.approx(0.25)


def test_bond_field_projects_on_the_bond_direction() -> None:
    field = dipole()
    on_axis = MV_PER_CM * (1 / 3.0**2 - 1 / 7.0**2)
    assert bond_field(field, [0, 0, 4.5], [0, 0, 5.5]) == pytest.approx(on_axis)
    assert bond_field(field, [0, 0, 5.5], [0, 0, 4.5]) == pytest.approx(-on_axis)
    assert bond_field(field, [-0.5, 0, 5], [0.5, 0, 5]) == pytest.approx(0.0)


def test_bond_field_leaves_out_the_bond_atoms() -> None:
    # A bond from z = 0 to z = 1 carrying its own charges, and one outside charge at z = 5.
    field = CoulombField([[0, 0, 0], [0, 0, 1], [0, 0, 5]], [0.5, -0.5, 1.0])
    outside = -MV_PER_CM / 4.5**2
    assert bond_field(field, [0, 0, 0], [0, 0, 1]) == pytest.approx(outside)
    own = MV_PER_CM * (0.5 / 0.5**2 + 0.5 / 0.5**2)
    assert bond_field(field, [0, 0, 0], [0, 0, 1], include_self=True) == pytest.approx(
        outside + own,
    )


def test_bond_field_of_a_bond_alone_is_zero() -> None:
    assert bond_field(dipole(), [0, 0, 2], [0, 0, -2]) == 0.0


def test_without_removes_charges_and_keeps_settings() -> None:
    field = CoulombField([[0, 0, 0], [5, 0, 0]], [1.0, 2.0], cutoff=9.0, dielectric=2.0)
    smaller = field.without([1])
    assert smaller.positions.tolist() == [[0.0, 0.0, 0.0]]
    assert smaller.charges.tolist() == [1.0]
    assert (smaller.cutoff, smaller.dielectric) == (9.0, 2.0)
    assert field.charges.tolist() == [1.0, 2.0]


# Dielectric


def test_constant_dielectric_divides_the_field() -> None:
    positions, charges, points = random_cloud()
    bare, _ = CoulombField(positions, charges).evaluate(points)
    damped, _ = CoulombField(positions, charges, dielectric=4.0).evaluate(points)
    assert damped == pytest.approx(bare / 4.0)


@pytest.mark.parametrize("debye_length", [None, 8.0])
@pytest.mark.parametrize("distance", [1.0, 3.0, 10.0])
def test_distance_dielectric_is_minus_the_potential_gradient(
    debye_length: float | None,
    distance: float,
) -> None:
    def potential(r: float) -> float:
        screening = 1.0 if debye_length is None else np.exp(-r / debye_length)
        return MV_PER_CM * screening / (4.0 * r * r)

    field = CoulombField(ORIGIN, [1.0], dielectric=4.0, distance_dielectric=True,
                         debye_length=debye_length)  # fmt: skip
    vectors, _ = field.evaluate([[distance, 0, 0]])
    h = 1e-5
    gradient = (potential(distance + h) - potential(distance - h)) / (2 * h)
    assert vectors[0, 0] == pytest.approx(-gradient, rel=1e-6)
    assert vectors[0, 1:] == pytest.approx(0.0)


def test_distance_dielectric_falls_off_as_inverse_cube() -> None:
    field = CoulombField(ORIGIN, [1.0], dielectric=4.0, distance_dielectric=True)
    vectors, _ = field.evaluate([[2.0, 0, 0], [4.0, 0, 0]])
    assert vectors[0, 0] == pytest.approx(2 * MV_PER_CM / (4.0 * 2.0**3))
    assert vectors[0, 0] / vectors[1, 0] == pytest.approx(8.0)


# Screening


@pytest.mark.parametrize("distance", [1.0, 5.0, 20.0])
def test_screened_point_charge(distance: float) -> None:
    debye = 8.0
    vectors, _ = CoulombField(ORIGIN, [1.0], debye_length=debye).evaluate([[distance, 0, 0]])
    x = distance / debye
    assert vectors[0, 0] == pytest.approx(MV_PER_CM / distance**2 * np.exp(-x) * (1 + x))


def test_screening_weakens_the_field_more_at_long_range() -> None:
    points = [[2.0, 0, 0], [20.0, 0, 0]]
    bare, _ = CoulombField(ORIGIN, [1.0]).evaluate(points)
    screened, _ = CoulombField(ORIGIN, [1.0], debye_length=8.0).evaluate(points)
    ratio = screened[:, 0] / bare[:, 0]
    assert 0 < ratio[1] < ratio[0] < 1


def test_long_debye_length_recovers_the_bare_field() -> None:
    positions, charges, points = random_cloud()
    bare, _ = CoulombField(positions, charges).evaluate(points)
    screened, _ = CoulombField(positions, charges, debye_length=1e9).evaluate(points)
    assert screened == pytest.approx(bare)


# Cutoff


def test_charges_beyond_the_cutoff_do_not_contribute() -> None:
    field = CoulombField([[0, 0, 0], [12, 0, 0]], [1.0, 1.0], cutoff=10.0)
    vectors, nearest = field.evaluate([[1.0, 0, 0], [30.0, 0, 0]])
    assert vectors[0] == pytest.approx([MV_PER_CM, 0.0, 0.0])
    assert vectors[1] == pytest.approx([0.0, 0.0, 0.0])
    assert nearest == pytest.approx([1.0, 18.0])


@pytest.mark.parametrize("debye_length", [None, 8.0])
def test_large_cutoff_matches_no_cutoff(debye_length: float | None) -> None:
    positions, charges, points = random_cloud()
    dense = CoulombField(positions, charges, debye_length=debye_length)
    sparse = CoulombField(positions, charges, debye_length=debye_length, cutoff=100.0)
    for got, expected in zip(sparse.evaluate(points), dense.evaluate(points), strict=True):
        assert got == pytest.approx(expected)


def test_cutoff_matches_a_masked_sum() -> None:
    positions, charges, points = random_cloud()
    vectors, _ = CoulombField(positions, charges, cutoff=6.0).evaluate(points)
    for point, vector in zip(points, vectors, strict=True):
        near = np.linalg.norm(positions - point, axis=1) <= 6.0
        expected, _ = CoulombField(positions[near], charges[near]).evaluate([point])
        assert vector == pytest.approx(expected[0])


def test_cutoff_ignores_a_charge_on_the_point() -> None:
    field = CoulombField([[0, 0, 0], [2, 0, 0]], [1.0, 1.0], cutoff=5.0)
    vectors, nearest = field.evaluate(ORIGIN)
    assert vectors[0] == pytest.approx([-MV_PER_CM / 4, 0.0, 0.0])
    assert nearest[0] == 0.0


def test_line_ends_where_the_cutoff_leaves_no_field() -> None:
    lines = trace(CoulombField(ORIGIN, [1.0], cutoff=4.0), [[3.0, 0.0, 0.0]])
    assert lines.points[-1, 0] == pytest.approx(4.25)
    assert lines.strengths[-1] == 0.0


# Potential


@pytest.mark.parametrize("cutoff", [None, 50.0])
def test_point_charge_potential(cutoff: float | None) -> None:
    field = CoulombField(ORIGIN, [-2.0], cutoff=cutoff, dielectric=4.0, debye_length=8.0)
    potential = field.potential([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
    assert potential == pytest.approx([-2.0 * VOLT_ANGSTROM * np.exp(-5.0 / 8.0) / 20.0, 0.0])


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"debye_length": 8.0}, {"cutoff": 6.0}, {"dielectric": 4.0, "distance_dielectric": True}],
)
def test_field_is_minus_the_gradient_of_the_potential(kwargs: dict[str, Any]) -> None:
    positions, charges, points = random_cloud()
    field = CoulombField(positions, charges, **kwargs)
    # Keep away from charges and from the cutoff, where the potential jumps.
    distances = np.linalg.norm(points[:, None] - positions[None], axis=2)
    if "cutoff" in kwargs:
        distances = np.minimum(distances, np.abs(distances - kwargs["cutoff"]))
    points = points[distances.min(axis=1) > 0.2]
    h = 1e-4
    gradient = np.stack(
        [
            (field.potential(points + h * axis) - field.potential(points - h * axis)) / (2 * h)
            for axis in np.eye(3)
        ],
        axis=1,
    )
    vectors, _ = field.evaluate(points)
    # 1 V/A is 100 MV/cm.
    assert vectors == pytest.approx(-100.0 * gradient, rel=1e-5, abs=1e-6)


def test_potential_is_the_same_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    positions, charges, points = random_cloud()
    field = CoulombField(positions, charges)
    whole = field.potential(points)
    monkeypatch.setattr(core, "_MAX_DENSE_PAIRS", 7 * len(positions))
    assert field.potential(points) == pytest.approx(whole)


# Grid


def grid_and_exact(**kwargs: Any) -> tuple[GridField, CoulombField, np.ndarray]:  # noqa: ANN401
    """Return the same random charges on a grid and summed directly, and probe points.

    The probe points are more than 3 angstrom from every charge.
    """
    rng = np.random.default_rng(2)
    positions = rng.uniform(-6, 6, size=(40, 3))
    charges = rng.uniform(-1, 1, size=40)
    points = rng.uniform(-12, 12, size=(400, 3))
    nearest = np.linalg.norm(points[:, None] - positions[None], axis=2).min(axis=1)
    points = points[nearest > 3.0]
    grid = GridField(positions, charges, **kwargs)
    exact_kwargs = {key: value for key, value in kwargs.items() if key != "spacing"}
    return grid, CoulombField(positions, charges, **exact_kwargs), points


def relative_errors(grid: GridField, exact: CoulombField, points: np.ndarray) -> np.ndarray:
    got, _ = grid.evaluate(points)
    expected, _ = exact.evaluate(points)
    return np.asarray(np.linalg.norm(got - expected, axis=1) / np.linalg.norm(expected, axis=1))


@pytest.mark.parametrize(
    ("kwargs", "median", "p90"),
    [
        ({}, 0.02, 0.06),
        ({"debye_length": 8.0}, 0.02, 0.06),
        ({"dielectric": 4.0}, 0.02, 0.06),
        # A 1 / r^2 potential is steeper, so central differences are coarser.
        ({"dielectric": 0.5, "distance_dielectric": True}, 0.04, 0.15),
    ],
)
def test_grid_matches_the_direct_sum_away_from_charges(
    kwargs: dict[str, Any],
    median: float,
    p90: float,
) -> None:
    grid, exact, points = grid_and_exact(**kwargs)
    errors = relative_errors(grid, exact, points)
    assert np.median(errors) < median
    assert np.percentile(errors, 90) < p90
    assert grid.evaluate(points)[1] == pytest.approx(exact.evaluate(points)[1])
    expected = exact.potential(points)
    potential_errors = np.abs(grid.potential(points) - expected)
    assert np.median(potential_errors) < 0.01 * np.median(np.abs(expected))


def test_finer_grid_is_more_accurate() -> None:
    coarse = np.median(relative_errors(*grid_and_exact(spacing=1.0)))
    fine = np.median(relative_errors(*grid_and_exact(spacing=0.5)))
    assert fine < coarse / 2


def test_grid_reaches_the_margin_and_is_zero_beyond() -> None:
    grid = GridField(ORIGIN, [1.0], spacing=0.25, margin=5.0)
    vectors, nearest = grid.evaluate([[4.5, 0, 0], [0, 0, -4.5], [8.0, 0, 0], [0, -30.0, 0]])
    assert vectors[0, 0] == pytest.approx(MV_PER_CM / 4.5**2, rel=0.02)
    assert vectors[1, 2] == pytest.approx(-MV_PER_CM / 4.5**2, rel=0.02)
    assert vectors[2:] == pytest.approx(0.0)
    assert grid.potential([[8.0, 0, 0]]) == pytest.approx([0.0])
    assert nearest == pytest.approx([4.5, 4.5, 8.0, 30.0])


def test_grid_reaches_the_covered_points() -> None:
    plain = GridField(ORIGIN, [1.0], margin=2.0)
    covered = GridField(ORIGIN, [1.0], margin=2.0, cover=[[20.0, 0, 0]])
    assert covered.shape[0] == plain.shape[0] + 20
    assert covered.shape[1:] == plain.shape[1:]
    vectors, _ = covered.evaluate([[20.0, 0, 0]])
    assert vectors[0, 0] == pytest.approx(MV_PER_CM / 20.0**2, rel=0.01)


def test_grid_kernel_is_reused_for_the_same_shape() -> None:
    core._kernel_spectrum.cache_clear()
    GridField(ORIGIN, [1.0])
    GridField([[1.0, 2.0, 3.0]], [-1.0])
    assert core._kernel_spectrum.cache_info().hits == 1


def test_line_traced_on_a_grid_follows_the_exact_one() -> None:
    seeds = [[1.5, 0.0, 0.0]]
    exact = trace(dipole(), seeds, step=0.1, max_steps=200)
    grid = GridField([[0, 0, 2], [0, 0, -2]], [1.0, -1.0], spacing=0.5)
    gridded = trace(grid, seeds, step=0.1, max_steps=200)
    # Both run from the positive to the negative charge, close together.
    ends = np.linalg.norm(gridded.points[[0, -1]] - exact.points[[0, -1]], axis=1)
    assert ends.max() < 0.5
    middle = gridded.points[np.abs(gridded.points[:, 2]).argmin()]
    assert middle == pytest.approx(exact.points[np.abs(exact.points[:, 2]).argmin()], abs=0.2)


def test_line_ends_at_the_edge_of_the_grid() -> None:
    lines = trace(GridField(ORIGIN, [1.0], margin=4.0), [[3.0, 0.0, 0.0]], max_steps=40)
    assert 4.0 < lines.points[-1, 0] < 5.5
    assert lines.strengths[-1] == 0.0


# Edge cases


def test_dense_sum_is_the_same_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    positions, charges, points = random_cloud()
    field = CoulombField(positions, charges)
    whole = field.evaluate(points)
    monkeypatch.setattr(core, "_MAX_DENSE_PAIRS", 7 * len(positions))
    for got, expected in zip(field.evaluate(points), whole, strict=True):
        assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    "field",
    [
        CoulombField(ORIGIN, [1.0]),
        CoulombField(ORIGIN, [1.0], cutoff=5.0),
        GridField(ORIGIN, [1.0]),
    ],
)
def test_no_points(field: core.Field) -> None:
    vectors, nearest = field.evaluate(np.zeros((0, 3)))
    assert vectors.shape == (0, 3)
    assert nearest.shape == (0,)
    assert field.potential(np.zeros((0, 3))).shape == (0,)


def test_no_seeds() -> None:
    lines = trace(CoulombField(ORIGIN, [1.0]), np.zeros((0, 3)))
    assert len(lines) == 0
    assert list(lines) == []
    assert lines.points.shape == (0, 3)


def test_seed_at_a_null_point_gives_a_single_point() -> None:
    field = CoulombField([[0, 0, 2], [0, 0, -2]], [1.0, 1.0])
    lines = trace(field, ORIGIN)
    assert lines.points == pytest.approx(ORIGIN)
    assert lines.strengths == pytest.approx([0.0])


def test_zero_steps_returns_the_seeds() -> None:
    seeds = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    lines = trace(CoulombField(ORIGIN, [1.0]), seeds, max_steps=0)
    assert lines.points == pytest.approx(np.array(seeds))
    assert lines.offsets.tolist() == [0, 1, 2]


def test_inputs_are_not_modified() -> None:
    seeds = np.array([[3.0, 0.0, 0.0]])
    trace(CoulombField(ORIGIN, [1.0]), seeds, max_steps=5)
    assert seeds.tolist() == [[3.0, 0.0, 0.0]]


@pytest.mark.parametrize(
    ("positions", "charges", "kwargs"),
    [
        ([0.0, 0.0, 0.0], [1.0], {}),
        ([[0.0, 0.0]], [1.0], {}),
        ([[0.0, 0.0, 0.0]], [1.0, 2.0], {}),
        (np.zeros((0, 3)), [], {}),
        ([[0.0, 0.0, 0.0]], [1.0], {"debye_length": 0.0}),
        ([[0.0, 0.0, 0.0]], [1.0], {"cutoff": -1.0}),
        ([[0.0, 0.0, 0.0]], [1.0], {"dielectric": 0.0}),
    ],
)
def test_invalid_field_arguments(
    positions: Any,  # noqa: ANN401
    charges: Any,  # noqa: ANN401
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="must|required"):
        CoulombField(positions, charges, **kwargs)
    if "cutoff" not in kwargs:
        with pytest.raises(ValueError, match="must|required"):
            GridField(positions, charges, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"spacing": 0.0}, {"margin": -1.0}, {"cover": [1.0, 0.0, 0.0]}],
)
def test_invalid_grid_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must"):
        GridField(ORIGIN, [1.0], **kwargs)


@pytest.mark.parametrize(
    ("seeds", "kwargs"),
    [
        ([1.0, 0.0, 0.0], {}),
        ([[1.0, 0.0, 0.0]], {"step": 0.0}),
        ([[1.0, 0.0, 0.0]], {"max_steps": -1}),
    ],
)
def test_invalid_trace_arguments(seeds: list[float], kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must"):
        trace(CoulombField(ORIGIN, [1.0]), seeds, **kwargs)


def test_invalid_points() -> None:
    with pytest.raises(ValueError, match="points"):
        CoulombField(ORIGIN, [1.0]).evaluate([1.0, 0.0, 0.0])


# Seeds that follow atoms


def rigid_motion(points: np.ndarray) -> np.ndarray:
    """Turn and shift points as a whole."""
    return np.asarray(Rotation.from_rotvec([0.4, -1.1, 0.7]).apply(points) + [2.0, -5.0, 3.0])


def test_seeds_turn_and_move_with_their_atoms() -> None:
    rng = np.random.default_rng(2)
    positions = rng.uniform(-10, 10, size=(40, 3))
    atoms = np.arange(5, 40)
    points = rng.uniform(-12, 12, size=(25, 3))
    seeds = Seeds.anchor(points, np.arange(25), positions, atoms, 25)
    assert len(seeds) == 25
    assert seeds.follow(positions) == pytest.approx(points)
    assert seeds.follow(rigid_motion(positions)) == pytest.approx(rigid_motion(points))
    # Each seed follows its nearest atom first, and only atoms it was allowed.
    nearest = np.linalg.norm(points[:, None] - positions[None, atoms], axis=2).argmin(axis=1)
    assert seeds.atoms[:, 0].tolist() == atoms[nearest].tolist()
    assert set(seeds.atoms.ravel()) <= set(atoms)
    assert len(set(map(tuple, np.sort(seeds.atoms, axis=1)))) > 1


@pytest.mark.parametrize(
    "positions",
    [
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]),
    ],
)
def test_seeds_only_follow_the_first_atom_if_the_others_fix_no_direction(
    positions: np.ndarray,
) -> None:
    points = np.array([[0.5, 4.0, -1.0], [-3.0, 0.2, 2.0]])
    seeds = Seeds.anchor(points, [7, 9], positions, np.arange(len(positions)), 10)
    moved = rigid_motion(positions)
    first = seeds.atoms[:, 0]
    assert seeds.follow(moved) == pytest.approx(moved[first] + points - positions[first])
    assert seeds.ids.tolist() == [7, 9]
    assert seeds.next_id == 10


def test_no_seeds_to_anchor() -> None:
    seeds = Seeds.anchor(np.zeros((0, 3)), [], ORIGIN, np.array([0]), 4)
    assert len(seeds) == 0
    assert seeds.follow(ORIGIN).shape == (0, 3)


def test_seeds_need_atoms_to_follow() -> None:
    with pytest.raises(ValueError, match="at least one atom"):
        Seeds.anchor([[1.0, 0.0, 0.0]], [0], ORIGIN, np.zeros(0, dtype=np.intp), 1)
