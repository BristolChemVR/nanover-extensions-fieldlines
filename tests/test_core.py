"""Tests for the Coulomb field and field line tracer."""

import numpy as np
import pytest

from nanover_extensions.fieldlines import core
from nanover_extensions.fieldlines.core import MV_PER_CM, CoulombField, bond_field, trace

ORIGIN = np.zeros((1, 3))


def dipole(separation: float = 4.0, **kwargs: float) -> CoulombField:
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


# Edge cases


def test_dense_sum_is_the_same_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    positions, charges, points = random_cloud()
    field = CoulombField(positions, charges)
    whole = field.evaluate(points)
    monkeypatch.setattr(core, "_MAX_DENSE_PAIRS", 7 * len(positions))
    for got, expected in zip(field.evaluate(points), whole, strict=True):
        assert got == pytest.approx(expected)


@pytest.mark.parametrize("cutoff", [None, 5.0])
def test_no_points(cutoff: float | None) -> None:
    vectors, nearest = CoulombField(ORIGIN, [1.0], cutoff=cutoff).evaluate(np.zeros((0, 3)))
    assert vectors.shape == (0, 3)
    assert nearest.shape == (0,)


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
    ],
)
def test_invalid_field_arguments(
    positions: object,
    charges: object,
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="must|required"):
        CoulombField(positions, charges, **kwargs)


@pytest.mark.parametrize(
    ("seeds", "kwargs"),
    [
        ([1.0, 0.0, 0.0], {}),
        ([[1.0, 0.0, 0.0]], {"step": 0.0}),
        ([[1.0, 0.0, 0.0]], {"max_steps": -1}),
    ],
)
def test_invalid_trace_arguments(seeds: list[float], kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must"):
        trace(CoulombField(ORIGIN, [1.0]), seeds, **kwargs)


def test_invalid_points() -> None:
    with pytest.raises(ValueError, match="points"):
        CoulombField(ORIGIN, [1.0]).evaluate([1.0, 0.0, 0.0])
