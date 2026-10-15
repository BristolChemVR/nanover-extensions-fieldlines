"""Tests for publishing field lines from an OpenMM simulation."""

# NanoVer is not fully annotated.
# mypy: disable-error-code="no-untyped-call"

import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future
from typing import Any, ParamSpec, TypeVar

import numpy as np
import openmm as mm
import pytest
from nanover.app import OmniRunner
from nanover.openmm import OpenMMSimulation
from nanover.testing.servers import ServerClientSetup, make_connected_server_client_setup
from openmm import app
from openmm.unit import femtosecond, kelvin, nanometer, picosecond
from scipy.spatial.transform import Rotation

from nanover_extensions.fieldlines import service as service_module
from nanover_extensions.fieldlines.core import CoulombField, FieldLines, GridField, Seeds
from nanover_extensions.fieldlines.service import (
    COLORS_KEY,
    CONFIGURE_COMMAND,
    IDS_KEY,
    LENGTHS_KEY,
    POSITIONS_KEY,
    RENDERER,
    SELECTION_ID,
    FieldLineConfig,
    FieldLineService,
    field_lines_selection,
    fibonacci_sphere,
    get_charges,
    grid_around,
    make_field,
    make_seeds,
    pack_lines,
    potential_colors,
    seeding_mode,
    solvent_mask,
    source_indices,
    spread_strongest,
    strength_colors,
    trace_lines,
)

CHARGES = [1.0, -1.0, 0.0, 0.5]
POSITIONS_NM = [[0.0, 0.0, 0.3], [0.0, 0.0, -0.3], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]
POSITIONS = 10.0 * np.array(POSITIONS_NM)
RESIDUES = ["AR"] * 4


def build_simulation(*, charged: bool = True, residues: list[str] = RESIDUES) -> app.Simulation:
    """Return a tiny non-periodic OpenMM simulation of four argon-like particles."""
    system = mm.System()
    topology = app.Topology()
    chain = topology.addChain()
    force = mm.NonbondedForce()
    force.setNonbondedMethod(force.NoCutoff)
    for charge, name in zip(CHARGES, residues, strict=True):
        system.addParticle(mass=40.0)
        residue = topology.addResidue(name=name, chain=chain)
        topology.addAtom(element=app.Element.getBySymbol("Ar"), name="Ar", residue=residue)
        force.addParticle(charge=charge, sigma=0.3, epsilon=0.0)
    system.addForce(mm.CMMotionRemover())
    if charged:
        system.addForce(force)
    integrator = mm.LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.01 * femtosecond)
    platform = mm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(topology, system, integrator, platform=platform)
    simulation.context.setPositions(np.array(POSITIONS_NM) * nanometer)
    return simulation


@pytest.fixture
def setup() -> Iterator[ServerClientSetup]:
    with make_connected_server_client_setup() as server_client:
        yield server_client


@pytest.fixture
def simulation(setup: ServerClientSetup) -> OpenMMSimulation:
    sim: OpenMMSimulation = OpenMMSimulation.from_simulation(build_simulation())
    sim.frame_interval = 1
    sim.reset(setup.app)
    return sim


P = ParamSpec("P")
T = TypeVar("T")


class ImmediateExecutor(Executor):
    """Runs each job on the calling thread as soon as it is submitted."""

    def submit(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        future: Future[T] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future


class ManualExecutor(Executor):
    """Holds jobs until the test runs them, like a helper thread that is still busy."""

    def __init__(self) -> None:
        self.jobs: list[tuple[Future[Any], Callable[[], Any]]] = []

    def submit(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        future: Future[T] = Future()
        self.jobs.append((future, lambda: fn(*args, **kwargs)))
        return future

    def run(self) -> None:
        """Finish every job submitted so far."""
        jobs, self.jobs = self.jobs, []
        for future, job in jobs:
            future.set_result(job())


def attach(simulation: OpenMMSimulation, config: FieldLineConfig | None = None) -> FieldLineService:
    """Attach a service whose background traces finish as soon as they start."""
    return FieldLineService(simulation, config, executor=ImmediateExecutor())


def seeds_for(config: FieldLineConfig, previous: np.ndarray | None = None) -> np.ndarray:
    """Return the seeds that `config` places around the test atoms."""
    charges = np.array(CHARGES)
    sources = source_indices(charges, config)
    field = CoulombField(POSITIONS[sources], charges[sources], cutoff=config.cutoff)
    seeds, _ = make_seeds(POSITIONS, charges, sources, field, config, previous=previous)
    return seeds


def seeds_of(lines: FieldLines) -> Seeds:
    """Return the seeds that traced lines record."""
    assert lines.seeds is not None
    return lines.seeds


def unpack(frame_dict: dict[str, bytes]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode the four field line entries of a frame."""
    return (
        np.frombuffer(frame_dict[POSITIONS_KEY], dtype=np.float32).reshape(-1, 3),
        np.frombuffer(frame_dict[LENGTHS_KEY], dtype=np.uint32),
        np.frombuffer(frame_dict[COLORS_KEY], dtype=np.uint8).reshape(-1, 4),
        np.frombuffer(frame_dict[IDS_KEY], dtype=np.uint32),
    )


# Settings


def test_default_config_is_valid() -> None:
    config = FieldLineConfig()
    assert config.enabled
    assert config.updated() == config


def test_updated_changes_settings_and_converts_indices() -> None:
    config = FieldLineConfig().updated(cutoff=None, seed_atoms=[np.int64(3), 1], stride=4)
    assert config.cutoff is None
    assert config.seed_atoms == (3, 1)
    assert config.stride == 4
    assert config.updated(seed_atoms=None).seed_atoms is None


@pytest.mark.parametrize(
    "changes",
    [
        {"stride": 0},
        {"seed_radius": -1.0},
        {"cutoff": 0.0},
        {"debye_length": -2.0},
        {"auto_seed_atoms": -1},
        {"grid_spacing": 0.0},
        {"seed_spacing": -1.0},
        {"dielectric": 0.0},
        {"seeding": "random"},
        {"seeding": "region"},
        {"min_points": 2},
        {"weak_strength": 0.0},
        {"weak_strength": 10.0, "strong_strength": 5.0},
        {"colour": "red"},
        {"field_method": "fft"},
        {"coloring": "charge"},
        {"cutoff": 12.0},
        {"field_spacing": 0.0},
        {"field_margin": -1.0},
        {"min_strength": -1.0},
        {"potential_limit": 0.0},
    ],
)
def test_invalid_settings_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must|Unknown"):
        FieldLineConfig().updated(**changes)


# Building blocks


def test_get_charges() -> None:
    assert get_charges(build_simulation().system).tolist() == CHARGES


def test_get_charges_needs_a_nonbonded_force() -> None:
    with pytest.raises(ValueError, match="NonbondedForce"):
        get_charges(build_simulation(charged=False).system)


@pytest.mark.parametrize("count", [1, 8, 50])
def test_fibonacci_sphere_points_are_on_the_unit_sphere(count: int) -> None:
    points = fibonacci_sphere(count)
    assert points.shape == (count, 3)
    assert np.linalg.norm(points, axis=1) == pytest.approx(1.0)


def test_fibonacci_sphere_is_balanced() -> None:
    assert np.abs(fibonacci_sphere(200).mean(axis=0)).max() < 0.01


def test_strength_colors_span_the_scale_logarithmically() -> None:
    colors = strength_colors(np.array([0.0, 0.1, 1.0, 10.0, 100.0, 1e6]), 1.0, 100.0)
    assert colors.dtype == np.uint8
    assert colors[:, 3].tolist() == [255] * 6
    assert colors[0].tolist() == colors[1].tolist() == colors[2].tolist() == [38, 89, 255, 255]
    assert colors[4].tolist() == colors[5].tolist() == [255, 230, 51, 255]
    assert colors[3, :3] == pytest.approx((colors[2, :3] / 2 + colors[4, :3] / 2), abs=1)


def test_potential_colors_run_from_red_through_white_to_blue() -> None:
    colors = potential_colors(np.array([-9.0, -1.0, -0.5, 0.0, 0.5, 1.0, 9.0]), 1.0)
    assert colors.dtype == np.uint8
    assert colors[:, 3].tolist() == [255] * 7
    assert colors[0].tolist() == colors[1].tolist() == [214, 31, 31, 255]
    assert colors[3].tolist() == [255, 255, 255, 255]
    assert colors[5].tolist() == colors[6].tolist() == [31, 64, 214, 255]
    assert colors[2, :3] == pytest.approx((colors[1, :3] / 2 + 255 / 2), abs=1)
    assert colors[4, :3] == pytest.approx((colors[5, :3] / 2 + 255 / 2), abs=1)


def test_solvent_mask_marks_water_and_monovalent_ions() -> None:
    residues = ["ALA", "HOH", "NA", "ZN"]
    topology = build_simulation(residues=residues).topology
    assert solvent_mask(topology).tolist() == [False, True, True, False]


def test_sources_leave_out_solvent_and_neutral_atoms() -> None:
    charges = np.array(CHARGES)
    solvent = np.array([False, True, False, False])
    config = FieldLineConfig()
    assert source_indices(charges, config).tolist() == [0, 1, 3]
    assert source_indices(charges, config, solvent).tolist() == [0, 3]
    assert source_indices(charges, config.updated(include_solvent=True), solvent).tolist() == [
        0,
        1,
        3,
    ]
    # An explicit choice wins over the solvent rule.
    assert source_indices(charges, config.updated(source_atoms=[1, 2]), solvent).tolist() == [1]


@pytest.mark.parametrize(
    ("changes", "mode"),
    [
        ({}, "strength"),
        ({"seed_atoms": [0]}, "atoms"),
        ({"region_atoms": [0], "seed_atoms": [0]}, "region"),
        ({"seeding": "atoms"}, "atoms"),
        ({"seeding": "strength", "region_atoms": [0]}, "strength"),
    ],
)
def test_auto_seeding_follows_the_chosen_atoms(changes: dict[str, object], mode: str) -> None:
    assert seeding_mode(FieldLineConfig().updated(**changes)) == mode


def test_grid_covers_the_shell_around_the_centres() -> None:
    centres = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    grid = grid_around(centres, shell=3.0, spacing=1.0)
    distance = np.linalg.norm(grid[:, None] - centres[None], axis=2).min(axis=1)
    assert distance.max() <= 3.0
    assert any(np.allclose(point, [3.0, 0.0, 0.0]) for point in grid)
    assert any(np.allclose(point, [0.0, -3.0, 0.0]) for point in grid)
    assert not any(np.allclose(point, [5.0, 0.0, 0.0]) for point in grid)
    assert len(grid) == 2 * len(grid_around(centres[:1], shell=3.0, spacing=1.0))


def test_grid_points_stay_put_when_the_centres_move() -> None:
    centres = np.array([[0.3, -1.1, 2.0], [7.9, 4.2, -3.3]])
    grid = grid_around(centres, shell=4.0, spacing=1.5)
    assert grid / 1.5 == pytest.approx(np.round(grid / 1.5), abs=1e-12)
    moved = grid_around(centres + [0.4, -0.2, 0.1], shell=4.0, spacing=1.5)
    shared = {tuple(point) for point in grid} & {tuple(point) for point in moved}
    assert len(shared) > 0.7 * len(grid)


def test_spread_puts_the_first_points_before_the_rest() -> None:
    points = np.array([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0], [9.0, 0, 0], [20.0, 0, 0]])
    strengths = np.array([3.0, 5.0, 1.0, 2.0, 0.0])
    first = np.array([True, False, True, False, True])
    # Point 0 now blocks the stronger point 1, and point 4 has no field.
    assert spread_strongest(points, strengths, 2.0, 10, first=first).tolist() == [0, 2, 3]


def test_allowed_previous_seeds_are_kept() -> None:
    config = FieldLineConfig(max_lines=5)
    fresh = seeds_for(config)
    # Off the grid, and weaker than every fresh seed.
    weak = np.array([[0.3, -7.0, 0.2]])
    kept = seeds_for(config, previous=weak)
    assert kept[0] == pytest.approx(weak[0])
    assert len(kept) == 5
    # A previous seed outside the shell, or too close to an atom, is not kept.
    for previous in (np.array([[100.0, 100.0, 100.0]]), POSITIONS[:1] + 0.5, np.zeros((0, 3))):
        assert seeds_for(config, previous=previous) == pytest.approx(fresh)


def test_kept_seeds_may_sit_closer_than_new_ones() -> None:
    config = FieldLineConfig(max_lines=50, seed_spacing=4.0, grid_spacing=3.0)
    # 3 angstrom apart: under `seed_spacing` (4) but over half of it.
    close = np.array([[0.0, -6.0, 0.0], [3.0, -6.0, 0.0]])
    seeds = seeds_for(config, previous=close)
    assert seeds[:2] == pytest.approx(close)
    gaps = np.linalg.norm(seeds[2:, None] - seeds[None], axis=2)
    assert gaps[gaps > 0].min() >= config.seed_spacing
    # Closer than half the spacing, only the stronger one stays.
    closer = np.array([[0.0, -6.0, 0.0], [1.5, -6.0, 0.0]])
    seeds = seeds_for(config, previous=closer)
    assert sum(np.allclose(seed, point) for seed in seeds for point in closer) == 1


def test_traced_lines_record_their_seeds_and_ids() -> None:
    charges = np.array(CHARGES)
    config = FieldLineConfig(max_lines=10)
    lines = trace_lines(POSITIONS, charges, config)
    seeds = seeds_of(lines)
    field = make_field(POSITIONS, charges, np.array([0, 1, 3]), config)
    expected, _ = make_seeds(POSITIONS, charges, np.array([0, 1, 3]), field, config)
    assert seeds.points == pytest.approx(expected)
    assert (seeds.ids.tolist(), seeds.next_id) == (list(range(len(seeds))), len(seeds))
    assert lines.ids is not None
    assert len(lines.ids) == len(lines)

    again = trace_lines(POSITIONS, charges, config, previous_seeds=seeds)
    assert seeds_of(again).points == pytest.approx(seeds.points)
    assert seeds_of(again).ids.tolist() == seeds.ids.tolist()
    assert again.points == pytest.approx(lines.points)


def test_new_seeds_get_new_ids() -> None:
    charges = np.array(CHARGES)
    few = trace_lines(POSITIONS, charges, FieldLineConfig(max_lines=3))
    more = trace_lines(POSITIONS, charges, FieldLineConfig(max_lines=8), previous_seeds=few.seeds)
    assert seeds_of(more).ids.tolist() == list(range(8))
    assert seeds_of(more).next_id == 8
    # Seeds that are no longer allowed (here: outside a small region) leave their ids unused.
    region = FieldLineConfig(region_atoms=(3,), seed_shell=3.0, max_lines=3)
    elsewhere = trace_lines(POSITIONS, charges, region, previous_seeds=more.seeds)
    assert seeds_of(elsewhere).ids.tolist() == [8, 9, 10]


@pytest.mark.parametrize("seeding", ["strength", "region"])
def test_kept_lines_move_with_the_molecule(seeding: str) -> None:
    charges = np.array(CHARGES)
    config = FieldLineConfig(seeding=seeding, region_atoms=(0, 1, 3), field_method="direct")
    lines = trace_lines(POSITIONS, charges, config)
    turn = Rotation.from_rotvec([0.0, 0.0, 0.7])
    shift = np.array([3.3, -1.2, 5.6])
    moved = trace_lines(turn.apply(POSITIONS) + shift, charges, config, previous_seeds=lines.seeds)
    assert lines.ids is not None
    assert moved.ids is not None
    by_id = {int(i): points for i, (points, _) in zip(moved.ids, moved, strict=True)}
    for line_id, (points, _) in zip(lines.ids, lines, strict=True):
        assert by_id[int(line_id)] == pytest.approx(turn.apply(points) + shift, abs=1e-6)


def test_atom_seeded_lines_are_numbered_in_order() -> None:
    charges = np.array(CHARGES)
    config = FieldLineConfig(seed_atoms=(0, 1), seeds_per_atom=3)
    lines = trace_lines(POSITIONS, charges, config)
    again = trace_lines(POSITIONS + 1.0, charges, config, previous_seeds=lines.seeds)
    for seeds in (seeds_of(lines), seeds_of(again)):
        assert (seeds.ids.tolist(), seeds.next_id) == (list(range(6)), 6)
    assert seeds_of(again).points == pytest.approx(seeds_of(lines).points + 1.0)


def test_spread_picks_the_strongest_points_apart() -> None:
    points = np.array([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0], [9.0, 0, 0], [20.0, 0, 0]])
    strengths = np.array([3.0, 5.0, 1.0, 2.0, 0.0])
    assert spread_strongest(points, strengths, spacing=2.0, count=10).tolist() == [1, 3, 2]
    assert spread_strongest(points, strengths, spacing=2.0, count=2).tolist() == [1, 3]
    assert len(spread_strongest(points, strengths, spacing=0.0, count=10)) == 4
    assert len(spread_strongest(points, np.zeros(5), spacing=2.0, count=10)) == 0


def test_strength_seeds_sit_in_the_space_around_the_sources() -> None:
    config = FieldLineConfig(max_lines=30)
    seeds = seeds_for(config)
    assert 0 < len(seeds) <= 30
    sources = POSITIONS[[0, 1, 3]]
    distance = np.linalg.norm(seeds[:, None] - sources[None], axis=2).min(axis=1)
    assert distance.min() >= config.seed_clearance
    assert distance.max() <= config.seed_shell
    gaps = np.linalg.norm(seeds[:, None] - seeds[None], axis=2)[np.triu_indices(len(seeds), 1)]
    assert gaps.min() >= config.seed_spacing


def test_strength_seeds_start_where_the_field_is_strongest() -> None:
    seeds = seeds_for(FieldLineConfig(max_lines=1, cutoff=None))
    charges = np.array(CHARGES)
    field = CoulombField(POSITIONS[[0, 1, 3]], charges[[0, 1, 3]])
    candidates = grid_around(POSITIONS[[0, 1, 3]], 8.0, 3.0)
    vectors, nearest = field.evaluate(candidates)
    strengths = np.where(nearest >= 2.5, np.linalg.norm(vectors, axis=1), 0.0)
    assert seeds[0] == pytest.approx(candidates[strengths.argmax()])


def test_region_seeds_stay_near_the_region() -> None:
    config = FieldLineConfig(region_atoms=(3,), seed_shell=5.0)
    seeds = seeds_for(config)
    assert len(seeds) > 0
    assert np.linalg.norm(seeds - POSITIONS[3], axis=1).max() <= 5.0
    assert len(seeds_for(config.updated(region_atoms=[]))) == 0


def test_seeds_surround_the_chosen_atoms() -> None:
    config = FieldLineConfig(seed_atoms=(2, 0), seeds_per_atom=5, seed_radius=2.0)
    seeds = seeds_for(config)
    assert seeds.shape == (10, 3)
    assert np.linalg.norm(seeds[:5] - POSITIONS[2], axis=1) == pytest.approx(2.0)
    assert np.linalg.norm(seeds[5:] - POSITIONS[0], axis=1) == pytest.approx(2.0)


def test_seeds_default_to_the_most_charged_source_atoms() -> None:
    config = FieldLineConfig(seeding="atoms", auto_seed_atoms=2, seeds_per_atom=1)
    seeds = seeds_for(config)
    assert np.linalg.norm(seeds - POSITIONS[[0, 1]], axis=1) == pytest.approx(1.5)

    seeds = seeds_for(config.updated(source_atoms=(1, 2, 3)))
    assert np.linalg.norm(seeds - POSITIONS[[1, 3]], axis=1) == pytest.approx(1.5)


def test_seeds_are_thinned_to_the_line_limit() -> None:
    config = FieldLineConfig(seed_atoms=(0, 1, 2, 3), seeds_per_atom=10, max_lines=8)
    seeds = seeds_for(config)
    assert len(seeds) == 8
    nearest_atom = np.linalg.norm(seeds[:, None] - POSITIONS[None], axis=2).argmin(axis=1)
    assert set(nearest_atom.tolist()) == {0, 1, 2, 3}


def test_solvent_charges_do_not_shape_the_lines() -> None:
    charges = np.array(CHARGES)
    solvent = np.array([False, True, False, False])
    config = FieldLineConfig(seed_atoms=(0,), cutoff=None)
    without_solvent = trace_lines(POSITIONS, charges, config, solvent)
    expected = trace_lines(POSITIONS, charges, config.updated(source_atoms=[0, 3]))
    assert without_solvent.points == pytest.approx(expected.points)


@pytest.mark.parametrize("method", ["grid", "direct"])
def test_a_constant_dielectric_scales_strengths_but_not_shapes(method: str) -> None:
    charges = np.array(CHARGES)
    # No lower strength limit, which would end the weaker lines sooner.
    config = FieldLineConfig(field_method=method, max_lines=20, min_strength=0.0)
    vacuum = trace_lines(POSITIONS, charges, config)
    damped = trace_lines(POSITIONS, charges, config.updated(dielectric=4.0))
    assert len(vacuum) > 0
    assert vacuum.potentials is not None
    assert damped.points == pytest.approx(vacuum.points)
    assert damped.strengths == pytest.approx(vacuum.strengths / 4.0)
    assert damped.potentials == pytest.approx(vacuum.potentials / 4.0)


def test_lines_end_where_the_field_is_weak() -> None:
    charges = np.array(CHARGES)
    config = FieldLineConfig(max_lines=20, min_strength=0.0)
    full = trace_lines(POSITIONS, charges, config)
    weak = np.percentile(full.strengths, 30)
    cut = trace_lines(POSITIONS, charges, config.updated(min_strength=weak))
    assert len(cut.points) < len(full.points)
    # Lines are traced both ways from their seed, so only the two ends may be at or below
    # the limit.
    inner = np.ones(len(cut.points), dtype=bool)
    inner[cut.offsets[:-1]] = False
    inner[cut.offsets[1:] - 1] = False
    assert cut.strengths[inner].min() > weak


@pytest.mark.parametrize(
    ("changes", "kind"),
    [({}, GridField), ({"field_method": "direct"}, CoulombField)],
)
def test_the_field_method_picks_the_field(changes: dict[str, object], kind: type) -> None:
    sources = np.array([0, 1, 3])
    field = make_field(POSITIONS, np.array(CHARGES), sources, FieldLineConfig().updated(**changes))
    assert isinstance(field, kind)
    assert field.positions == pytest.approx(POSITIONS[sources])


def test_the_direct_field_takes_the_settings() -> None:
    config = FieldLineConfig(field_method="direct", cutoff=9.0, debye_length=None, dielectric=2.0)
    field = make_field(POSITIONS, np.array(CHARGES), np.array([0, 1]), config)
    assert isinstance(field, CoulombField)
    assert (field.cutoff, field.debye_length, field.dielectric) == (9.0, None, 2.0)


@pytest.mark.parametrize("name", ["region_atoms", "seed_atoms"])
def test_the_grid_reaches_the_seeding_atoms(name: str) -> None:
    # Atom 2 is 15 angstrom from the sources, beyond a 2 angstrom margin.
    config = FieldLineConfig(field_margin=2.0).updated(**{name: [2]})
    field = make_field(POSITIONS, np.array(CHARGES), np.array([0, 1]), config)
    near_atom_2 = POSITIONS[2] - [1.0, 0.0, 0.0]
    vectors, _ = field.evaluate([near_atom_2])
    assert np.linalg.norm(vectors[0]) > 0
    plain = make_field(
        POSITIONS,
        np.array(CHARGES),
        np.array([0, 1]),
        FieldLineConfig(field_margin=2.0),
    )
    assert plain.evaluate([near_atom_2])[0] == pytest.approx(0.0)


@pytest.mark.parametrize(("coloring", "has_potentials"), [("potential", True), ("strength", False)])
def test_lines_carry_potentials_when_coloured_by_them(coloring: str, has_potentials: bool) -> None:  # noqa: FBT001
    charges = np.array(CHARGES)
    config = FieldLineConfig(max_lines=20, coloring=coloring)
    lines = trace_lines(POSITIONS, charges, config)
    assert (lines.potentials is not None) == has_potentials
    if lines.potentials is not None:
        field = make_field(POSITIONS, charges, np.array([0, 1, 3]), config)
        assert lines.potentials == pytest.approx(field.potential(lines.points))
        # Lines run from positive towards negative potential.
        starts, ends = lines.offsets[:-1], lines.offsets[1:] - 1
        assert np.all(lines.potentials[starts] > lines.potentials[ends])


@pytest.mark.parametrize("name", ["seed_atoms", "source_atoms", "region_atoms"])
@pytest.mark.parametrize("index", [-1, 4])
def test_atom_indices_outside_the_system_are_rejected(name: str, index: int) -> None:
    config = FieldLineConfig().updated(**{name: [0, index]})
    with pytest.raises(ValueError, match=name):
        trace_lines(POSITIONS, np.array(CHARGES), config)


def test_lines_run_between_the_opposite_charges() -> None:
    config = FieldLineConfig(seed_atoms=(0,), cutoff=None, source_atoms=(0, 1))
    lines = trace_lines(POSITIONS, np.array(CHARGES), config)
    assert len(lines) > 0
    for points, strengths in lines:
        assert len(points) >= config.min_points
        assert np.linalg.norm(points[0] - POSITIONS[0]) <= 0.9
        assert np.all(strengths > 0)
    ends = lines.points[lines.offsets[1:] - 1]
    assert np.any(np.linalg.norm(ends - POSITIONS[1], axis=1) <= 0.9)


def test_short_lines_are_dropped() -> None:
    charges = np.array(CHARGES)
    short = FieldLineConfig(seed_atoms=(0,), cutoff=None, max_steps=4, min_points=3)
    assert len(trace_lines(POSITIONS, charges, short)) == 8
    assert len(trace_lines(POSITIONS, charges, short.updated(min_points=50))) == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"source_atoms": [2]},
        {"source_atoms": []},
        {"seed_atoms": []},
        {"seeding": "atoms", "auto_seed_atoms": 0},
        {"region_atoms": [], "seeding": "region"},
    ],
)
def test_nothing_to_trace_gives_no_lines(changes: dict[str, object]) -> None:
    lines = trace_lines(POSITIONS, np.array(CHARGES), FieldLineConfig().updated(**changes))
    assert len(lines) == 0
    assert lines.points.shape == (0, 3)


def test_packed_lines_are_in_nanometers_with_lengths_and_colors() -> None:
    lines = FieldLines(
        points=np.arange(15, dtype=np.float64).reshape(5, 3),
        strengths=np.array([1.0, 1.0, 1000.0, 1000.0, 1000.0]),
        offsets=np.array([0, 2, 5]),
    )
    positions, lengths, colors, ids = unpack(
        pack_lines(lines, FieldLineConfig(coloring="strength"))
    )
    assert positions == pytest.approx(lines.points / 10.0)
    assert lengths.tolist() == [2, 3]
    assert colors.tolist() == [[38, 89, 255, 255]] * 2 + [[255, 230, 51, 255]] * 3
    assert ids.tolist() == [0, 1]


def test_packed_lines_keep_their_ids() -> None:
    lines = FieldLines(
        points=np.zeros((6, 3)),
        strengths=np.ones(6),
        offsets=np.array([0, 3, 6]),
        ids=np.array([7, 12]),
    )
    *_, ids = unpack(pack_lines(lines, FieldLineConfig(coloring="strength")))
    assert ids.tolist() == [7, 12]


def test_packed_lines_are_coloured_by_potential() -> None:
    lines = FieldLines(
        points=np.zeros((3, 3)),
        strengths=np.ones(3),
        offsets=np.array([0, 3]),
        potentials=np.array([2.0, 0.0, -2.0]),
    )
    _, _, colors, _ = unpack(pack_lines(lines, FieldLineConfig(potential_limit=2.0)))
    assert colors.tolist() == [[31, 64, 214, 255], [255, 255, 255, 255], [214, 31, 31, 255]]
    with pytest.raises(ValueError, match="potentials"):
        pack_lines(FieldLines(lines.points, lines.strengths, lines.offsets), FieldLineConfig())


# Service


@pytest.mark.parametrize("background", [True, False])
def test_frames_carry_lines_that_match_the_positions(
    setup: ServerClientSetup,
    simulation: OpenMMSimulation,
    background: bool,  # noqa: FBT001
) -> None:
    attach(simulation, FieldLineConfig(cutoff=None, background=background))
    simulation.advance_by_one_step()

    frame = setup.server_current_frame
    positions, lengths, colors, ids = unpack(frame.frame_dict)
    assert len(lengths) > 0
    assert lengths.sum() == len(positions) == len(colors)
    assert len(ids) == len(lengths)
    assert lengths.min() >= FieldLineConfig().min_points

    expected = trace_lines(
        10.0 * np.asarray(frame.particle_positions, dtype=np.float64),
        np.array(CHARGES),
        FieldLineConfig(cutoff=None),
    )
    assert positions == pytest.approx(expected.points / 10.0, abs=1e-6)


def test_the_service_leaves_out_solvent_from_the_topology(setup: ServerClientSetup) -> None:
    residues = ["AR", "HOH", "AR", "AR"]
    simulation = OpenMMSimulation.from_simulation(build_simulation(residues=residues))
    simulation.reset(setup.app)
    config = FieldLineConfig(cutoff=None)
    attach(simulation, config)
    simulation.advance_by_one_step()

    frame = setup.server_current_frame
    expected = trace_lines(
        10.0 * np.asarray(frame.particle_positions, dtype=np.float64),
        np.array(CHARGES),
        config.updated(source_atoms=[0, 3]),
    )
    assert unpack(frame.frame_dict)[0] == pytest.approx(expected.points / 10.0, abs=1e-6)


def test_lines_reach_the_client_as_bytes(
    setup: ServerClientSetup,
    simulation: OpenMMSimulation,
) -> None:
    attach(simulation)
    simulation.advance_by_one_step()
    setup.assert_frames_synced_soon()
    received = setup.client_current_frame.frame_dict
    assert isinstance(received[POSITIONS_KEY], bytes)
    positions, lengths, _, ids = unpack(received)
    assert lengths.sum() == len(positions)
    assert len(ids) == len(lengths)


def test_lines_are_traced_every_stride_frames(
    simulation: OpenMMSimulation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traced = []
    original = service_module.trace_lines

    def counting(*args: Any) -> FieldLines:  # noqa: ANN401
        assert simulation.simulation is not None
        traced.append(simulation.simulation.currentStep)
        return original(*args)

    monkeypatch.setattr(service_module, "trace_lines", counting)
    attach(simulation, FieldLineConfig(stride=3))
    for _ in range(7):
        simulation.advance_by_one_step()
    assert traced == [1, 4, 7]


@pytest.mark.parametrize("background", [True, False])
def test_each_trace_starts_from_the_last_seeds(
    simulation: OpenMMSimulation,
    monkeypatch: pytest.MonkeyPatch,
    background: bool,  # noqa: FBT001
) -> None:
    given: list[Seeds | None] = []
    returned: list[Seeds | None] = []
    original = service_module.trace_lines

    def recording(*args: Any) -> FieldLines:  # noqa: ANN401
        given.append(args[4])
        lines = original(*args)
        returned.append(lines.seeds)
        return lines

    monkeypatch.setattr(service_module, "trace_lines", recording)
    service = attach(simulation, FieldLineConfig(background=background))
    simulation.advance_by_one_step()
    simulation.advance_by_one_step()
    assert given[0] is None
    assert given[1] is returned[0]
    assert returned[0] is not None
    assert len(returned[0]) > 0

    # New settings start afresh, and `keep_seeds` can turn the reuse off.
    service.configure(max_lines=50)
    simulation.advance_by_one_step()
    assert given[2] is None
    service.configure(keep_seeds=False)
    simulation.advance_by_one_step()
    simulation.advance_by_one_step()
    assert given[3] is None
    assert given[4] is None


def test_disabling_clears_the_lines_once(
    setup: ServerClientSetup,
    simulation: OpenMMSimulation,
) -> None:
    service = attach(simulation)
    simulation.advance_by_one_step()
    assert len(setup.server_current_frame[LENGTHS_KEY]) > 0

    service.configure(enabled=False)
    assert POSITIONS_KEY in simulation.make_regular_frame()
    assert POSITIONS_KEY not in simulation.make_regular_frame()
    simulation.advance_by_one_step()
    assert POSITIONS_KEY not in simulation.make_regular_frame()

    service.configure(enabled=True)
    simulation.advance_by_one_step()
    assert len(setup.server_current_frame[LENGTHS_KEY]) > 0


def test_disabled_from_the_start_publishes_nothing(
    setup: ServerClientSetup,
    simulation: OpenMMSimulation,
) -> None:
    attach(simulation, FieldLineConfig(enabled=False))
    simulation.advance_by_one_step()
    assert POSITIONS_KEY not in setup.server_current_frame


def test_clients_can_configure_with_a_command(
    setup: ServerClientSetup,
    simulation: OpenMMSimulation,
) -> None:
    service = attach(simulation)
    service.register_commands(setup.app)

    result = setup.client.run_command_blocking(CONFIGURE_COMMAND, seed_atoms=[3], stride=2)
    assert result["stride"] == 2
    assert service.config.seed_atoms == (3,)
    assert service.config.cutoff == FieldLineConfig().cutoff

    simulation.advance_by_one_step()
    positions, lengths, _, _ = unpack(setup.server_current_frame.frame_dict)
    starts = 10.0 * positions[np.cumsum(lengths) - lengths]
    ends = 10.0 * positions[np.cumsum(lengths) - 1]
    atom = 10.0 * np.asarray(setup.server_current_frame.particle_positions[3])
    near_atom = np.minimum(
        np.linalg.norm(starts - atom, axis=1),
        np.linalg.norm(ends - atom, axis=1),
    )
    assert np.any(near_atom <= 0.9)


def test_the_selection_asks_clients_to_draw_the_lines(simulation: OpenMMSimulation) -> None:
    runner = OmniRunner.with_basic_server(simulation, name="selection-test", port=0)
    try:
        runner.set_simulation_selections(simulation, field_lines_selection())
        runner.load(0)
        selection = runner.app_server.copy_state()[SELECTION_ID]
    finally:
        runner.close()
    assert selection["selected"] == {"particle_ids": []}
    assert selection["properties"]["nanover.rendering.renderer"] == RENDERER
    assert selection["properties"]["nanover.interaction.method"] == "none"


def test_configure_restarts_the_stride(simulation: OpenMMSimulation) -> None:
    service = attach(simulation, FieldLineConfig(stride=100))
    assert POSITIONS_KEY in simulation.make_regular_frame()
    assert POSITIONS_KEY not in simulation.make_regular_frame()
    service.configure(max_lines=5)
    frame = simulation.make_regular_frame()
    assert len(unpack(frame.frame_dict)[1]) <= 5


def test_a_reset_restarts_the_stride(
    setup: ServerClientSetup,
    simulation: OpenMMSimulation,
) -> None:
    attach(simulation, FieldLineConfig(stride=100))
    simulation.advance_by_one_step()
    simulation.reset(setup.app)
    assert POSITIONS_KEY not in setup.server_current_frame
    simulation.advance_by_one_step()
    assert POSITIONS_KEY in setup.server_current_frame


@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("charged", [True, False])
def test_a_failed_trace_does_not_stop_the_simulation(
    setup: ServerClientSetup,
    caplog: pytest.LogCaptureFixture,
    background: bool,  # noqa: FBT001
    charged: bool,  # noqa: FBT001
) -> None:
    # A bad seed atom fails inside the trace; a missing NonbondedForce fails before it.
    simulation = OpenMMSimulation.from_simulation(build_simulation(charged=charged))
    simulation.reset(setup.app)
    attach(simulation, FieldLineConfig(seed_atoms=(99,), background=background))
    simulation.advance_by_one_step()
    assert POSITIONS_KEY not in setup.server_current_frame
    assert "Could not trace field lines" in caplog.text
    assert setup.server_current_frame.particle_positions is not None


def test_lines_from_a_background_trace_go_out_with_a_later_frame(
    simulation: OpenMMSimulation,
) -> None:
    executor = ManualExecutor()
    FieldLineService(simulation, executor=executor)
    first = simulation.make_regular_frame()
    assert POSITIONS_KEY not in first
    assert len(executor.jobs) == 1

    # Still busy: no lines yet, and no second trace queued behind the first.
    assert POSITIONS_KEY not in simulation.make_regular_frame()
    assert len(executor.jobs) == 1

    executor.run()
    later = simulation.make_regular_frame()
    expected = trace_lines(
        10.0 * np.asarray(first.particle_positions, dtype=np.float64),
        np.array(CHARGES),
        FieldLineConfig(),
    )
    assert unpack(later.frame_dict)[0] == pytest.approx(expected.points / 10.0, abs=1e-6)
    assert len(executor.jobs) == 1  # The next trace has started.


@pytest.mark.parametrize("event", ["configure", "reset"])
def test_a_trace_started_before_a_change_is_thrown_away(
    setup: ServerClientSetup,
    simulation: OpenMMSimulation,
    event: str,
) -> None:
    executor = ManualExecutor()
    service = FieldLineService(simulation, executor=executor)
    simulation.make_regular_frame()
    if event == "configure":
        service.configure(max_lines=5)
    else:
        simulation.reset(setup.app)

    executor.run()
    frame = simulation.make_regular_frame()
    assert POSITIONS_KEY not in frame
    assert len(executor.jobs) == 1  # A fresh trace with the new state.
    executor.run()
    assert len(unpack(simulation.make_regular_frame().frame_dict)[1]) > 0


def test_a_real_background_thread_publishes_lines_and_stops(
    simulation: OpenMMSimulation,
) -> None:
    service = FieldLineService(simulation)
    deadline = time.monotonic() + 10.0
    while POSITIONS_KEY not in simulation.make_regular_frame():
        assert time.monotonic() < deadline, "No lines arrived from the background thread."
        time.sleep(0.01)

    service.detach()
    for thread in threading.enumerate():
        if thread.name.startswith("fieldlines"):
            thread.join(timeout=5.0)
            assert not thread.is_alive()


def test_detach_restores_the_simulation(
    setup: ServerClientSetup,
    simulation: OpenMMSimulation,
) -> None:
    service = FieldLineService(simulation)
    service.detach()
    simulation.advance_by_one_step()
    assert POSITIONS_KEY not in setup.server_current_frame
    assert getattr(simulation.make_regular_frame, "__self__", None) is simulation
