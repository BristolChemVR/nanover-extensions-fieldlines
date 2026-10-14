"""Publish electric field lines for a running OpenMM simulation in NanoVer frames.

The lines are added to the simulation's regular frames under three keys:

- `fieldlines.positions`: float32 (M, 3), line vertices in nanometers.
- `fieldlines.lengths`: uint32 (L,), number of vertices in each line.
- `fieldlines.colors`: uint8 (M, 4), RGBA colour of each vertex, set by field strength.

All three are sent as raw bytes. Vertices of a line are ordered along the field
direction (from positive towards negative charge).

By default the field comes from every charged atom except water and monovalent ions,
and lines start in the space around the molecule where the field is strongest.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from openmm import NonbondedForce, unit
from scipy.spatial import cKDTree

from nanover_extensions.fieldlines.core import CoulombField, FieldLines, trace

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Executor, Future

    import numpy.typing as npt
    from nanover.core import AppServer
    from nanover.openmm import OpenMMSimulation
    from nanover.trajectory import FrameData
    from openmm import System
    from openmm.app import Simulation, Topology

    from nanover_extensions.fieldlines.core import FloatArray, IndexArray

    BoolArray = npt.NDArray[np.bool_]

logger = logging.getLogger(__name__)

POSITIONS_KEY = "fieldlines.positions"
LENGTHS_KEY = "fieldlines.lengths"
COLORS_KEY = "fieldlines.colors"
CONFIGURE_COMMAND = "fieldlines/configure"

ANGSTROM_PER_NM = 10.0
MIN_LINE_POINTS = 3
"""The client's spline nodes need at least three vertices per line."""

SEEDING_MODES = ("auto", "strength", "region", "atoms")
SOLVENT_RESIDUES = frozenset(
    {
        # Water models.
        "HOH", "WAT", "SOL", "H2O", "TIP", "TIP3", "TIP4", "TIP5", "T3P", "T4P", "SPC",
        # Monovalent ions. Multivalent ions are often structural, so they stay sources.
        "NA", "CL", "K", "LI", "RB", "CS", "Na+", "Cl-", "K+", "SOD", "CLA", "POT", "CES",
    },
)  # fmt: skip
"""Residue names left out of the field unless `include_solvent` is set."""

_TRACE_FAILED = "Could not trace field lines."
_WEAK_COLOR = np.array([0.15, 0.35, 1.0])
_STRONG_COLOR = np.array([1.0, 0.9, 0.2])
_INDEX_FIELDS = ("source_atoms", "seed_atoms", "region_atoms")


@dataclass(frozen=True)
class FieldLineConfig:
    """Settings for tracing and publishing field lines.

    Lengths are in angstrom and field strengths in MV/cm.

    Seeding modes:

    - `strength`: on a grid around the source atoms, at the points where the field is
      strongest, at least `seed_spacing` apart (after VMD's field line seeding).
    - `region`: the same, but only within `seed_shell` of the `region_atoms`.
    - `atoms`: on small spheres around `seed_atoms`.
    - `auto`: `region` if `region_atoms` is set, else `atoms` if `seed_atoms` is set,
      else `strength`.

    Attributes:
        enabled: Whether lines are published.
        stride: Start a trace at most every `stride` frames. Other frames keep the
            previous lines. In the background, a trace waits for the previous one to end.
        background: Trace on a helper thread instead of while the frame is made.
        seeding: How lines are started, one of `SEEDING_MODES`.
        source_atoms: Indices of the atoms whose charges make the field. If None, all
            atoms except solvent (see `include_solvent`).
        include_solvent: Let water and monovalent ions make the field when
            `source_atoms` is None.
        region_atoms: Atoms to start lines around in `region` mode.
        seed_atoms: Atoms to start lines around in `atoms` mode. If None, the
            `auto_seed_atoms` source atoms with the largest absolute charge are used.
        auto_seed_atoms: Number of atoms picked when `seed_atoms` is None.
        seeds_per_atom: Number of lines started around each seed atom.
        seed_radius: Distance of the seeds from their atom.
        grid_spacing: Spacing of the grid of candidate seeds in `strength` and `region` mode.
        seed_shell: Largest distance of a candidate seed from the atoms it surrounds.
        seed_clearance: Smallest distance of a candidate seed from any source atom.
        seed_spacing: Smallest distance between two seeds.
        max_lines: Upper limit on the number of lines traced.
        debye_length: Debye-Hueckel screening length. No screening if None.
        cutoff: Distance beyond which charges are ignored. No cutoff if None.
        dielectric: Relative permittivity, or its slope per angstrom if
            `distance_dielectric` is set.
        distance_dielectric: Use a permittivity of `dielectric * r`.
        step: Tracing step length.
        max_steps: Maximum number of steps in each direction from a seed.
        min_points: Lines with fewer vertices than this are dropped.
        weak_strength: Field strength at (and below) which lines get the weak colour.
        strong_strength: Field strength at (and above) which lines get the strong colour.
    """

    enabled: bool = True
    stride: int = 1
    background: bool = True
    seeding: str = "auto"
    source_atoms: tuple[int, ...] | None = None
    include_solvent: bool = False
    region_atoms: tuple[int, ...] | None = None
    seed_atoms: tuple[int, ...] | None = None
    auto_seed_atoms: int = 8
    seeds_per_atom: int = 8
    seed_radius: float = 1.5
    grid_spacing: float = 3.0
    seed_shell: float = 8.0
    seed_clearance: float = 2.5
    seed_spacing: float = 4.0
    max_lines: int = 200
    debye_length: float | None = None
    cutoff: float | None = 12.0
    dielectric: float = 1.0
    distance_dielectric: bool = False
    step: float = 0.25
    max_steps: int = 100
    min_points: int = 6
    weak_strength: float = 1.0
    strong_strength: float = 1000.0

    def __post_init__(self) -> None:
        self._check_numbers()
        if self.seeding not in SEEDING_MODES:
            raise ValueError(f"`seeding` must be one of {SEEDING_MODES}, not {self.seeding!r}.")
        if self.seeding == "region" and self.region_atoms is None:
            msg = "`region_atoms` must be set for `region` seeding."
            raise ValueError(msg)

    def _check_numbers(self) -> None:
        positive = (
            "stride",
            "seeds_per_atom",
            "seed_radius",
            "grid_spacing",
            "seed_shell",
            "max_lines",
            "dielectric",
            "step",
            "max_steps",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"`{name}` must be positive, not {getattr(self, name)}.")
        for name in ("auto_seed_atoms", "seed_clearance", "seed_spacing"):
            if getattr(self, name) < 0:
                raise ValueError(f"`{name}` must not be negative, not {getattr(self, name)}.")
        for name in ("debye_length", "cutoff"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"`{name}` must be positive or None, not {value}.")
        if self.min_points < MIN_LINE_POINTS:
            raise ValueError(f"`min_points` must be at least {MIN_LINE_POINTS}.")
        if not 0 < self.weak_strength < self.strong_strength:
            msg = "`weak_strength` must be positive and below `strong_strength`."
            raise ValueError(msg)

    def updated(self, **changes: Any) -> FieldLineConfig:  # noqa: ANN401
        """Return a copy with some settings changed.

        Args:
            **changes: New values by setting name. Atom index lists may be any sequence.

        Returns:
            The new settings.

        Raises:
            ValueError: If a name is not a setting or a value is not allowed.
        """
        unknown = set(changes) - {field.name for field in fields(self)}
        if unknown:
            raise ValueError(f"Unknown field line settings: {sorted(unknown)}.")
        for name in _INDEX_FIELDS:
            if changes.get(name) is not None:
                changes[name] = tuple(int(index) for index in changes[name])
        return replace(self, **changes)


def get_charges(system: System) -> FloatArray:
    """Read the particle charges, in elementary charges, from an OpenMM system.

    Args:
        system: A system with a `NonbondedForce`.

    Returns:
        The charge of each particle, shape (N,).

    Raises:
        ValueError: If the system has no `NonbondedForce`.
    """
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            return np.array(
                [
                    force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                    for i in range(force.getNumParticles())
                ],
                dtype=np.float64,
            )
    msg = "The system has no NonbondedForce to read charges from."
    raise ValueError(msg)


def solvent_mask(topology: Topology) -> BoolArray:
    """Mark the atoms that belong to water or monovalent ion residues.

    Args:
        topology: The OpenMM topology of the system.

    Returns:
        True for each solvent atom, shape (N,).
    """
    return np.array([atom.residue.name in SOLVENT_RESIDUES for atom in topology.atoms()], bool)


def fibonacci_sphere(count: int) -> FloatArray:
    """Return `count` points spread evenly over the unit sphere, shape (count, 3)."""
    index = np.arange(count) + 0.5
    z = 1.0 - 2.0 * index / count
    radius = np.sqrt(1.0 - z * z)
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), z], axis=1)


def strength_colors(strengths: FloatArray, weak: float, strong: float) -> npt.NDArray[np.uint8]:
    """Map field strengths to RGBA colours on a logarithmic scale.

    Args:
        strengths: Field magnitudes, shape (M,).
        weak: Strength at (and below) which the weak colour is used.
        strong: Strength at (and above) which the strong colour is used.

    Returns:
        Opaque colours, shape (M, 4).
    """
    with np.errstate(divide="ignore"):
        fraction = (np.log(strengths) - np.log(weak)) / (np.log(strong) - np.log(weak))
    fraction = np.clip(fraction, 0.0, 1.0)[:, None]
    rgb = (1.0 - fraction) * _WEAK_COLOR + fraction * _STRONG_COLOR
    rgba = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)
    return np.round(255.0 * rgba).astype(np.uint8)


def _select(indices: tuple[int, ...], count: int, name: str) -> IndexArray:
    """Turn an index list into an index array, checking the range."""
    selected = np.array(indices, dtype=np.intp)
    if len(selected) and (selected.min() < 0 or selected.max() >= count):
        raise ValueError(f"`{name}` has indices outside the {count} atoms of the system.")
    return selected


def source_indices(
    charges: FloatArray,
    config: FieldLineConfig,
    solvent: BoolArray | None = None,
) -> IndexArray:
    """Choose the atoms whose charges make the field.

    Args:
        charges: Atom charges, shape (N,).
        config: The settings that choose the source atoms.
        solvent: True for each solvent atom, shape (N,). No atom is solvent if None.

    Returns:
        Indices of the charged source atoms.
    """
    if config.source_atoms is not None:
        sources = _select(config.source_atoms, len(charges), "source_atoms")
    elif solvent is None or config.include_solvent:
        sources = np.arange(len(charges))
    else:
        sources = np.flatnonzero(~solvent)
    return np.asarray(sources[charges[sources] != 0], dtype=np.intp)


def seeding_mode(config: FieldLineConfig) -> str:
    """Resolve `auto` seeding to the mode that will be used."""
    if config.seeding != "auto":
        return config.seeding
    if config.region_atoms is not None:
        return "region"
    if config.seed_atoms is not None:
        return "atoms"
    return "strength"


def _atom_seeds(
    positions: FloatArray,
    charges: FloatArray,
    sources: IndexArray,
    config: FieldLineConfig,
) -> FloatArray:
    """Place seeds on small spheres around the seed atoms."""
    if config.seed_atoms is None:
        strongest = np.argsort(-np.abs(charges[sources]), kind="stable")
        atoms = sources[strongest[: config.auto_seed_atoms]]
    else:
        atoms = _select(config.seed_atoms, len(positions), "seed_atoms")
    sphere = config.seed_radius * fibonacci_sphere(config.seeds_per_atom)
    seeds = (positions[atoms, None, :] + sphere[None, :, :]).reshape(-1, 3)
    if len(seeds) > config.max_lines:
        # Thin evenly so that every seed atom keeps some of its lines.
        keep = np.linspace(0, len(seeds) - 1, config.max_lines).round().astype(np.intp)
        seeds = seeds[keep]
    return seeds


def grid_around(centres: FloatArray, shell: float, spacing: float) -> FloatArray:
    """Return the points of a cubic grid that lie within `shell` of any centre.

    Args:
        centres: Positions to surround, shape (C, 3), with C > 0.
        shell: Largest distance from the nearest centre.
        spacing: Grid spacing.

    Returns:
        Grid points, shape (G, 3).
    """
    low = centres.min(axis=0) - shell
    high = centres.max(axis=0) + shell
    axes = [np.arange(lo, hi + spacing / 2, spacing) for lo, hi in zip(low, high, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    distance, _ = cKDTree(centres).query(grid, distance_upper_bound=shell * (1 + 1e-9))
    return grid[np.isfinite(np.asarray(distance, dtype=np.float64))]


def spread_strongest(
    points: FloatArray,
    strengths: FloatArray,
    spacing: float,
    count: int,
) -> FloatArray:
    """Pick up to `count` points, strongest first, keeping them `spacing` apart.

    Points with no field are never picked.

    Args:
        points: Candidate points, shape (P, 3).
        strengths: Field magnitude at each candidate, shape (P,).
        spacing: Smallest distance between two picked points.
        count: Largest number of points to pick.

    Returns:
        The picked points in order of decreasing strength, shape (S, 3).
    """
    candidates = np.flatnonzero(strengths > 0)
    order = candidates[np.argsort(-strengths[candidates], kind="stable")]
    tree = cKDTree(points)
    blocked = np.zeros(len(points), dtype=bool)
    chosen: list[int] = []
    for index in order:
        if len(chosen) == count:
            break
        if blocked[index]:
            continue
        chosen.append(int(index))
        blocked[tree.query_ball_point(points[index], spacing)] = True
    return points[np.array(chosen, dtype=np.intp)].reshape(-1, 3)


def make_seeds(
    positions: FloatArray,
    charges: FloatArray,
    sources: IndexArray,
    field: CoulombField,
    config: FieldLineConfig,
) -> FloatArray:
    """Place the starting points of the lines.

    Args:
        positions: Atom positions in angstrom, shape (N, 3).
        charges: Atom charges, shape (N,).
        sources: Indices of the atoms that make `field`.
        field: The field of the source atoms.
        config: The settings that choose the seeding mode and its parameters.

    Returns:
        Seed points in angstrom, at most `config.max_lines` of them, shape (S, 3).
    """
    mode = seeding_mode(config)
    if mode == "atoms":
        return _atom_seeds(positions, charges, sources, config)
    if mode == "region":
        region = _select(config.region_atoms or (), len(positions), "region_atoms")
        centres = positions[region]
    else:
        centres = positions[sources]
    if len(centres) == 0:
        return np.zeros((0, 3))
    candidates = grid_around(centres, config.seed_shell, config.grid_spacing)
    clearance, _ = cKDTree(field.positions).query(candidates)
    candidates = candidates[clearance >= config.seed_clearance]
    vectors, _ = field.evaluate(candidates)
    strengths = np.linalg.norm(vectors, axis=1)
    return spread_strongest(candidates, strengths, config.seed_spacing, config.max_lines)


_NO_LINES = FieldLines(np.zeros((0, 3)), np.zeros(0), np.zeros(1, dtype=np.intp))


def trace_lines(
    positions: FloatArray,
    charges: FloatArray,
    config: FieldLineConfig,
    solvent: BoolArray | None = None,
) -> FieldLines:
    """Trace the field lines of a set of atoms.

    Args:
        positions: Atom positions in angstrom, shape (N, 3).
        charges: Atom charges in elementary charges, shape (N,).
        config: The tracing settings.
        solvent: True for each solvent atom, shape (N,). No atom is solvent if None.

    Returns:
        The lines with at least `config.min_points` vertices, in angstrom and MV/cm.
    """
    sources = source_indices(charges, config, solvent)
    if len(sources) == 0:
        return _NO_LINES
    field = CoulombField(
        positions[sources],
        charges[sources],
        debye_length=config.debye_length,
        cutoff=config.cutoff,
        dielectric=config.dielectric,
        distance_dielectric=config.distance_dielectric,
    )
    seeds = make_seeds(positions, charges, sources, field, config)
    if len(seeds) == 0:
        return _NO_LINES
    lines = trace(field, seeds, step=config.step, max_steps=config.max_steps)

    lengths = np.diff(lines.offsets)
    keep = lengths >= config.min_points
    vertices = np.repeat(keep, lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths[keep])]).astype(np.intp)
    return FieldLines(lines.points[vertices], lines.strengths[vertices], offsets)


def pack_lines(lines: FieldLines, config: FieldLineConfig) -> dict[str, bytes]:
    """Convert lines to the frame keys and byte layout the client reads.

    Args:
        lines: Lines in angstrom and MV/cm.
        config: The settings that choose the colour scale.

    Returns:
        The three `fieldlines.*` frame entries.
    """
    colors = strength_colors(lines.strengths, config.weak_strength, config.strong_strength)
    return {
        POSITIONS_KEY: (lines.points / ANGSTROM_PER_NM).astype(np.float32).tobytes(),
        LENGTHS_KEY: np.diff(lines.offsets).astype(np.uint32).tobytes(),
        COLORS_KEY: colors.tobytes(),
    }


@dataclass(frozen=True)
class _Job:
    """A trace running in the background, and what it was started with."""

    future: Future[FieldLines]
    generation: int
    config: FieldLineConfig


class FieldLineService:
    """Adds field lines to the frames of an OpenMM simulation served by NanoVer.

    With `background` set (the default), lines are traced on a helper thread so the
    simulation never waits for them. A trace starts from a copy of the frame's
    positions, and its lines go out with the first frame made after it finishes.
    Only one trace runs at a time, and a trace started before a reset or a change of
    settings is thrown away.

    Args:
        simulation: The NanoVer OpenMM simulation to follow.
        config: Initial settings. Defaults are used if None.
        executor: Runs background traces. If None, the service starts its own
            single-thread executor when it first needs one, and shuts it down on `detach`.
    """

    def __init__(
        self,
        simulation: OpenMMSimulation,
        config: FieldLineConfig | None = None,
        executor: Executor | None = None,
    ) -> None:
        self.config = config or FieldLineConfig()
        self._simulation = simulation
        self._make_regular_frame: Callable[..., FrameData] = simulation.make_regular_frame
        self._make_topology_frame: Callable[..., FrameData] = simulation.make_topology_frame
        self._system: System | None = None
        self._charges: FloatArray = np.zeros(0)
        self._solvent: BoolArray = np.zeros(0, dtype=bool)
        self._frames_until_trace = 0
        self._published = False
        self._executor = executor
        self._owns_executor = executor is None
        self._job: _Job | None = None
        # Raised by every reset and change of settings, so older traces can be recognised.
        self._generation = 0
        # Commands arrive on a server thread while frames are made on the simulation's.
        self._lock = threading.Lock()

        simulation.make_regular_frame = self._regular_frame  # type: ignore[method-assign]
        simulation.make_topology_frame = self._topology_frame  # type: ignore[method-assign]

    def detach(self) -> None:
        """Stop adding lines and give the simulation its own frame methods back."""
        self._simulation.make_regular_frame = self._make_regular_frame  # type: ignore[method-assign]
        self._simulation.make_topology_frame = self._make_topology_frame  # type: ignore[method-assign]
        with self._lock:
            self._restart()
            self._job = None
            if self._owns_executor and self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

    def register_commands(self, app_server: AppServer) -> None:
        """Let clients change the settings with the `fieldlines/configure` command.

        Args:
            app_server: The NanoVer server that clients connect to.
        """
        app_server.register_command(CONFIGURE_COMMAND, self.configure, asdict(self.config))

    def configure(self, **changes: Any) -> dict[str, Any]:  # noqa: ANN401
        """Change some settings. The next frame is traced with the new settings.

        Args:
            **changes: New values by setting name, see `FieldLineConfig`.

        Returns:
            All current settings.

        Raises:
            ValueError: If a name is not a setting or a value is not allowed.
        """
        config = self.config.updated(**changes)
        with self._lock:
            self.config = config
            self._restart()
        return asdict(config)

    def _restart(self) -> None:
        """Trace on the next frame and disown any trace already running. Needs the lock."""
        self._frames_until_trace = 0
        self._generation += 1

    def _topology_frame(self, *args: object, **kwargs: object) -> FrameData:
        # A topology frame starts a new frame sequence, which clears the lines on clients.
        with self._lock:
            self._restart()
            self._published = False
        return self._make_topology_frame(*args, **kwargs)

    def _regular_frame(self, positions: object = None) -> FrameData:
        frame = self._make_regular_frame(positions)
        with self._lock:
            self._add_lines(frame)
        return frame

    def _add_lines(self, frame: FrameData) -> None:
        """Add new lines to a frame, or clear them, as the settings ask. Needs the lock."""
        config = self.config
        if not config.enabled:
            if self._published:
                self._publish(frame, _NO_LINES, config)
                self._published = False
            return
        self._collect(frame)
        if self._frames_until_trace > 0:
            self._frames_until_trace -= 1
        elif not config.background:
            self._frames_until_trace = config.stride - 1
            self._trace_now(frame, config)
        elif self._job is None:
            # If a trace is still running, try again on the next frame.
            self._frames_until_trace = config.stride - 1
            self._start(frame, config)
            # An executor that finishes straight away has lines for this frame.
            self._collect(frame)

    def _trace_now(self, frame: FrameData, config: FieldLineConfig) -> None:
        try:
            charges, solvent = self._system_data()
            lines = trace_lines(self._positions(frame), charges, config, solvent)
        except Exception:
            # A failed trace must not stop the simulation.
            logger.exception(_TRACE_FAILED)
            return
        self._publish(frame, lines, config)

    def _start(self, frame: FrameData, config: FieldLineConfig) -> None:
        try:
            charges, solvent = self._system_data()
            positions = self._positions(frame)  # A new array, so the trace owns it.
            future = self._get_executor().submit(trace_lines, positions, charges, config, solvent)
        except Exception:
            logger.exception(_TRACE_FAILED)
            return
        self._job = _Job(future, self._generation, config)

    def _collect(self, frame: FrameData) -> None:
        """Publish the lines of a finished background trace, unless they are out of date."""
        job = self._job
        if job is None or not job.future.done():
            return
        self._job = None
        if job.generation != self._generation:
            return
        try:
            lines = job.future.result()
        except Exception:
            logger.exception(_TRACE_FAILED)
            return
        self._publish(frame, lines, job.config)

    def _publish(self, frame: FrameData, lines: FieldLines, config: FieldLineConfig) -> None:
        frame.frame_dict.update(pack_lines(lines, config))
        self._published = True

    def _get_executor(self) -> Executor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fieldlines")
        return self._executor

    @staticmethod
    def _positions(frame: FrameData) -> FloatArray:
        return ANGSTROM_PER_NM * np.asarray(frame.particle_positions, dtype=np.float64)

    def _system_data(self) -> tuple[FloatArray, BoolArray]:
        """Charges and solvent atoms of the current system, read once per system."""
        simulation = cast("Simulation", self._simulation.simulation)
        if simulation.system is not self._system:
            self._charges = get_charges(simulation.system)
            self._solvent = solvent_mask(simulation.topology)
            self._system = simulation.system
        return self._charges, self._solvent
