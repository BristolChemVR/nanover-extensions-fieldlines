"""Publish electric field lines for a running OpenMM simulation in NanoVer frames.

The lines are added to the simulation's regular frames under three keys:

- `fieldlines.positions`: float32 (M, 3), line vertices in nanometers.
- `fieldlines.lengths`: uint32 (L,), number of vertices in each line.
- `fieldlines.colors`: uint8 (M, 4), RGBA colour of each vertex, set by the potential
  (red negative, white zero, blue positive) or by the field strength.

All three are sent as raw bytes. Vertices of a line are ordered along the field
direction (from positive towards negative charge).

By default the field comes from every charged atom except water and monovalent ions,
screened as in salt water and computed on a grid, and lines start in the space around
the molecule where the field is strongest.
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

from nanover_extensions.fieldlines.core import CoulombField, FieldLines, GridField, Seeds, trace

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Executor, Future

    import numpy.typing as npt
    from nanover.core import AppServer
    from nanover.openmm import OpenMMSimulation
    from nanover.trajectory import FrameData
    from openmm import System
    from openmm.app import Simulation, Topology

    from nanover_extensions.fieldlines.core import Field, FloatArray, IndexArray

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
FIELD_METHODS = ("grid", "direct")
COLORINGS = ("potential", "strength")
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
_NEGATIVE_COLOR = np.array([0.84, 0.12, 0.12])
_ZERO_COLOR = np.array([1.0, 1.0, 1.0])
_POSITIVE_COLOR = np.array([0.12, 0.25, 0.84])
_INDEX_FIELDS = ("source_atoms", "seed_atoms", "region_atoms")
_KEPT_SPACING = 0.5
"""Fraction of `seed_spacing` that kept seeds need between them. The slack stops a seed
from being dropped and replaced every time its neighbour wanders a little closer."""


@dataclass(frozen=True)
class FieldLineConfig:
    """Settings for tracing and publishing field lines.

    Lengths are in angstrom, field strengths in MV/cm and potentials in volts.

    Seeding modes:

    - `strength`: on a grid around the source atoms, at the points where the field is
      strongest, at least `seed_spacing` apart (after VMD's field line seeding).
    - `region`: the same, but only within `seed_shell` of the `region_atoms`.
    - `atoms`: on small spheres around `seed_atoms`.
    - `auto`: `region` if `region_atoms` is set, else `atoms` if `seed_atoms` is set,
      else `strength`.

    Field methods:

    - `grid`: by FFT on a grid (`GridField`). Fast with all charges, but smeared out
      within a couple of grid spacings of each charge.
    - `direct`: summed over the charges at every step (`CoulombField`). Exact, and the
      only method with a `cutoff`, but slow for large systems without one.

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
        keep_seeds: In `strength` and `region` mode, start lines from the last trace's
            seeds, moved with their nearest atoms, and add new seeds only in the gaps,
            so lines stay with the molecule instead of jumping between updates.
        field_method: How the field is computed, one of `FIELD_METHODS`.
        field_spacing: Grid spacing for the `grid` method.
        field_margin: How far the grid reaches past the source and seeding atoms. Lines
            end where they leave the grid.
        debye_length: Debye-Hueckel screening length. No screening if None. The default
            of 8 angstrom is for about 150 mM salt.
        cutoff: Distance beyond which charges are ignored, with the `direct` method
            only. No cutoff if None.
        dielectric: Relative permittivity, or its slope per angstrom if
            `distance_dielectric` is set.
        distance_dielectric: Use a permittivity of `dielectric * r`.
        step: Tracing step length.
        max_steps: Maximum number of steps in each direction from a seed.
        min_strength: Field strength at (and below) which a line ends.
        min_points: Lines with fewer vertices than this are dropped.
        coloring: What sets the colour of the lines, one of `COLORINGS`.
        potential_limit: Potential (plus or minus) at which the colour is fully blue or
            red when colouring by potential.
        weak_strength: Field strength at (and below) which lines get the weak colour
            when colouring by strength.
        strong_strength: Field strength at (and above) which lines get the strong colour
            when colouring by strength.
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
    keep_seeds: bool = True
    field_method: str = "grid"
    field_spacing: float = 1.0
    field_margin: float = 20.0
    debye_length: float | None = 8.0
    cutoff: float | None = None
    dielectric: float = 1.0
    distance_dielectric: bool = False
    step: float = 0.25
    max_steps: int = 100
    min_strength: float = 5.0
    min_points: int = 6
    coloring: str = "potential"
    potential_limit: float = 5.0
    weak_strength: float = 1.0
    strong_strength: float = 1000.0

    def __post_init__(self) -> None:
        self._check_numbers()
        for name, allowed in (
            ("seeding", SEEDING_MODES),
            ("field_method", FIELD_METHODS),
            ("coloring", COLORINGS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"`{name}` must be one of {allowed}, not {getattr(self, name)!r}.")
        if self.cutoff is not None and self.field_method != "direct":
            msg = "`cutoff` must be None unless `field_method` is 'direct'."
            raise ValueError(msg)
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
            "field_spacing",
            "dielectric",
            "potential_limit",
            "step",
            "max_steps",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"`{name}` must be positive, not {getattr(self, name)}.")
        for name in (
            "auto_seed_atoms",
            "seed_clearance",
            "seed_spacing",
            "field_margin",
            "min_strength",
        ):
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


def potential_colors(potentials: FloatArray, limit: float) -> npt.NDArray[np.uint8]:
    """Map potentials to RGBA colours: red negative, white zero, blue positive.

    Args:
        potentials: Potentials, shape (M,).
        limit: Size of the potential at (and beyond) which the colour is fully red or blue.

    Returns:
        Opaque colours, shape (M, 4).
    """
    fraction = np.clip(potentials / limit, -1.0, 1.0)[:, None]
    end = np.where(fraction < 0, _NEGATIVE_COLOR, _POSITIVE_COLOR)
    rgb = _ZERO_COLOR + np.abs(fraction) * (end - _ZERO_COLOR)
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

    The grid points are whole multiples of `spacing`, so they stay put when the centres
    move, and a point in range one frame is exactly the same point the next.

    Args:
        centres: Positions to surround, shape (C, 3), with C > 0.
        shell: Largest distance from the nearest centre.
        spacing: Grid spacing.

    Returns:
        Grid points, shape (G, 3).
    """
    low = np.ceil((centres.min(axis=0) - shell) / spacing)
    high = np.floor((centres.max(axis=0) + shell) / spacing)
    axes = [spacing * np.arange(lo, hi + 1) for lo, hi in zip(low, high, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    distance, _ = cKDTree(centres).query(grid, distance_upper_bound=shell * (1 + 1e-9))
    return grid[np.isfinite(np.asarray(distance, dtype=np.float64))]


def spread_strongest(
    points: FloatArray,
    strengths: FloatArray,
    spacing: float,
    count: int,
    *,
    first: BoolArray | None = None,
    first_spacing: float | None = None,
) -> IndexArray:
    """Pick up to `count` points, strongest first, keeping them `spacing` apart.

    Points with no field are never picked.

    Args:
        points: Candidate points, shape (P, 3).
        strengths: Field magnitude at each candidate, shape (P,).
        spacing: Smallest distance between two picked points.
        count: Largest number of points to pick.
        first: True for points to consider before all others, shape (P,).
        first_spacing: Smallest distance between two `first` points, if not `spacing`.

    Returns:
        Indices of the picked points, the `first` ones before the rest, each group in
        order of decreasing strength.
    """
    is_first = np.zeros(len(points), dtype=bool) if first is None else first
    candidates = np.flatnonzero(strengths > 0)
    order = candidates[np.lexsort((-strengths[candidates], ~is_first[candidates]))]
    near_first = spacing if first_spacing is None else first_spacing
    tree = cKDTree(points)
    blocked = np.zeros(len(points), dtype=bool)
    blocked_first = np.zeros(len(points), dtype=bool)
    chosen: list[int] = []
    for index in order:
        if len(chosen) == count:
            break
        if (blocked_first if is_first[index] else blocked)[index]:
            continue
        chosen.append(int(index))
        blocked[tree.query_ball_point(points[index], spacing)] = True
        blocked_first[tree.query_ball_point(points[index], near_first)] = True
    return np.array(chosen, dtype=np.intp)


def make_seeds(
    positions: FloatArray,
    charges: FloatArray,
    sources: IndexArray,
    field: Field,
    config: FieldLineConfig,
    *,
    previous: FloatArray | None = None,
) -> tuple[FloatArray, IndexArray]:
    """Place the starting points of the lines.

    Args:
        positions: Atom positions in angstrom, shape (N, 3).
        charges: Atom charges, shape (N,).
        sources: Indices of the atoms that make `field`.
        field: The field of the source atoms.
        config: The settings that choose the seeding mode and its parameters.
        previous: Seeds to keep if they can, shape (P, 3). In `strength` and `region`
            mode, those that would pass as candidates are picked before any new ones,
            and need only stay half of `seed_spacing` apart from each other.

    Returns:
        Seed points in angstrom, at most `config.max_lines` of them, shape (S, 3), and
        the index in `previous` of each kept one (-1 for new seeds), shape (S,).
    """
    mode = seeding_mode(config)
    if mode == "atoms":
        seeds = _atom_seeds(positions, charges, sources, config)
        return seeds, np.full(len(seeds), -1, dtype=np.intp)
    if mode == "region":
        region = _select(config.region_atoms or (), len(positions), "region_atoms")
        centres = positions[region]
    else:
        centres = positions[sources]
    if len(centres) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.intp)
    candidates = grid_around(centres, config.seed_shell, config.grid_spacing)
    origin = np.full(len(candidates), -1, dtype=np.intp)
    if previous is not None and len(previous):
        shell, _ = cKDTree(centres).query(previous)
        inside = np.flatnonzero(np.asarray(shell) <= config.seed_shell)
        candidates = np.concatenate([previous[inside], candidates])
        origin = np.concatenate([inside, origin])
    clearance, _ = cKDTree(field.positions).query(candidates)
    admissible = np.asarray(clearance) >= config.seed_clearance
    candidates, origin = candidates[admissible], origin[admissible]
    vectors, _ = field.evaluate(candidates)
    strengths = np.linalg.norm(vectors, axis=1)
    picked = spread_strongest(
        candidates,
        strengths,
        config.seed_spacing,
        config.max_lines,
        first=origin >= 0,
        first_spacing=_KEPT_SPACING * config.seed_spacing,
    )
    return candidates[picked].reshape(-1, 3), origin[picked]


_NO_LINES = FieldLines(np.zeros((0, 3)), np.zeros(0), np.zeros(1, dtype=np.intp), np.zeros(0))


def make_field(
    positions: FloatArray,
    charges: FloatArray,
    sources: IndexArray,
    config: FieldLineConfig,
) -> Field:
    """Build the field of the source atoms with the configured method.

    Args:
        positions: Atom positions in angstrom, shape (N, 3).
        charges: Atom charges, shape (N,).
        sources: Indices of the atoms that make the field, at least one.
        config: The settings that choose the method and the medium.

    Returns:
        The field. A grid also reaches the region and seed atoms.
    """
    if config.field_method == "direct":
        return CoulombField(
            positions[sources],
            charges[sources],
            debye_length=config.debye_length,
            cutoff=config.cutoff,
            dielectric=config.dielectric,
            distance_dielectric=config.distance_dielectric,
        )
    seeding_atoms = [
        _select(atoms, len(positions), name)
        for name in ("region_atoms", "seed_atoms")
        if (atoms := getattr(config, name)) is not None
    ]
    return GridField(
        positions[sources],
        charges[sources],
        debye_length=config.debye_length,
        dielectric=config.dielectric,
        distance_dielectric=config.distance_dielectric,
        spacing=config.field_spacing,
        margin=config.field_margin,
        cover=positions[np.concatenate([np.zeros(0, dtype=np.intp), *seeding_atoms])],
    )


def trace_lines(
    positions: FloatArray,
    charges: FloatArray,
    config: FieldLineConfig,
    solvent: BoolArray | None = None,
    previous_seeds: Seeds | None = None,
) -> FieldLines:
    """Trace the field lines of a set of atoms.

    Args:
        positions: Atom positions in angstrom, shape (N, 3).
        charges: Atom charges in elementary charges, shape (N,).
        config: The tracing settings.
        solvent: True for each solvent atom, shape (N,). No atom is solvent if None.
        previous_seeds: Seeds of the last trace, moved with their atoms and kept where
            allowed, with their ids (see `make_seeds`). Ignored in `atoms` mode.

    Returns:
        The lines with at least `config.min_points` vertices, in angstrom and MV/cm,
        with potentials if coloured by potential, their seeds and their ids.
    """
    sources = source_indices(charges, config, solvent)
    if len(sources) == 0:
        return _NO_LINES
    field = make_field(positions, charges, sources, config)
    previous = None if seeding_mode(config) == "atoms" else previous_seeds
    followed = None if previous is None else previous.follow(positions)
    points, origin = make_seeds(positions, charges, sources, field, config, previous=followed)
    if len(points) == 0:
        return _NO_LINES
    # Kept seeds come first and keep their ids; new ones are numbered on from `next_id`.
    new = origin < 0
    next_id = 0 if previous is None else previous.next_id
    ids = next_id + np.cumsum(new) - 1
    if previous is not None:
        ids[~new] = previous.ids[origin[~new]]
    seeds = Seeds.anchor(points, ids, positions, sources, next_id + int(new.sum()))
    lines = trace(
        field,
        points,
        step=config.step,
        max_steps=config.max_steps,
        min_strength=config.min_strength,
    )

    lengths = np.diff(lines.offsets)
    keep = lengths >= config.min_points
    vertices = np.repeat(keep, lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths[keep])]).astype(np.intp)
    vertex_points = lines.points[vertices]
    potentials = field.potential(vertex_points) if config.coloring == "potential" else None
    return FieldLines(
        vertex_points,
        lines.strengths[vertices],
        offsets,
        potentials,
        seeds,
        seeds.ids[keep],
    )


def pack_lines(lines: FieldLines, config: FieldLineConfig) -> dict[str, bytes]:
    """Convert lines to the frame keys and byte layout the client reads.

    Args:
        lines: Lines in angstrom and MV/cm, with potentials if coloured by potential.
        config: The settings that choose the colouring and its scale.

    Returns:
        The three `fieldlines.*` frame entries.

    Raises:
        ValueError: If the lines are coloured by potential but have no potentials.
    """
    if config.coloring == "strength":
        colors = strength_colors(lines.strengths, config.weak_strength, config.strong_strength)
    elif lines.potentials is None:
        msg = "The lines have no potentials to colour them by."
        raise ValueError(msg)
    else:
        colors = potential_colors(lines.potentials, config.potential_limit)
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
        # Seeds of the last published lines, to start the next trace from.
        self._seeds: Seeds | None = None
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
        # By keyword: newer NanoVer servers make every argument after the callback keyword-only.
        app_server.register_command(
            CONFIGURE_COMMAND,
            self.configure,
            default_arguments=asdict(self.config),
        )

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
        self._seeds = None

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
            positions = self._positions(frame)
            lines = trace_lines(positions, charges, config, solvent, self._previous_seeds(config))
        except Exception:
            # A failed trace must not stop the simulation.
            logger.exception(_TRACE_FAILED)
            return
        self._publish(frame, lines, config)

    def _start(self, frame: FrameData, config: FieldLineConfig) -> None:
        try:
            charges, solvent = self._system_data()
            positions = self._positions(frame)  # A new array, so the trace owns it.
            future = self._get_executor().submit(
                trace_lines,
                positions,
                charges,
                config,
                solvent,
                self._previous_seeds(config),
            )
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
        self._seeds = lines.seeds

    def _previous_seeds(self, config: FieldLineConfig) -> Seeds | None:
        return self._seeds if config.keep_seeds else None

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
