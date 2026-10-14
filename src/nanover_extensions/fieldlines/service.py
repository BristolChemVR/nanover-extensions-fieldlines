"""Publish electric field lines for a running OpenMM simulation in NanoVer frames.

The lines are added to the simulation's regular frames under three keys:

- `fieldlines.positions`: float32 (M, 3), line vertices in nanometers.
- `fieldlines.lengths`: uint32 (L,), number of vertices in each line.
- `fieldlines.colors`: uint8 (M, 4), RGBA colour of each vertex, set by field strength.

All three are sent as raw bytes. Vertices of a line are ordered along the field
direction (from positive towards negative charge).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, fields, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from openmm import NonbondedForce, unit

from nanover_extensions.fieldlines.core import CoulombField, FieldLines, trace

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt
    from nanover.core import AppServer
    from nanover.openmm import OpenMMSimulation
    from nanover.trajectory import FrameData
    from openmm import System
    from openmm.app import Simulation

    from nanover_extensions.fieldlines.core import FloatArray

logger = logging.getLogger(__name__)

POSITIONS_KEY = "fieldlines.positions"
LENGTHS_KEY = "fieldlines.lengths"
COLORS_KEY = "fieldlines.colors"
CONFIGURE_COMMAND = "fieldlines/configure"

ANGSTROM_PER_NM = 10.0
MIN_LINE_POINTS = 3
"""The client's spline nodes need at least three vertices per line."""

_WEAK_COLOR = np.array([0.15, 0.35, 1.0])
_STRONG_COLOR = np.array([1.0, 0.9, 0.2])
_INDEX_FIELDS = ("source_atoms", "seed_atoms")


@dataclass(frozen=True)
class FieldLineConfig:
    """Settings for tracing and publishing field lines.

    Lengths are in angstrom and field strengths in MV/cm.

    Attributes:
        enabled: Whether lines are published.
        stride: Trace on every `stride`-th frame. Other frames keep the previous lines.
        source_atoms: Indices of the atoms whose charges make the field. All atoms if None.
        seed_atoms: Indices of the atoms to start lines around. If None, the
            `auto_seed_atoms` source atoms with the largest absolute charge are used.
        auto_seed_atoms: Number of atoms picked when `seed_atoms` is None.
        seeds_per_atom: Number of lines started around each seed atom.
        seed_radius: Distance of the seeds from their atom.
        max_lines: Upper limit on the number of lines traced.
        debye_length: Debye-Hueckel screening length. No screening if None.
        cutoff: Distance beyond which charges are ignored. No cutoff if None.
        step: Tracing step length.
        max_steps: Maximum number of steps in each direction from a seed.
        min_points: Lines with fewer vertices than this are dropped.
        weak_strength: Field strength at (and below) which lines get the weak colour.
        strong_strength: Field strength at (and above) which lines get the strong colour.
    """

    enabled: bool = True
    stride: int = 1
    source_atoms: tuple[int, ...] | None = None
    seed_atoms: tuple[int, ...] | None = None
    auto_seed_atoms: int = 8
    seeds_per_atom: int = 8
    seed_radius: float = 1.5
    max_lines: int = 200
    debye_length: float | None = None
    cutoff: float | None = 12.0
    step: float = 0.25
    max_steps: int = 100
    min_points: int = 6
    weak_strength: float = 1.0
    strong_strength: float = 1000.0

    def __post_init__(self) -> None:
        positive = ("stride", "seeds_per_atom", "seed_radius", "max_lines", "step", "max_steps")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"`{name}` must be positive, not {getattr(self, name)}.")
        for name in ("debye_length", "cutoff"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"`{name}` must be positive or None, not {value}.")
        if self.auto_seed_atoms < 0:
            raise ValueError(f"`auto_seed_atoms` must not be negative, not {self.auto_seed_atoms}.")
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


def _select(indices: tuple[int, ...] | None, count: int, name: str) -> npt.NDArray[np.intp]:
    """Turn an optional index list into an index array, checking the range."""
    if indices is None:
        return np.arange(count)
    selected = np.array(indices, dtype=np.intp)
    if len(selected) and (selected.min() < 0 or selected.max() >= count):
        raise ValueError(f"`{name}` has indices outside the {count} atoms of the system.")
    return selected


def make_seeds(positions: FloatArray, charges: FloatArray, config: FieldLineConfig) -> FloatArray:
    """Place the starting points of the lines on small spheres around the seed atoms.

    Args:
        positions: Atom positions in angstrom, shape (N, 3).
        charges: Atom charges, shape (N,).
        config: The settings that choose the seed atoms and the sphere.

    Returns:
        Seed points in angstrom, at most `config.max_lines` of them, shape (S, 3).
    """
    if config.seed_atoms is None:
        sources = _select(config.source_atoms, len(positions), "source_atoms")
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


def trace_lines(positions: FloatArray, charges: FloatArray, config: FieldLineConfig) -> FieldLines:
    """Trace the field lines of a set of atoms.

    Args:
        positions: Atom positions in angstrom, shape (N, 3).
        charges: Atom charges in elementary charges, shape (N,).
        config: The tracing settings.

    Returns:
        The lines with at least `config.min_points` vertices, in angstrom and MV/cm.
    """
    sources = _select(config.source_atoms, len(positions), "source_atoms")
    sources = sources[charges[sources] != 0]
    seeds = make_seeds(positions, charges, config)
    if len(sources) == 0 or len(seeds) == 0:
        return FieldLines(np.zeros((0, 3)), np.zeros(0), np.zeros(1, dtype=np.intp))

    field = CoulombField(
        positions[sources],
        charges[sources],
        debye_length=config.debye_length,
        cutoff=config.cutoff,
    )
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


_NO_LINES = FieldLines(np.zeros((0, 3)), np.zeros(0), np.zeros(1, dtype=np.intp))


class FieldLineService:
    """Adds field lines to the frames of an OpenMM simulation served by NanoVer.

    Args:
        simulation: The NanoVer OpenMM simulation to follow.
        config: Initial settings. Defaults are used if None.
    """

    def __init__(self, simulation: OpenMMSimulation, config: FieldLineConfig | None = None) -> None:
        self.config = config or FieldLineConfig()
        self._simulation = simulation
        self._make_regular_frame: Callable[..., FrameData] = simulation.make_regular_frame
        self._make_topology_frame: Callable[..., FrameData] = simulation.make_topology_frame
        self._system: System | None = None
        self._charges: FloatArray = np.zeros(0)
        self._frames_until_trace = 0
        self._published = False

        simulation.make_regular_frame = self._regular_frame  # type: ignore[method-assign]
        simulation.make_topology_frame = self._topology_frame  # type: ignore[method-assign]

    def detach(self) -> None:
        """Stop adding lines and give the simulation its own frame methods back."""
        self._simulation.make_regular_frame = self._make_regular_frame  # type: ignore[method-assign]
        self._simulation.make_topology_frame = self._make_topology_frame  # type: ignore[method-assign]

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
        self.config = self.config.updated(**changes)
        self._frames_until_trace = 0
        return asdict(self.config)

    def _topology_frame(self, *args: object, **kwargs: object) -> FrameData:
        # A topology frame starts a new frame sequence, which clears the lines on clients.
        self._frames_until_trace = 0
        self._published = False
        return self._make_topology_frame(*args, **kwargs)

    def _regular_frame(self, positions: object = None) -> FrameData:
        frame = self._make_regular_frame(positions)
        config = self.config
        if not config.enabled:
            if self._published:
                frame.frame_dict.update(pack_lines(_NO_LINES, config))
                self._published = False
            return frame
        if self._frames_until_trace > 0:
            self._frames_until_trace -= 1
            return frame
        self._frames_until_trace = config.stride - 1
        try:
            lines = trace_lines(self._positions(frame), self._current_charges(), config)
            frame.frame_dict.update(pack_lines(lines, config))
            self._published = True
        except Exception:
            # A failed trace must not stop the simulation.
            logger.exception("Could not trace field lines.")
        return frame

    @staticmethod
    def _positions(frame: FrameData) -> FloatArray:
        return ANGSTROM_PER_NM * np.asarray(frame.particle_positions, dtype=np.float64)

    def _current_charges(self) -> FloatArray:
        system = cast("Simulation", self._simulation.simulation).system
        if system is not self._system:
            self._charges = get_charges(system)
            self._system = system
        return self._charges
