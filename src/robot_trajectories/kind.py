"""A kind of trajectory is a config record (from V2): a frozen, keyword-only dataclass whose fields are the configuration
and whose class carries the catalog tags. Python is the ground truth; JSON, the repr and the catalog card are generated.

Zach (Daily Note, 2026-11-18): "You pass in a robot description and a configuration, and then you get out like a very
long joint trajectory". A record is one line of that configuration: ``Chirp(joint="can11", start_hz=0.2, end_hz=5.0)``.

Adding a kind is one class in one file: subclass ``Kind``, declare the fields with ``param(default, unit, doc)``, write
``pieces(context)`` with the shared kernels. ``__init_subclass__`` makes it a frozen keyword-only dataclass and puts it
in the catalog; ``kinds/`` imports every module in its folder, so a new file there needs no other edit.

The class tags speak the full catalog's vocabulary (``trajectory_catalog/kind.py``): ``catalog_id`` is the record's row
there, ``needs`` are the catalog's needs that this code reads, ``without`` the soft ones (what it does instead), and
``starts_as`` says time signal or path. The entry level (Zach's ladder: joint limits -> kinematics -> dynamics -> torque
limits) is derived from ``needs`` by the catalog's own rule, read from the catalog's vocabulary (one place for both).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from . import kernels as k

from .time_law import TimeLaw
# The catalog's input ladder, cumulative: each rung includes the ones before it. WHY imported, not copied (the round's v6
# kept a copy because its package could not import the round's catalog/): the catalog is part of this package now, so the
# records that generate and the catalog rows they claim read one vocabulary (stdlib only, so the leaf rule still holds).
from .trajectory_catalog.kind import ENTRY_LEVEL_OF_NEED, ENTRY_LEVELS

_REGISTRY: dict[str, type[Kind]] = {}

NEEDS: tuple[str, ...] = tuple(ENTRY_LEVEL_OF_NEED)


def param(default: Any, unit: str = "", doc: str = "") -> Any:
    """A record field with its SI unit and one line of meaning (``describe()`` and the landing page show both)."""
    return dataclasses.field(default=default, metadata={"unit": unit, "doc": doc})


def entry_level(needs: Iterable[str]) -> str:
    """The lowest rung of the ladder whose inputs cover ``needs`` (the catalog's ``entry_level`` rule)."""
    rungs = [ENTRY_LEVEL_OF_NEED[need] for need in needs if ENTRY_LEVEL_OF_NEED.get(need)]
    return max(rungs, key=ENTRY_LEVELS.index, default=ENTRY_LEVELS[0])


# ------------------------------------------------------------------ the reachable set, as a record reads it


@dataclass(frozen=True, eq=False)
class ReachSet:
    """A reach map as the generator reads it: every joint solution, which are valid, and the map's own grid step (V2)."""

    configurations: np.ndarray  # (M, J)
    valid: np.ndarray  # (M,) bool
    grid_step: np.ndarray  # (J,) rad

    @classmethod
    def from_reach_map(cls, reach_map: Any) -> ReachSet:
        """Duck-typed on bam_reach.ReachMap's ``joint_solutions`` (P, O, J) and ``valid_mask`` (P, O)."""
        solutions = np.asarray(reach_map.joint_solutions, float)
        configurations = solutions.reshape(-1, solutions.shape[-1])
        finite = np.all(np.isfinite(configurations), axis=1)
        # WHY the finite test: a real ReachMap may store NaN where IK found nothing; those rows are neither valid nor a point.
        valid = np.asarray(reach_map.valid_mask, bool).reshape(-1) & finite
        steps = []
        for column in configurations[finite].T:
            values = np.unique(np.round(column, 9))
            steps.append(float(np.min(np.diff(values))) if len(values) > 1 else math.inf)
        return cls(configurations, valid, np.array(steps))

    @property
    def reachable(self) -> np.ndarray:
        return self.configurations[self.valid]

    @property
    def unreachable(self) -> np.ndarray:
        return self.configurations[~self.valid & np.all(np.isfinite(self.configurations), axis=1)]

    def reachable_on_grid(self, grid_step: float) -> np.ndarray:
        """The reachable configurations on a coarser grid (``grid_step`` a whole multiple of the map's step), in map order."""
        stride = grid_step / self.grid_step
        if np.any(np.abs(stride - np.round(stride)) > 1e-6) or np.any(np.round(stride) < 1):
            raise ValueError(f"grid_step {math.degrees(grid_step):.3g}° is not a whole multiple of the reach map's step "
                             f"{np.round(np.degrees(self.grid_step), 3).tolist()}°")
        finite = np.where(np.isfinite(self.configurations), self.configurations, 0.0)
        keys = np.round((finite - finite[self.valid].min(axis=0)) / self.grid_step).astype(int)
        on_grid = self.valid & np.all(keys % np.round(stride).astype(int) == 0, axis=1)
        return self.configurations[on_grid]

    def nearest_is_reachable(self, configuration: np.ndarray) -> bool:
        distance = np.max(np.abs(np.nan_to_num(self.configurations, nan=np.inf) - configuration) / self.grid_step, axis=1)
        return bool(self.valid[int(np.argmin(distance))])

    def fingerprint(self) -> str:
        digest = hashlib.sha256(np.ascontiguousarray(self.configurations).tobytes() + self.valid.tobytes())
        return digest.hexdigest()[:16]


@dataclass(frozen=True, eq=False)
class Context:
    """What a record's ``pieces`` reads: the robot's numbers, where this record starts, and the reachable set if known."""

    joint_names: tuple[str, ...]
    bounds: k.Bounds
    rate_hz: float
    start: np.ndarray  # where the previous record ended (the program's start for the first)
    home: np.ndarray
    reach: ReachSet | None = None
    notes: list[str] = field(default_factory=list)  # what a record wants the program summary to say (a clamp, a skip)

    def at(self, start: np.ndarray) -> Context:
        """The same robot, starting somewhere else (a record that holds records, like Repeat, generates them in turn)."""
        return dataclasses.replace(self, start=np.asarray(start, float))

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def joint(self, which: str | int | None) -> int:
        """A joint by name or index; ``None`` is allowed on a 1-DOF robot only."""
        if which is None:
            if len(self.joint_names) == 1:
                return 0
            raise ValueError(f"which joint? this robot has {self.joint_names}: pass joint=<name>")
        if isinstance(which, str):
            if which not in self.joint_names:
                raise ValueError(f"no joint {which!r}; joints are {self.joint_names}")
            return self.joint_names.index(which)
        if not 0 <= int(which) < len(self.joint_names):
            raise ValueError(f"joint index {which} out of range for {self.joint_names}")
        return int(which)

    def joints(self, which: Sequence[str | int] | str | None) -> list[int]:
        """Several joints; ``None`` is every joint."""
        if which is None:
            return list(range(len(self.joint_names)))
        if isinstance(which, (str, int)):
            return [self.joint(which)]
        return [self.joint(name) for name in which]

    def room_around(self, centre: np.ndarray) -> k.Bounds:
        """The box a kind may use around ``centre``: the limit box, or with a reach map the largest box around ``centre``
        that holds no unreachable configuration.

        WHY here and not in each kind (all five proposals converged on it): safe by construction. Every centred kind asks
        the context for its room, so a reach map keeps all of them inside the reachable set at once, and no kind needs
        to know what a reach map is.
        """
        if self.reach is None:
            return self.bounds
        return k.inscribed_box(centre, self.reach.unreachable, self.bounds)


# ------------------------------------------------------------------ the record base


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _plain(value: Any) -> Any:
    """Lists become tuples and numpy scalars Python numbers, so a record hashes, compares and prints cleanly."""
    if isinstance(value, (Kind, TimeLaw)):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_plain(item) for item in value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _encode(value: Any) -> Any:
    """A field value as JSON keeps it: nested records and time laws become dicts, tuples lists."""
    if isinstance(value, Kind):
        return value.to_config()
    if isinstance(value, TimeLaw):
        return value.to_config()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    """The inverse of ``_encode``: a dict with "kind" is a record, with "time_law" a time law."""
    if isinstance(value, dict) and "kind" in value:
        return Kind.from_config(value)
    if isinstance(value, dict) and "time_law" in value:
        return TimeLaw.from_config(value)
    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    return value


@typing.dataclass_transform(kw_only_default=True, field_specifiers=(param, dataclasses.field))
@dataclass(frozen=True, kw_only=True, repr=False)
class Kind:
    """One kind of trajectory as a config record. Subclasses declare fields with ``param`` and implement ``pieces``."""

    id: ClassVar[str] = ""  # snake_case of the class name unless set: the "kind" key in JSON
    catalog_id: ClassVar[str | None] = None  # its row in trajectory_catalog (None: a building block, e.g. Home)
    family: ClassVar[str] = ""
    starts_as: ClassVar[str] = "time_signal"  # or "path": the geometry is the content, a TimeLaw times it
    use_cases: ClassVar[tuple[str, ...]] = ()
    needs: ClassVar[tuple[str, ...]] = ("joint_limits", "velocity_limits", "acceleration_limits")
    without: ClassVar[dict[str, str]] = {}  # soft needs: need -> what the kind does without it (status "fallback")
    unchecked: ClassVar[tuple[str, ...]] = ()  # needs the catalog lists that this package does not check (e.g. collisions)

    label: str = param("", "text", "what this segment is called in the program (default: the kind's name)")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # WHY the dataclass is applied here: a kind is then one plain class statement, and every kind is frozen and
        # keyword-only the same way (dataclass_transform above tells type checkers so). repr=False keeps Kind.__repr__.
        dataclass(frozen=True, kw_only=True, repr=False)(cls)
        if "id" not in cls.__dict__:
            cls.id = _snake(cls.__name__)
        if cls.__name__.startswith("_"):
            return
        unknown = set(cls.needs) - set(NEEDS)
        if unknown:
            raise TypeError(f"{cls.__name__}.needs {sorted(unknown)} are not in the catalog's vocabulary {NEEDS}")
        existing = _REGISTRY.get(cls.id)
        if existing is not None and (existing.__module__, existing.__qualname__) != (cls.__module__, cls.__qualname__):
            raise TypeError(f"kind id {cls.id!r} is taken by {existing.__module__}.{existing.__qualname__}")
        if cls.catalog_id is not None:
            for other in _REGISTRY.values():
                if other.catalog_id == cls.catalog_id and other.id != cls.id:
                    raise TypeError(f"{cls.__name__} and {other.__name__} both claim catalog row {cls.catalog_id!r}")
        _REGISTRY[cls.id] = cls

    def __post_init__(self) -> None:
        for spec in dataclasses.fields(self):
            object.__setattr__(self, spec.name, _plain(getattr(self, spec.name)))
        self.check()

    # ---- what a kind implements

    def check(self) -> None:
        """Refuse impossible field values at construction (override when a kind has any)."""

    def pieces(self, context: Context) -> list[k.Arrays]:
        """The motion, as shared-kernel pieces starting at ``context.start`` and ending at rest."""
        raise NotImplementedError(f"{type(self).__name__}.pieces")

    # ---- what every kind gets

    @classmethod
    def name(cls) -> str:
        return cls.id.replace("_", " ").capitalize()

    @classmethod
    def one_line(cls) -> str:
        return " ".join((cls.__doc__ or "").strip().split("\n\n")[0].split())

    @classmethod
    def entry_level(cls) -> str:
        return entry_level(cls.needs)

    @property
    def display_label(self) -> str:
        return self.label or self.id.replace("_", " ")

    @classmethod
    def fields_info(cls) -> list[dict[str, Any]]:
        """Each configurable field: name, SI unit, default, meaning (label last)."""
        return [{"name": spec.name, "unit": spec.metadata.get("unit", ""), "default": spec.default,
                 "doc": spec.metadata.get("doc", "")} for spec in _label_last(cls)]

    @classmethod
    def describe(cls) -> str:
        """The catalog card: what it is, what it serves, what it needs, and every field with its unit (angles in degrees)."""
        lines = [f"{cls.__name__} ({cls.id}{', catalog ' + cls.catalog_id if cls.catalog_id else ''}; {cls.family}, "
                 f"starts as a {cls.starts_as.replace('_', ' ')}): {cls.one_line()}",
                 f"    use cases: {', '.join(cls.use_cases) or '-'}    needs: {', '.join(cls.needs)} (entry: {cls.entry_level().replace('_', ' ')})"]
        for info in cls.fields_info():
            lines.append(f"    {info['name']:<24} {_shown(info['default'], info['unit']):<30} {info['doc']}")
        return "\n".join(lines)

    def to_config(self) -> dict[str, Any]:
        """``{"kind": id, every field}``. WHY every field and not only the changed ones: a logged config must regenerate
        the same motion even after a default changes."""
        config: dict[str, Any] = {"kind": self.id}
        for spec in _label_last(self):
            value = getattr(self, spec.name)
            if spec.name == "label" and not value:
                continue
            config[spec.name] = _encode(value)
        return config

    @staticmethod
    def from_config(config: dict[str, Any]) -> Kind:
        """The record a ``to_config()`` dict describes (unknown kinds and unknown fields are refused by name)."""
        values = dict(config)
        kind_id = values.pop("kind", None)
        if kind_id not in _REGISTRY:
            raise ValueError(f"unknown kind {kind_id!r}; the catalog has {sorted(_REGISTRY)} "
                             "(a kind defined outside robot_trajectories.kinds must be imported first)")
        cls = _REGISTRY[kind_id]
        known = {spec.name for spec in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"{cls.__name__} has no field {unknown}; its fields are {sorted(known)}")
        return cls(**{name: _decode(value) for name, value in values.items()})

    def __repr__(self) -> str:
        """Only the fields that differ from their defaults, as valid Python (Keras-style): ``Chirp(joint='can11')``."""
        changed = [f"{spec.name}={getattr(self, spec.name)!r}" for spec in _label_last(self)
                   if getattr(self, spec.name) != spec.default]
        return f"{type(self).__name__}({', '.join(changed)})"


def _label_last(record: Any) -> list[dataclasses.Field]:
    """The fields in declaration order with the base's ``label`` moved last (it names the segment, not the motion)."""
    specs = dataclasses.fields(record)
    return [spec for spec in specs if spec.name != "label"] + [spec for spec in specs if spec.name == "label"]


def _shown(value: Any, unit: str) -> str:
    """A default as a person reads it: radians in degrees, other units as they are."""
    if unit == "rad" and isinstance(value, (int, float)):
        return f"{math.degrees(value):.4g}°"
    if unit in ("", "1", "text", "count", "joint", "joints", "seed", "bool", "records", "time law"):
        return repr(value)
    return f"{value!r} {unit}"


# ------------------------------------------------------------------ the catalog of registered kinds


class Catalog(tuple):
    """The record classes, filtered. ``print(catalog(...))`` prints each card; ``.ids`` the kind ids; ``["chirp"]`` one class."""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            for kind in self:
                if key in (kind.id, kind.catalog_id):
                    return kind
            raise KeyError(f"no kind {key!r} here; kinds are {self.ids}")
        return super().__getitem__(key)

    @property
    def ids(self) -> list[str]:
        return [kind.id for kind in self]

    def __str__(self) -> str:
        return "\n\n".join(kind.describe() for kind in self)

    def to_json(self) -> list[dict[str, Any]]:
        """The catalog as data (generated from the records, never edited): what a landing page reads."""
        return [{"id": kind.id, "class": kind.__name__, "catalog_id": kind.catalog_id, "family": kind.family,
                 "starts_as": kind.starts_as, "one_line": kind.one_line(), "use_cases": list(kind.use_cases),
                 "needs": list(kind.needs), "without": dict(kind.without), "entry_level": kind.entry_level(),
                 "fields": [{**info, "default": _encode(info["default"])} for info in kind.fields_info()]} for kind in self]


def catalog(use_case: str | None = None, *, family: str | None = None, starts_as: str | None = None) -> Catalog:
    """Every registered kind, or those serving ``use_case``, in ``family``, or starting as a time signal / a path."""
    return Catalog(kind for kind in _REGISTRY.values()
                   if (use_case is None or use_case in kind.use_cases)
                   and (family is None or kind.family == family)
                   and (starts_as is None or kind.starts_as == starts_as))
