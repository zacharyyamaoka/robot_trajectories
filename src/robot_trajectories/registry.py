"""The kind contract: ``@kind(...)`` on a plain function ``f(space, ..., start=None) -> JointTrajectory``.

    @kind(use_cases=("bandwidth",), catalog_id="faded_sine")
    def sine_dwell(space, joint=None, *, frequency_hz=1.0, duration_s=10.0, label="sine dwell", start=None):
        '''One sine at a fixed frequency on one joint, faded in and out.'''
        return kernels.chirp(start, space.joint_index(joint), space.room(start), space.rate_hz, ...)

Zach (Daily Note 2026-11-18): "I really like the Signal library approach... it's pure, it's functional, it feels reusable...
easier to extend because the type is the API". The decorator does, in one place, everything a caller relies on, so a kind's
body is only its motion:

1. **Refuses what the space lacks**: every ``needs`` the space does not have raises ``MissingRung`` naming the rung and the
   call that adds it. A ``better_with`` need missing runs the kind's stated fallback and says so in the notes.
2. **Resolves ``start``** (None = the space's home, a configuration, or a JointTrajectory = where it ends) and hands the body
   a configuration. The body returns kernel pieces starting there (``kernels.Arrays``, ``Piece``, a list, or a ``Motion``
   with notes), or a JointTrajectory it chained from other kinds (a higher-order kind).
3. **Joins** the pieces on one clock (a piece that does not start where the last ended gets a min-jerk transition).
4. **Guards** the output: inside the position, velocity, acceleration (and torque) limits, no jumps, at rest at both ends
   unless the kind declares ``starts="moving"`` / ``ends="moving"`` (a boundary state handed on exactly, ``chain`` checks
   it) or ``ends="anywhere"`` (event-ended), and free of collision when the space has a collision model.
5. **Records the call** in ``header.meta["recipe"]``: the name, every argument with defaults written out (a moving target by
   its data form: a ``LinearMotion`` or a ``PoseTrajectory``, never a callable), the space's fingerprint. A higher-order kind's recipe is its own one call; its segments keep the calls that make each alone.

Kinds are found by importing ``kinds/*.py`` (one file per kind) and every module an installed package names in the
entry-point group ``robot_trajectories.kinds`` (``plugins.py``). Both register the same way, and are listed the same way.

The rule kinds obey (Zach, "Robot Trajectories"): task-agnostic. A kind knows joints, poses, limits and time; it never knows
what the motion is for. No task words in a kind's name, parameters or labels (no grasp, belt, item); what it serves is a
use-case tag, nothing more.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory

from . import kernels as k
from .moving import LinearMotion, has_data_form, is_target_record, target_from_record, target_record
from .segments import RECIPE_FORMAT, TRANSITION_SPEED, Motion, Piece, _Block, assemble, guard, segment_info, with_segments
from .space import JointSpace, MissingRung, State, Status
from .time_law import TimeLaw
from .trajectory_catalog.kind import (DIMENSIONS, ENTRY_LEVEL_OF_NEED, ENTRY_LEVELS, FAMILIES, NEEDS, STARTS_AS,
                                      USE_CASE_TAGS)

LIMITS = ("joint_limits", "velocity_limits", "acceleration_limits")
STARTS = ("at_rest", "moving")
ENDS = ("at_rest", "moving", "anywhere")
COMMANDS = ("position", "effort")


@dataclass(frozen=True)
class KindInfo:
    """One registered kind: the function, and what it is for, needs and is."""

    name: str
    function: Callable[..., JointTrajectory]
    use_cases: tuple[str, ...]
    needs: tuple[str, ...]
    better_with: Mapping[str, str]  # need -> what the kind does without it (status "fallback")
    unchecked: tuple[str, ...]  # needs the catalog lists that this kind does not check when the space lacks them
    catalog_id: str | None
    family: str
    starts_as: str
    dimensions: tuple[str, ...]
    starts: str
    ends: str
    command: str
    higher_order: bool
    until: str | None
    module: str
    plugin: str | None = None  # the distribution that provided it, when it came from an entry point
    doc: str = field(default="", repr=False)

    @property
    def one_line(self) -> str:
        return " ".join(self.doc.strip().split("\n\n")[0].split())

    @property
    def entry_level(self) -> str:
        rungs = [ENTRY_LEVEL_OF_NEED[need] for need in self.needs if ENTRY_LEVEL_OF_NEED.get(need)]
        return max(rungs, key=ENTRY_LEVELS.index, default=ENTRY_LEVELS[0])

    @property
    def parameters(self) -> dict[str, Any]:
        """Every parameter after ``space`` with its default (``inspect.Parameter.empty`` when it has none)."""
        signature = inspect.signature(self.function)
        return {name: parameter.default for name, parameter in signature.parameters.items() if name != "space"}

    def call_text(self, params: Mapping[str, Any] | None = None, *, space: str = "space", brief: bool = False) -> str:
        """The call as Python: ``chirp(space, joint='can11', end_hz=8.0)``; a recipe call's values where they differ from the
        defaults, the required ones always. ``brief``: for a person, not for eval (a pose as its point in mm, floats to 4 digits)."""
        shown = []
        for name, default in self.parameters.items():
            if params is not None and name in params:
                value = _decode(params[name])
            elif default is inspect.Parameter.empty:
                shown.append(f"{name}=...")
                continue
            else:
                continue
            if default is not inspect.Parameter.empty and _encode(name, value) == _encode(name, default):
                continue
            shown.append(f"{name}={_brief(value) if brief else repr(value)}")
        return f"{self.name}({', '.join([space, *shown])})"

    def describe(self) -> str:
        """The kind's card: what it is, what it serves and needs, every parameter with its default (angles in degrees)."""
        tags = f"{self.family}, starts as a {self.starts_as.replace('_', ' ')}, {'/'.join(self.dimensions)}"
        lines = [f"{self.name}{' (catalog ' + self.catalog_id + ')' if self.catalog_id else ''}; {tags}"
                 f"{'; higher-order' if self.higher_order else ''}{'; plugin ' + self.plugin if self.plugin else ''}: {self.one_line}",
                 f"    use cases: {', '.join(self.use_cases) or '-'}    needs: {', '.join(self.needs)} "
                 f"(entry: {self.entry_level.replace('_', ' ')})"]
        for name, default in self.parameters.items():
            shown = "required" if default is inspect.Parameter.empty else _shown(name, default)
            lines.append(f"    {name:<26} {shown}")
        return "\n".join(lines)


_REGISTRY: dict[str, KindInfo] = {}


def kinds() -> dict[str, KindInfo]:
    """The registry: every kind found (in-tree and from plugins), by name, in registration order."""
    return dict(_REGISTRY)


def _row(catalog_id: str | None) -> Any:
    if catalog_id is None:
        return None
    from .trajectory_catalog.catalog import KINDS as ROWS

    if catalog_id not in ROWS:
        raise TypeError(f"catalog_id {catalog_id!r} is not a row of trajectory_catalog (add the row in catalog.py first)")
    return ROWS[catalog_id]


def kind(*, use_cases: Sequence[str], needs: Sequence[str] = LIMITS, better_with: Mapping[str, str] | None = None,
         unchecked: Sequence[str] = (), catalog_id: str | None = None, family: str | None = None, starts_as: str | None = None,
         dimensions: Sequence[str] | None = None, starts: str = "at_rest", ends: str = "at_rest", command: str = "position",
         higher_order: bool = False, until: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., JointTrajectory]]:
    """Register ``f(space, ..., start=None)`` as a kind; the decorated function returns a guarded, recorded JointTrajectory.

    ``use_cases`` and ``needs`` speak the catalog's vocabulary (``trajectory_catalog/kind.py``). ``catalog_id`` names the
    catalog row the kind implements (its family, starts_as and dimensions default to the row's). ``starts="moving"``: the
    kind continues a moving trajectory (its body is handed ``start`` as a ``State``: q, qd, qdd); ``ends="moving"``: it ends
    in motion, and whatever follows must start in that exact state; ``ends="anywhere"``: an event-ended kind (the player
    stops it on ``until``, the event in words). ``command="effort"`` marks a torque-commanded kind (its ``effort`` is the
    command, its positions the prediction)."""
    row = _row(catalog_id)
    family = family or (row.family if row else None)
    starts_as = starts_as or (row.starts_as if row else None)
    dimensions = tuple(dimensions) if dimensions is not None else (tuple(row.dimensions) if row else None)
    better_with = dict(better_with or {})
    problems = [f"use case {tag!r}" for tag in use_cases if tag not in USE_CASE_TAGS]
    problems += [f"need {need!r}" for need in (*needs, *better_with, *unchecked) if need not in NEEDS]
    if family not in FAMILIES:
        problems.append(f"family {family!r} (one of {FAMILIES}; required without a catalog_id)")
    if starts_as not in STARTS_AS:
        problems.append(f"starts_as {starts_as!r} (one of {STARTS_AS})")
    if not dimensions or any(value not in DIMENSIONS for value in dimensions):
        problems.append(f"dimensions {dimensions!r} (a run of {DIMENSIONS}; required without a catalog_id)")
    if starts not in STARTS or ends not in ENDS or command not in COMMANDS:
        problems.append(f"starts {starts!r} / ends {ends!r} / command {command!r} (one of {STARTS} / {ENDS} / {COMMANDS})")

    def decorate(body: Callable[..., Any]) -> Callable[..., JointTrajectory]:
        name = body.__name__
        if problems:
            raise TypeError(f"kind {name}: not in the catalog's vocabulary: {'; '.join(problems)}")
        signature = inspect.signature(body)
        parameters = list(signature.parameters.values())
        if not parameters or parameters[0].name != "space" or "start" not in signature.parameters \
                or signature.parameters["start"].default is not None:
            raise TypeError(f"kind {name}: the signature must be {name}(space, ..., start=None)")
        existing = _REGISTRY.get(name)
        if existing is not None and existing.module != body.__module__:
            raise TypeError(f"kind name {name!r} is taken by {existing.module}")
        if catalog_id is not None:
            for other in _REGISTRY.values():
                if other.catalog_id == catalog_id and other.name != name:
                    raise TypeError(f"{name} and {other.name} both claim catalog row {catalog_id!r}")
        has_label = "label" in signature.parameters

        @functools.wraps(body)
        def generate(space: JointSpace, *args: Any, **kwargs: Any) -> JointTrajectory:
            if not isinstance(space, JointSpace):
                raise TypeError(f"{name}(space, ...): the first argument is a JointSpace (JointSpace.from_limits(limits)), "
                                f"not {type(space).__name__}")
            info = _REGISTRY[name]
            bound = signature.bind(space, *args, **kwargs)
            bound.apply_defaults()
            for need in info.needs:
                if need not in info.better_with and need != "camera" and not space.has(need):
                    raise MissingRung(name, need)
            params = {key: _normalize(name, key, value) for key, value in bound.arguments.items() if key != "space"}
            start = space.state_at(params["start"]) if info.starts == "moving" else space.start_at(params["start"])
            call = {"kind": name, **{key: _encode(key, value) for key, value in params.items()}}
            if info.starts == "moving":
                call["start"] = start.to_config()  # WHY the whole state: a moving start is not rebuilt from a position
            label = (params.get("label") if has_label else "") or name.replace("_", " ")
            notes = [f"{name}: {info.better_with[need]} (no {need.replace('_', ' ')})" for need in info.better_with
                     if not space.has(need)]
            result = body(space, **{**params, "start": start})
            here = start.q if info.starts == "moving" else start
            if isinstance(result, JointTrajectory):
                return _composed(info, space, result, call, here, notes)
            return _primitive(info, space, result, call, here, label, notes)

        generate.kind = name  # type: ignore[attr-defined]
        _REGISTRY[name] = KindInfo(name, generate, tuple(use_cases), tuple(needs), better_with, tuple(unchecked), catalog_id,
                                   family, starts_as, dimensions, starts, ends, command, higher_order, until, body.__module__,
                                   doc=inspect.getdoc(body) or "")
        return generate

    return decorate


def _primitive(info: KindInfo, space: JointSpace, result: Any, call: dict[str, Any], start: np.ndarray, label: str,
               notes: list[str]) -> JointTrajectory:
    extra: dict[str, Any] = {}
    if isinstance(result, Motion):
        pieces, notes, extra = list(result.pieces), [*notes, *result.notes], dict(result.entry)
    elif isinstance(result, k.Arrays):
        pieces = [result]
    else:
        pieces = list(result)
    if not pieces:
        pieces = [k.hold(start, 0.0, space.rate_hz, label=label)]
    if not np.allclose(pieces[0].q[0], start, atol=1e-9):
        if info.starts == "moving":
            raise RuntimeError(f"{info.name} starts moving but its motion does not begin at its start state")
        pieces.insert(0, k.min_jerk(start, pieces[0].q[0], space.bounds, space.rate_hz, speed=TRANSITION_SPEED,
                                    label=f"transition into {label}"))
    joined, used = k.join(pieces, space.bounds, space.rate_hz, transition_speed=TRANSITION_SPEED)
    blocks = {id(piece): _Block(piece, [piece.label], np.zeros(len(piece.t), dtype=np.int64),
                                dict(getattr(piece, "channels", {}) or {}), getattr(piece, "effort", None), {})
              for piece in pieces}
    entry: dict[str, Any] = {"label": label, "call": call}
    if notes:
        entry["notes"] = list(dict.fromkeys(notes))
    for key, value, default in (("starts", info.starts, "at_rest"), ("ends", info.ends, "at_rest"),
                                ("command", info.command, "position"), ("until", info.until, None)):
        if value != default:
            entry[key] = value
    entry.update({key: value for key, value in extra.items() if key not in ("label", "call")})
    recipe = {"format": RECIPE_FORMAT, "space": space.record(), "calls": [call]}
    out = assemble(joined, used, blocks, limits=space.limits, recipe=recipe, transition_segment=None, segment_entry=entry)
    guard(joined, space.limits, info.name, rest_start=info.starts == "at_rest", rest_end=info.ends == "at_rest", effort=out.effort,
          collides=space.collision_with if space.checks_collision else None)
    return out


def _composed(info: KindInfo, space: JointSpace, result: JointTrajectory, call: dict[str, Any], start: np.ndarray,
              notes: list[str]) -> JointTrajectory:
    """A higher-order kind's output: the chain its body made, recorded as its own one call."""
    if not np.allclose(result.positions[0], start, atol=1e-9):
        raise RuntimeError(f"{info.name} does not begin at its start: a higher-order kind passes start= to its first call")
    record = json.loads(result.header.meta["recipe"])
    if record["space"]["fingerprint"] != space.fingerprint:
        raise RuntimeError(f"{info.name} chained trajectories of another space")

    def update(index: int, entry: dict[str, Any]) -> dict[str, Any]:
        entry["made_by"] = info.name
        if index == 0 and notes:
            entry["notes"] = list(dict.fromkeys([*notes, *entry.get("notes", ())]))
        return entry

    out = with_segments(result, update, recipe={"format": RECIPE_FORMAT, "space": space.record(), "calls": [call]})
    entries = segment_info(out)
    # WHY collisions only on the transitions here: every other segment was guarded by its own kind already
    transition_rows = np.isin(np.asarray(out.channels["segment"]), [i for i, entry in enumerate(entries) if entry["call"] is None])
    arrays = k.Arrays(out.t, out.positions, out.velocities, out.accelerations)
    guard(arrays, space.limits, info.name, rest_start=entries[0].get("starts", "at_rest") == "at_rest",
          rest_end=entries[-1].get("ends", "at_rest") == "at_rest", effort=out.effort,
          collides=space.collision_with if space.checks_collision else None, collision_rows=transition_rows)
    return out


# ====================================================================== parameters as text and back


def _normalize(kind_name: str, key: str, value: Any) -> Any:
    """A parameter as a kind's body reads it: tuples for sequences, Python numbers, a TimeLaw for a time-law dict."""
    if key == "start":
        if value is None or isinstance(value, (JointTrajectory, Mapping)) or type(value).__name__ == "State":
            return value
        return tuple(float(item) for item in np.asarray(value, float).reshape(-1))
    if key in ("seed", "seeds"):
        for item in np.atleast_1d(np.asarray(value, dtype=object)) if value is not None else ():
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                # WHY refused, not defaulted (V1's decision 4): an unseeded or float seed cannot be rebuilt from its recipe
                raise ValueError(f"{kind_name}: {key} must be an int (it is part of the recipe), got {value!r}")
    return _decode(value)


def _decode(value: Any) -> Any:
    if isinstance(value, Mapping) and "time_law" in value:
        return TimeLaw.from_config(value)
    if is_target_record(value):
        return target_from_record(value)  # WHY the object, not the mapping: the body is handed what the first call was handed
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_decode(item) for item in value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _encode(key: str, value: Any) -> Any:
    """A parameter as a recipe keeps it (JSON), or a refusal: a recipe must be text."""
    if isinstance(value, JointTrajectory):
        return [float(item) for item in value.positions[-1]]
    if isinstance(value, State):
        return value.to_config()
    if isinstance(value, TimeLaw):
        return value.to_config()
    if has_data_form(value):
        return target_record(value)  # a LinearMotion or a PoseTrajectory, tagged "moving_target" (moving.py)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_encode(key, item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(name): _encode(key, item) for name, item in value.items()}
    if callable(value) or (callable(getattr(value, "target", None)) and callable(getattr(value, "velocity", None))):
        # WHY refused, not pickled or skipped: a recipe must rebuild the same motion from text, and a function is code
        raise TypeError(f"parameter {key}={value!r} cannot be written as text: a function of time has no data form. A moving "
                        "target is recorded as a LinearMotion (constant velocity; LinearMotion.from_sync(pose, sync_dir, "
                        "sync_speed)) or a robot_msgs PoseTrajectory (sampled poses): pass one of those")
    raise TypeError(f"parameter {key}={value!r} cannot be written as text: pass numbers, names, lists, a TimeLaw, or a LinearMotion "
                    "or PoseTrajectory as a moving target")


def _brief(value: Any) -> str:
    """A value as a person reads it in a summary: a 4x4 pose as its point in mm, a float to 4 significant digits."""
    if isinstance(value, tuple) and len(value) == 4 and all(isinstance(row, tuple) and len(row) == 4 for row in value):
        return "pose@(" + ", ".join(f"{1000 * float(row[3]):.1f}" for row in value[:3]) + ") mm"
    if isinstance(value, LinearMotion):
        point = ", ".join(f"{1000 * float(v):.1f}" for v in value.pose[:3, 3])
        return f"LinearMotion(@({point}) mm, {value.speed:.4g} m/s along ({', '.join(f'{v:+.3f}' for v in value.direction)}))"
    if isinstance(value, PoseTrajectory):
        return f"PoseTrajectory({len(value.t)} poses, {float(value.t[0]):.3f}..{float(value.t[-1]):.3f} s)"
    if isinstance(value, tuple):
        return "(" + ", ".join(_brief(item) for item in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, float):
        return f"{value:.4g}"
    return repr(value)


def _shown(name: str, value: Any) -> str:
    """A default as a person reads it: radians in degrees."""
    if name.endswith("_rad") and isinstance(value, (int, float)):
        return f"{math.degrees(value):.4g}°"
    return repr(value)


# ====================================================================== the catalog: what generates, and what does not yet


@dataclass(frozen=True)
class Entry:
    """One line of ``catalog()``: a kind that generates, or a catalog row nothing implements yet."""

    name: str
    catalog_id: str | None
    implemented: bool
    family: str
    starts_as: str
    dimensions: tuple[str, ...]
    entry_level: str
    use_cases: tuple[str, ...]
    one_line: str
    status: Status | None = None  # on the space given
    higher_order: bool = False
    plugin: str | None = None


class Catalog(tuple):
    """``catalog(...)``'s answer. ``print()`` it; ``.names``, ``.implemented``, ``.todo``; ``["chirp"]`` one entry."""

    def __new__(cls, entries: Sequence[Entry], heading: str = "") -> Catalog:
        catalog = super().__new__(cls, entries)
        catalog.heading = heading
        return catalog

    @property
    def names(self) -> list[str]:
        return [entry.name for entry in self]

    @property
    def implemented(self) -> Catalog:
        return Catalog([entry for entry in self if entry.implemented], self.heading)

    @property
    def todo(self) -> Catalog:
        return Catalog([entry for entry in self if not entry.implemented], self.heading)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            for entry in self:
                if key in (entry.name, entry.catalog_id):
                    return entry
            raise KeyError(f"no kind {key!r} here; kinds are {self.names}")
        return super().__getitem__(key)

    def __str__(self) -> str:
        width = max((len(entry.name) for entry in self), default=4)
        lines = [self.heading]
        for entry in self:
            made = ("higher-order" if entry.higher_order else "generates") if entry.implemented else "catalogued"
            state = f"{entry.status.state:<10}" if entry.status else ""
            note = ""
            if entry.status and entry.status.state != "available":
                note = f"{entry.status.why} -> {entry.status.fix}"
            elif entry.implemented and entry.catalog_id and entry.catalog_id != entry.name:
                note = f"catalog row {entry.catalog_id}"
            lines.append(f"  {entry.name:<{width}}  {made:<12} {state}{entry.entry_level.replace('_', ' '):<13} "
                         f"{entry.starts_as.replace('_', ' '):<12}{'/'.join(entry.dimensions):<10} {note}".rstrip())
        return "\n".join(lines)


def catalog(use_case: str | None = None, *, space: JointSpace | None = None, family: str | None = None,
            starts_as: str | None = None) -> Catalog:
    """The kinds that generate (in-tree and plugins) and, after them, the catalog rows nothing implements yet, honestly
    marked. ``use_case`` / ``family`` / ``starts_as`` filter; ``space`` grades each: available, fallback or locked (and the fix)."""
    from .trajectory_catalog.catalog import CATALOG as ROWS

    if use_case is not None and use_case not in USE_CASE_TAGS:
        raise ValueError(f"no use case {use_case!r}; the vocabulary is {USE_CASE_TAGS}")
    entries: list[Entry] = []
    for info in _REGISTRY.values():
        status = None if space is None else space.status(info.needs, better_with=info.better_with, unchecked=info.unchecked)
        entries.append(Entry(info.name, info.catalog_id, True, info.family, info.starts_as, info.dimensions, info.entry_level,
                             info.use_cases, info.one_line, status, info.higher_order, info.plugin))
    claimed = {info.catalog_id for info in _REGISTRY.values()}
    for row in ROWS:
        if row.id in claimed:
            continue
        status = None if space is None else space.status(row.needs, reach_from_kinematics=True)
        entries.append(Entry(row.id, row.id, False, row.family, row.starts_as, tuple(row.dimensions), row.entry_level,
                             tuple(row.use_cases), row.one_line, status))
    picked = [entry for entry in entries if (use_case is None or use_case in entry.use_cases)
              and (family is None or entry.family == family) and (starts_as is None or entry.starts_as == starts_as)]
    made = sum(entry.implemented for entry in picked)
    heading = (f"{made} kinds generate{' for ' + use_case if use_case else ''}, {len(picked) - made} catalogued, not yet"
               + (f"; on {space.source}" if space is not None else ""))
    if space is not None:
        counts = {state: sum(entry.implemented and entry.status.state == state for entry in picked)
                  for state in ("available", "fallback", "locked")}
        heading += " (" + ", ".join(f"{count} {state}" for state, count in counts.items()) + " of those that generate)"
    return Catalog(picked, heading)


def _mark_plugin(names: Sequence[str], provider: str) -> None:
    """Record which distribution provided these kinds (``plugins.load`` calls it)."""
    for name in names:
        info = _REGISTRY[name]
        _REGISTRY[name] = KindInfo(**{**info.__dict__, "plugin": provider})


__all__ = ["Catalog", "Entry", "KindInfo", "Motion", "Piece", "catalog", "kind", "kinds"]
