"""Workspace (from V4, on Zach's input ladder): what the generator knows about the robot, one rung at a time.

    Workspace.from_limits(limits)                                    rung 1: joint limits (every "now" kind runs here)
    Workspace.from_description(description, max_acceleration=6.0)    rung 1 from a URDF, its missing acceleration supplied
    workspace.within(box)                                            positions tightened (a demo box above the table)
    workspace.with_kinematics(kinematics)                            rung 2: FK / IK (a slot: it gates kinds, generates nothing yet)
    workspace.with_reach_map(reach_map)                              rung 2's optional cache: the reachable set, precomputed
    workspace.with_dynamics(model)                                   rung 3: a model that predicts torque (a slot too)
    print(workspace)                                                 each rung known or missing, and what that costs

Zach's refined spec (2026-11-18): "joint limits are enough ... pass in like a FK/IK function ... the reach map is too
coarse". So the ladder is joint limits -> + kinematics -> + dynamics (-> + torque limits), and a reach map is a cache of
the kinematics rung, not an entry point. WHY the slots exist before anything uses them: the first release needs no
Cartesian path (every "now" kind runs on rung 1), but the API shape must not change when the five-bar's snake paths
move in; a filled slot already unlocks the catalog rows that need it, and an empty one names the call that fills it.

The leaf rule holds: a description is anything with ``joint_limits()``, kinematics anything shaped like
``bam_reach.Kinematics`` (``forward``, ``inverse``, ``joint_names``, ``cache_key``), a dynamics model anything with
``pin_utils.PinDynamics.inverse_dynamics(q, velocity, acceleration)``, a reach map anything with ``joint_solutions`` and
``valid_mask``. Nothing here imports them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np
from robot_msgs.trajectory import JointLimits

from . import kernels as k

from .kind import ENTRY_LEVEL_OF_NEED, ENTRY_LEVELS, Kind, ReachSet, catalog, entry_level


class MissingInput(ValueError):
    """Something the workspace needs is not in what was passed. The message names the call that fixes it."""


class DescriptionLike(Protocol):
    """bam_robot_descriptions.RobotDescription, as far as the generator reads it."""

    def joint_limits(self, names: Sequence[str] | None = None) -> JointLimits: ...


class KinematicsLike(Protocol):
    """bam_reach.kinematics.Kinematics (kinematics.py:107), as far as the generator will read it. Never imported."""

    joint_names: tuple[str, ...]
    cache_key: str | None

    def forward(self, joint_positions: np.ndarray) -> Any: ...

    def inverse(self, target_pose: np.ndarray, seed: Any = None) -> Any: ...


class DynamicsLike(Protocol):
    """pin_utils.PinDynamics (dynamics.py:58), as far as the generator will read it."""

    def inverse_dynamics(self, q: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray) -> np.ndarray: ...


class ReachMapLike(Protocol):
    """bam_reach.ReachMap's fields the generator reads."""

    joint_names: tuple[str, ...]
    joint_solutions: np.ndarray  # (P, O, J)
    valid_mask: np.ndarray  # (P, O)


# ------------------------------------------------------------------ what a need costs, in words

RUNG_TITLES = {"joint_limits": "joint limits", "kinematics": "kinematics", "dynamics": "dynamics", "torque_limits": "torque limits"}
# WHY not on the ladder: the catalog's own words. A collision check rides on the kinematics rung but a 1-DOF rig has nothing
# to collide with, and a camera measures the output; neither is a generator input.
UNCHECKED_NEEDS = ("collision_model",)
NOT_INPUTS = ("camera",)


@dataclass(frozen=True)
class Status:
    """Whether something runs on a workspace: available, fallback (runs, a stated way) or locked (and the fix)."""

    state: str  # "available" | "fallback" | "locked"
    missing: tuple[str, ...] = ()
    why: str = ""
    fix: str = ""
    unchecked: tuple[str, ...] = ()

    def __str__(self) -> str:
        text = self.state
        if self.why:
            text += f": {self.why}"
        if self.fix:
            text += f" ({self.fix})"
        if self.unchecked:
            text += f" [not checked here: {', '.join(n.replace('_', ' ') for n in self.unchecked)}]"
        return text


def _number(value: float) -> str:
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _deg(values: np.ndarray, unit: str = "°") -> str:
    """Radians as degrees for a person: one number when every joint agrees, else a list."""
    texts = [_number(math.degrees(float(value))) for value in np.atleast_1d(values)]
    return (texts[0] if len(set(texts)) == 1 else "[" + ", ".join(texts) + "]") + unit


def _ranges(low: np.ndarray, high: np.ndarray) -> str:
    pairs = [f"{_number(math.degrees(a))}..{_number(math.degrees(b))}" for a, b in zip(low, high)]
    return (pairs[0] if len(set(pairs)) == 1 else ", ".join(pairs)) + "°"


def _finite(values: Any) -> bool:
    return values is not None and bool(np.all(np.isfinite(np.asarray(values, float))))


def limits_to_json(limits: JointLimits) -> dict[str, Any]:
    out: dict[str, Any] = {"joint_names": list(limits.joint_names)}
    for bound in ("min_position", "max_position", "max_velocity", "max_acceleration", "max_jerk", "max_effort"):
        values = getattr(limits, bound)
        if values is not None:
            out[bound] = [float(value) for value in values]
    if limits.position_kind is not None:
        out["position_kind"] = list(limits.position_kind)
    return out


def limits_from_json(values: Mapping[str, Any]) -> JointLimits:
    arrays = {key: np.array(value, float) for key, value in values.items() if key not in ("joint_names", "position_kind")}
    kind = values.get("position_kind")
    return JointLimits(joint_names=tuple(values["joint_names"]), position_kind=None if kind is None else tuple(kind), **arrays)


def _per_joint(value: float | Sequence[float], count: int) -> np.ndarray:
    array = np.asarray(value, float)
    return np.full(count, float(array)) if array.ndim == 0 else array


# ------------------------------------------------------------------ the workspace


@dataclass(frozen=True, eq=False)
class Workspace:
    """The joints a generator may move, at the rung of knowledge it was built with. Build it with a ``from_*`` method."""

    limits: JointLimits  # the effective limits: every trajectory is generated inside these, and carries them
    built_from: str  # "joint limits", "the description 'ur5e'", ... (+ what was added)
    home: np.ndarray
    named_home: bool = False  # True when the caller named it: the catalog's ``home_pose``
    name: str = ""
    kinematics: Any = None
    dynamics: Any = None
    reach: ReachSet | None = None
    reach_name: str = ""
    assumptions: tuple[str, ...] = ()  # every value supplied rather than read, in words (logged in the config)

    # ------------------------------------------------------------------ rung 1

    @classmethod
    def from_limits(cls, limits: JointLimits, *, home: Sequence[float] | None = None,
                    max_acceleration: float | Sequence[float] | None = None, name: str = "",
                    built_from: str = "joint limits") -> Workspace:
        """Bare robot_msgs JointLimits (positions, velocities, accelerations), e.g. a rig's schedule values."""
        limits, assumptions = _with_acceleration(limits, max_acceleration, where=name or "these joint limits",
                                                 fix="Workspace.from_limits(limits, max_acceleration=<rad/s²>)")
        for bound, fix in (("min_position", "a finite position range"), ("max_position", "a finite position range"),
                           ("max_velocity", "max_velocity")):
            if not _finite(getattr(limits, bound)):
                raise MissingInput(f"{name or 'these joint limits'}: {bound} must be finite to generate; state {fix} "
                                   "(a continuous joint needs .within((low, high)))")
        return cls._build(limits, built_from=built_from, home=home, name=name, assumptions=assumptions)

    @classmethod
    def from_description(cls, description: DescriptionLike, *, max_acceleration: float | Sequence[float] | None = None,
                         home: Sequence[float] | None = None, joints: Sequence[str] | None = None) -> Workspace:
        """A robot description's limits (anything with ``joint_limits()``).

        WHY it refuses without ``max_acceleration``: a URDF has position and velocity limits but never an acceleration
        limit, every kind needs one, and the generator will not guess the value the drives can actually do."""
        name = str(getattr(description, "name", "") or "")
        limits = description.joint_limits(None if joints is None else tuple(joints))
        where = f"{name}'s URDF" if name else "the description's URDF"
        limits, assumptions = _with_acceleration(limits, max_acceleration, where=where,
                                                 fix="Workspace.from_description(description, max_acceleration=<rad/s²>)")
        built_from = f"the description {name!r}" if name else "a description"
        if not (_finite(limits.min_position) and _finite(limits.max_position)):
            raise MissingInput(f"{where} has a continuous joint: tighten it with "
                               "Workspace.from_description(...).within((low, high))")
        return cls._build(limits, built_from=built_from, home=home, name=name, assumptions=assumptions)

    @classmethod
    def from_reach_map(cls, reach_map: ReachMapLike, limits: JointLimits, *, home: Sequence[float] | None = None) -> Workspace:
        """Shorthand for ``Workspace.from_limits(limits).with_reach_map(reach_map)``. WHY limits travel with the map: a
        reach map says where the robot can be, never how fast it may go."""
        names = tuple(reach_map.joint_names)
        return cls.from_limits(limits.select(names)).with_reach_map(reach_map, home=home)

    @classmethod
    def _build(cls, limits: JointLimits, *, built_from: str, home: Any, name: str, assumptions: tuple[str, ...],
               reach: ReachSet | None = None, reach_name: str = "", kinematics: Any = None, dynamics: Any = None) -> Workspace:
        bounds = k.Bounds.from_limits(limits)
        named = home is not None
        if home is None:
            home = bounds.centre
            if reach is not None:
                # WHY nearest the middle (V2): a stable, explainable default; the scenario pins the five-bar's anyway
                reachable = reach.reachable
                middle = reachable.mean(axis=0)
                home = reachable[int(np.argmin(np.max(np.abs(reachable - middle), axis=1)))]
        home = np.asarray(home, float)
        if home.shape != (len(limits.joint_names),):
            raise ValueError(f"home has shape {home.shape}, expected ({len(limits.joint_names)},)")
        if np.any(home < bounds.low - 1e-12) or np.any(home > bounds.high + 1e-12):
            raise ValueError(f"home {np.round(np.degrees(home), 2).tolist()}° is outside the position limits")
        if reach is not None and not reach.nearest_is_reachable(home):
            raise ValueError(f"home {np.round(np.degrees(home), 2).tolist()}° is not in the reach map's reachable set")
        return cls(limits=limits, built_from=built_from, home=home, named_home=named, name=name, kinematics=kinematics,
                   dynamics=dynamics, reach=reach, reach_name=reach_name, assumptions=tuple(assumptions))

    # ------------------------------------------------------------------ tightening and the next rungs

    def within(self, box: JointLimits | tuple[Sequence[float], Sequence[float]]) -> Workspace:
        """Positions (and any speeds the box states) tightened to ``box``, a JointLimits or a ``(low, high)`` pair; never loosened."""
        limits = self.limits
        changes: dict[str, Any] = {}
        if isinstance(box, JointLimits):
            tighter = box.select(limits.joint_names)
            for bound, pick in (("min_position", np.maximum), ("max_position", np.minimum), ("max_velocity", np.minimum),
                                ("max_acceleration", np.minimum)):
                value, current = getattr(tighter, bound), getattr(limits, bound)
                if value is not None:
                    changes[bound] = value if current is None else pick(current, value)
            if tighter.position_kind is not None:
                changes["position_kind"] = tighter.position_kind
        else:
            low, high = (np.asarray(value, float) for value in box)
            changes["min_position"] = low if limits.min_position is None else np.maximum(limits.min_position, low)
            changes["max_position"] = high if limits.max_position is None else np.minimum(limits.max_position, high)
        tightened = limits.replace(**changes)
        note = f"positions boxed to {_deg(tightened.min_position)} .. {_deg(tightened.max_position)}"
        return Workspace._build(tightened, built_from=self.built_from, home=self.home if self.named_home else None,
                                name=self.name, assumptions=(*self.assumptions, note), reach=self.reach,
                                reach_name=self.reach_name, kinematics=self.kinematics, dynamics=self.dynamics)

    def with_kinematics(self, kinematics: KinematicsLike) -> Workspace:
        """Rung 2: forward and inverse kinematics. Duck-typed on bam_reach's ``Kinematics``; it gates kinds (the catalog's
        FK/IK rows show as ready), and nothing generates from it yet."""
        missing = [name for name in ("forward", "inverse") if not callable(getattr(kinematics, name, None))]
        missing += [name for name in ("joint_names", "cache_key") if not hasattr(kinematics, name)]
        if missing:
            raise TypeError(f"kinematics needs {missing}: the shape of bam_reach.kinematics.Kinematics "
                            "(forward(q), inverse(pose, seed), joint_names, cache_key)")
        if tuple(kinematics.joint_names) != tuple(self.limits.joint_names):
            raise ValueError(f"the kinematics' joints {tuple(kinematics.joint_names)} are not this workspace's {self.limits.joint_names}")
        return replace(self, kinematics=kinematics, built_from=f"{self.built_from} + kinematics")

    def with_dynamics(self, model: DynamicsLike) -> Workspace:
        """Rung 3: a model that predicts torque (``inverse_dynamics(q, velocity, acceleration)``, pin_utils' shape). A slot:
        it unlocks the catalog's optimised-excitation and torque-feasibility rows; nothing here generates from it yet."""
        if not callable(getattr(model, "inverse_dynamics", None)):
            raise TypeError("a dynamics model needs inverse_dynamics(q, velocity, acceleration) (pin_utils.PinDynamics' shape)")
        return replace(self, dynamics=model, built_from=f"{self.built_from} + dynamics")

    def with_reach_map(self, reach_map: ReachMapLike, *, home: Sequence[float] | None = None) -> Workspace:
        """Rung 2's optional cache: the reachable set, precomputed. Every centred kind then stays inside the largest box
        around where it starts that holds no unreachable configuration, and ``WorkspaceSweep`` tours the set itself."""
        names = tuple(reach_map.joint_names)
        if names != tuple(self.limits.joint_names):
            raise ValueError(f"the reach map's joints {names} are not this workspace's {self.limits.joint_names}")
        reach = ReachSet.from_reach_map(reach_map)
        if home is None and self.named_home:
            home = self.home
        workspace = Workspace._build(self.limits, built_from=f"{self.built_from} + a reach map", home=home, name=self.name,
                                     assumptions=self.assumptions, reach=reach, reach_name=str(getattr(reach_map, "kinematics_name", "")),
                                     kinematics=self.kinematics, dynamics=self.dynamics)
        if home is None:
            return replace(workspace, assumptions=(*workspace.assumptions, "home: the reachable configuration nearest the middle "
                                                   f"of the reach map, {_deg(workspace.home)}"))
        return workspace

    # ------------------------------------------------------------------ what it knows

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.limits.joint_names)

    @property
    def bounds(self) -> k.Bounds:
        return k.Bounds.from_limits(self.limits)

    def room_around(self, q: np.ndarray) -> k.Bounds:
        """The hole-free box around ``q`` (the five proposals' shared "room around here")."""
        if self.reach is None:
            return self.bounds
        return k.inscribed_box(q, self.reach.unreachable, self.bounds)

    def has(self, need: str) -> bool:
        """Whether this workspace knows ``need`` (a word of the catalog's needs vocabulary)."""
        limits = self.limits
        if need == "joint_limits":
            return _finite(limits.min_position) and _finite(limits.max_position)
        if need in ("velocity_limits", "acceleration_limits", "jerk_limits"):
            values = getattr(limits, "max_" + need.split("_")[0])
            return _finite(values) and bool(np.all(np.asarray(values) > 0))
        if need == "torque_limits":
            return _finite(limits.max_effort)
        if need == "home_pose":
            return self.named_home
        if need == "reach_map":
            return self.reach is not None
        if need in ("forward_kinematics", "inverse_kinematics"):
            return self.kinematics is not None
        if need == "dynamics_model":
            return self.dynamics is not None
        if need in UNCHECKED_NEEDS or need in NOT_INPUTS:
            return False
        raise ValueError(f"unknown need {need!r}; the catalog's needs are {tuple(ENTRY_LEVEL_OF_NEED)}")

    def gap(self, need: str) -> tuple[str, str]:
        """(what is missing, the call that adds it), in words."""
        return {
            "joint_limits": ("no finite position range", "Workspace.from_limits(...) with min/max_position, or .within((low, high))"),
            "velocity_limits": ("no velocity limit", "state max_velocity in the JointLimits"),
            "acceleration_limits": ("no acceleration limit (a URDF carries none)", "pass max_acceleration= to Workspace.from_*"),
            "jerk_limits": ("no jerk limit", "state max_jerk in the JointLimits"),
            "torque_limits": ("no torque limit", "state max_effort in the JointLimits (a URDF's <limit effort=...> has it)"),
            "home_pose": ("no named home", "pass home=READY to Workspace.from_*(...)"),
            "reach_map": ("no reach map", "workspace.with_reach_map(reach_map), or sample it from with_kinematics once that is built"),
            "forward_kinematics": ("no kinematics", "workspace.with_kinematics(kinematics), bam_reach's Kinematics shape"),
            "inverse_kinematics": ("no kinematics", "workspace.with_kinematics(kinematics), bam_reach's Kinematics shape"),
            "dynamics_model": ("no dynamics model", "workspace.with_dynamics(model), pin_utils.PinDynamics' shape"),
            "collision_model": ("no collision model", "not a slot yet: a collision check rides on the kinematics rung"),
            "camera": ("no camera", "a camera measures the output; the generator never needs one"),
        }[need]

    def status(self, needs: Iterable[str], *, without: Mapping[str, str] | None = None, unchecked: Iterable[str] = (),
               reach_from_kinematics: bool = True) -> Status:
        """Grade anything that ``needs`` these inputs here: a registered kind, or a catalog row nothing generates yet.

        WHY ``reach_from_kinematics``: on Zach's ladder a reach map is a cache of the kinematics rung, so for a catalog row
        (is the input there?) kinematics covers it. A registered kind's code reads the actual reachable set, which nothing
        builds from the kinematics slot yet, so ``kind_status`` asks for the map itself."""
        without = dict(without or {})
        needs = [need for need in needs if need not in NOT_INPUTS]
        skipped = tuple(dict.fromkeys([need for need in needs if need in UNCHECKED_NEEDS] + list(unchecked)))

        def known(need: str) -> bool:
            return self.has(need) or (need == "reach_map" and reach_from_kinematics and self.kinematics is not None)

        missing = [need for need in needs if need not in UNCHECKED_NEEDS and not known(need)]
        hard = [need for need in missing if need not in without]
        if hard:
            # WHY: with no kinematics, a reach map's own gap repeats it (the map is the kinematics rung's cache)
            shown = [need for need in hard if not (need == "reach_map" and reach_from_kinematics and
                                                   {"forward_kinematics", "inverse_kinematics"} & set(hard))]
            gaps = [self.gap(need) for need in shown]
            return Status("locked", tuple(hard), "; ".join(dict.fromkeys(why for why, _ in gaps)),
                          "; ".join(dict.fromkeys(fix for _, fix in gaps)), skipped)
        if missing:
            why, fix = self.gap(missing[0])
            return Status("fallback", tuple(missing), f"{without[missing[0]]} ({why})", fix, skipped)
        return Status("available", (), "", "", skipped)

    def row_status(self, row: Mapping[str, Any]) -> Status:
        """A catalog.json row's status here: the record that claims it, if one does (its code's needs), else the row's own needs.
        WHY one rule: the strip, ``describe()`` and ``generator.kinds()`` then never disagree about the same row."""
        for kind in catalog():
            if kind.catalog_id == row["id"]:
                return self.kind_status(kind)
        return self.status(row["needs"])

    def kind_status(self, kind: type[Kind]) -> Status:
        status = self.status(kind.needs, without=kind.without, unchecked=kind.unchecked, reach_from_kinematics=False)
        if status.state == "fallback" and "reach_map" in status.missing and self.kinematics is not None:
            return Status("fallback", status.missing, f"{kind.without['reach_map']} (the kinematics is known, but sampling the "
                          "reachable set from it is not built yet)", "workspace.with_reach_map(reach_map)", status.unchecked)
        return status

    # ------------------------------------------------------------------ as configuration

    def to_config(self) -> dict[str, Any]:
        """What decides the motion, as JSON-ready data (a reach map, kinematics or dynamics only by fingerprint: they are
        data, not configuration, so a rebuild is handed the same ones and refuses others)."""
        return {
            "built_from": self.built_from, "name": self.name, "limits": limits_to_json(self.limits),
            "home": [float(value) for value in self.home], "named_home": self.named_home, "assumptions": list(self.assumptions),
            "reach_map": None if self.reach is None else {"fingerprint": self.reach.fingerprint(), "configurations": int(len(self.reach.configurations))},
            "kinematics": None if self.kinematics is None else {"cache_key": getattr(self.kinematics, "cache_key", None),
                                                               "name": str(getattr(self.kinematics, "name", type(self.kinematics).__name__))},
            "dynamics": None if self.dynamics is None else {"type": type(self.dynamics).__name__},
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, reach_map: ReachMapLike | None = None, kinematics: Any = None,
                    dynamics: Any = None) -> Workspace:
        """The workspace a ``to_config()`` dict describes; a reach map, kinematics or dynamics must be handed back."""
        reach = None
        if config.get("reach_map"):
            if reach_map is None:
                raise MissingInput(f"this was generated on a reach map (fingerprint {config['reach_map']['fingerprint']}): "
                                   "pass reach_map=<that map>, or a generator built on it")
            reach = ReachSet.from_reach_map(reach_map)
            if reach.fingerprint() != config["reach_map"]["fingerprint"]:
                raise ValueError(f"reach map fingerprint {reach.fingerprint()} is not the one this was made on "
                                 f"({config['reach_map']['fingerprint']})")
        for slot, given in (("kinematics", kinematics), ("dynamics", dynamics)):
            if config.get(slot) and given is None:
                raise MissingInput(f"this was generated with {slot} ({config[slot]}): pass {slot}=<the same model>")
        workspace = cls(limits=limits_from_json(config["limits"]), built_from=config["built_from"], home=np.asarray(config["home"], float),
                        named_home=bool(config.get("named_home")), name=config.get("name", ""), kinematics=kinematics,
                        dynamics=dynamics, reach=reach, assumptions=tuple(config.get("assumptions", ())))
        return workspace

    @property
    def fingerprint(self) -> str:
        """16 hex characters over every limit, the home, and the fingerprints of the reach map / kinematics / dynamics."""
        return hashlib.sha256(json.dumps(self.to_config(), sort_keys=True).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ printing: each rung known or missing, and the cost

    def describe(self, catalog_rows: Sequence[Mapping[str, Any]] | None = None) -> str:
        """The rungs, what each costs in this package's kinds, and (given ``catalog_rows``, catalog.json's kinds) in the
        full catalog's. Angles in degrees."""
        kinds = list(catalog())
        statuses = {kind.id: self.kind_status(kind) for kind in kinds}
        names = self.joint_names
        joints = ", ".join(names) if len(names) <= 3 else f"{names[0]} .. {names[-1]}"
        lines = [f"Workspace of {len(names)} joint{'s' * (len(names) != 1)} ({joints}), built from {self.built_from}"]
        limits = self.limits
        known = [f"position {_ranges(limits.min_position, limits.max_position)}", f"velocity {_deg(limits.max_velocity, '°/s')}",
                 f"acceleration {_deg(limits.max_acceleration, '°/s²')}"]
        known.append(f"jerk {_deg(limits.max_jerk, '°/s³')}" if _finite(limits.max_jerk) else "no jerk limit")

        def cost(rung: str) -> str:
            ours = [kind for kind in kinds if kind.entry_level() == rung]
            fallback = [kind.id for kind in kinds if statuses[kind.id].state == "fallback" and
                        any(ENTRY_LEVEL_OF_NEED.get(need) == rung for need in statuses[kind.id].missing)]
            text = f"{len(ours)} of this package's kinds need it"
            if fallback:
                text += f"; {', '.join(fallback)} falls back without it"
            if catalog_rows is not None:
                rows = [row for row in catalog_rows if row.get("entry_level") == rung]
                text += f"; {len(rows)} of the catalog's {len(catalog_rows)} kinds start on this rung"
            return text

        lines.append(f"  1 joint limits   known    {'; '.join(known)}")
        rung_have = {"kinematics": self.kinematics is not None, "dynamics": self.dynamics is not None, "torque_limits": self.has("torque_limits")}
        fixes = {"kinematics": "with_kinematics(kinematics)", "dynamics": "with_dynamics(model)", "torque_limits": "state max_effort in the limits"}
        for index, rung in enumerate(ENTRY_LEVELS[1:], start=2):
            title = RUNG_TITLES[rung]
            if rung_have[rung]:
                detail = {"kinematics": f"{getattr(self.kinematics, 'name', 'kinematics')} (cache_key {getattr(self.kinematics, 'cache_key', None)})",
                          "dynamics": type(self.dynamics).__name__, "torque_limits": f"effort {np.round(limits.max_effort, 2).tolist()} N·m"}[rung]
                lines.append(f"  {index} {title:<14} known    {detail}")
            else:
                lines.append(f"  {index} {title:<14} missing  {fixes[rung]} · {cost(rung)}")
            if rung == "kinematics":
                if self.reach is not None:
                    lines.append(f"    reach map        known    {int(self.reach.valid.sum())} of {len(self.reach.configurations)} configurations "
                                 f"reachable on a {_deg(self.reach.grid_step)} grid{' (' + self.reach_name + ')' if self.reach_name else ''}: "
                                 "every centred kind stays inside the hole-free box around its start")
                else:
                    fallback = [kind.id for kind in kinds if "reach_map" in statuses[kind.id].missing]
                    lines.append(f"    reach map        missing  an optional cache of rung 2 · with_reach_map(map)"
                                 + (f" · without it {', '.join(fallback)} falls back to a box grid (every configuration in the box assumed reachable)" if fallback else ""))
        where = "named" if self.named_home else ("the box centre" if self.reach is None else "nearest the middle of the reachable set")
        lines.append(f"  home             {_deg(self.home)} ({where})")
        for assumption in self.assumptions:
            lines.append(f"  supplied         {assumption}")
        counts = {state: sum(status.state == state for status in statuses.values()) for state in ("available", "fallback", "locked")}
        lines.append(f"  this package's {len(kinds)} kinds here: " + ", ".join(f"{count} {state}" for state, count in counts.items()))
        if catalog_rows is not None:
            graded = [(row, self.row_status(row)) for row in catalog_rows]
            locked = [row for row, status in graded if status.state == "locked"]
            higher = sum(row["entry_level"] != "joint_limits" for row in locked)
            lines.append(f"  the catalog's {len(catalog_rows)} kinds here: {len(catalog_rows) - len(locked)} have every input, "
                         f"{len(locked)} are locked ({higher} start on a higher rung, {len(locked) - higher} need a rung-1 input "
                         "this workspace lacks: a jerk limit or a named home)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()

    def __repr__(self) -> str:
        return f"Workspace({len(self.joint_names)} joints, {self.built_from}, {self.fingerprint})"


def _with_acceleration(limits: JointLimits, supplied: float | Sequence[float] | None, *, where: str,
                       fix: str) -> tuple[JointLimits, tuple[str, ...]]:
    """Fill a missing acceleration limit with ``supplied`` (or tighten a stated one), or refuse naming ``fix``."""
    count = len(limits.joint_names)
    stated = limits.max_acceleration
    if supplied is None:
        if not _finite(stated):
            raise MissingInput(f"{where} states no max_acceleration, and every kind needs one: {fix}, the value the drives "
                               "can actually do (a URDF never carries one, and the generator will not guess)")
        return limits, ()
    value = _per_joint(supplied, count)
    if not _finite(stated):
        filled = value if stated is None else np.where(np.isfinite(stated), stated, value)
        return limits.replace(max_acceleration=filled), (f"max_acceleration {_deg(value, '°/s²')} supplied by the caller ({where} has none)",)
    tightened = np.minimum(stated, value)
    return limits.replace(max_acceleration=tightened), (f"max_acceleration tightened to {_deg(tightened, '°/s²')}",)


__all__ = ["DescriptionLike", "DynamicsLike", "KinematicsLike", "MissingInput", "ReachMapLike", "Status", "Workspace",
           "entry_level", "limits_from_json", "limits_to_json"]
