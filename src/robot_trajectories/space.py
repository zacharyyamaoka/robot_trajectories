"""JointSpace: the robot, as one frozen value every kind reads. The physical manifestation is injected, never imported.

    space = JointSpace.from_limits(limits)                                   rung 1: joint limits (every joint-space kind)
    space = JointSpace.from_description(description, max_acceleration=6.0)   rung 1 from a URDF, its missing acceleration stated
    space.within(box)                                                        positions tightened (a demo box above the table)
    space.with_kinematics(kinematics)                                        rung 2: forward(q) -> 4x4, inverse(pose, seed) -> q | None
    space.with_reach_map(reach_map)                                          rung 2's optional cache: the reachable set, precomputed
    space.with_dynamics(model)                                               rung 3: torque(q, qd, qdd), optionally forward(q, qd, tau)
    space.with_collision(scene), space.with_self_collision(model)            beside the ladder: in_collision(q) -> bool

Zach (Daily Note 2026-11-18, "Robot Trajectories"): "robot trajectories kinda exist completely independent from any actual
physical manifestation or task. The actual physical manifestation eventually you can inject ... via ik/fk, dynamics model,
collision model ... at this level we never really know about like what the semantic meaning is of things. that is why it
works for like any robot type, or any task." So this module is the only place a robot enters, and it enters by shape: a
description is anything with ``joint_limits()``, kinematics anything with ``forward`` / ``inverse`` (bam_reach's
``Kinematics`` protocol, kinematics.py:107, results coerced from its FKResult / IKResult too), dynamics anything with
``torque`` (or pin_utils' ``inverse_dynamics``), a collision checker anything with ``in_collision``, a reach map anything
with ``joint_solutions`` / ``valid_mask``. Nothing here imports them (the leaf rule; tests/test_boundary.py).

A missing input is a ``MissingRung`` that names the input, the rung it sits on and the call that adds it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, NamedTuple, Protocol

import numpy as np
from robot_msgs.trajectory import JointLimits, JointTrajectory

from . import kernels as k
# WHY the catalog's vocabulary: the kinds that generate and the catalog rows they claim grade needs by one rule.
from .trajectory_catalog.kind import ENTRY_LEVEL_OF_NEED, ENTRY_LEVELS

DEFAULT_RATE_HZ = k.DEFAULT_RATE_HZ
SPACE_FORMAT = "robot_trajectories.space/1"


class State(NamedTuple):
    """A joint state: positions, velocities, accelerations (each (J,)). A kind that declares ``starts="moving"`` is handed
    its start as a State (from the JointTrajectory it continues), and a time law can end in one."""

    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray

    @classmethod
    def rest(cls, q: Any) -> State:
        q = np.array(q, float)
        return cls(q, np.zeros_like(q), np.zeros_like(q))

    @property
    def at_rest(self) -> bool:
        return not (np.any(self.qd != 0.0) or np.any(self.qdd != 0.0))

    def to_config(self) -> dict[str, list[float]]:
        return {"q": [float(v) for v in self.q], "qd": [float(v) for v in self.qd], "qdd": [float(v) for v in self.qdd]}


# ====================================================================== the injected shapes (Protocols, never imports)


class Kinematics(Protocol):
    """Forward and inverse kinematics of the tool, in the robot's base frame (bam_reach.kinematics.Kinematics' shape).

    Optional attributes the space reads: ``cache_key`` (a stable name for the model, fingerprinted into recipes), ``name``
    and ``orientation_constraint`` ("none" for a position-only robot such as a planar five-bar, else "full")."""

    joint_names: tuple[str, ...]

    def forward(self, joint_positions: np.ndarray) -> Any: ...  # (4, 4) tool pose; (3,) xyz; or an FKResult

    def inverse(self, target_pose: np.ndarray, seed: Any = None) -> Any: ...  # q, None, or an IKResult


class Dynamics(Protocol):
    """A model that predicts joint torque: ``torque(q, qd, qdd)``; optionally ``forward(q, qd, tau) -> qdd`` to simulate the
    positions an effort-commanded kind produces (``space.predict``)."""

    def torque(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray: ...


class CollisionChecker(Protocol):
    """A scene (the environment) or a self-collision model: ``in_collision(q) -> bool``, optionally ``distance(q) -> float``
    (metres of clearance, negative inside).

    It checks a finished trajectory; it never searches for one (that is a planner, a later plugin role). The reference a real
    space plugs in is bam_ws's MPlib-derived check: ``bam_collision.acm.generate_allowed_collision_matrix`` (MoveIt Setup
    Assistant's self-collision matrix by sampling, copied from MPlib) plus Pinocchio's ``computeCollisions``, wrapped in an
    object with ``in_collision(q)``."""

    def in_collision(self, q: np.ndarray) -> bool: ...


class DescriptionLike(Protocol):
    """bam_robot_descriptions.RobotDescription, as far as a space reads it."""

    def joint_limits(self, names: Sequence[str] | None = None) -> JointLimits: ...


class ReachMapLike(Protocol):
    """bam_reach.ReachMap's fields a space reads."""

    joint_names: tuple[str, ...]
    joint_solutions: np.ndarray  # (P, O, J)
    valid_mask: np.ndarray  # (P, O)


# ====================================================================== what a need costs, in words

RUNG_TITLES = {"joint_limits": "joint limits", "kinematics": "kinematics", "dynamics": "dynamics", "torque_limits": "torque limits"}
NOT_INPUTS = ("camera",)  # the catalog's own word: a camera measures the output, the generator never needs one
NEED_WORDS = {
    "joint_limits": "a position range", "velocity_limits": "a velocity limit", "acceleration_limits": "an acceleration limit",
    "jerk_limits": "a jerk limit", "torque_limits": "a torque limit", "home_pose": "a named home", "reach_map": "a reach map",
    "forward_kinematics": "forward kinematics", "inverse_kinematics": "inverse kinematics", "dynamics_model": "a dynamics model",
    "collision_model": "a collision checker", "camera": "a camera",
}
KINEMATICS_FIX = "space.with_kinematics(kinematics), where forward(q) -> 4x4 tool pose and inverse(pose, seed) -> q or None"
GAPS = {
    "joint_limits": ("no finite position range", "JointSpace.from_limits(limits) with min/max_position, or space.within((low, high))"),
    "velocity_limits": ("no velocity limit", "state max_velocity in the JointLimits"),
    "acceleration_limits": ("no acceleration limit (a URDF carries none)",
                            "JointSpace.from_limits(limits, max_acceleration=<rad/s²>) or "
                            "JointSpace.from_description(description, max_acceleration=<rad/s²>)"),
    "jerk_limits": ("no jerk limit", "state max_jerk in the JointLimits"),
    "torque_limits": ("no torque limit", "state max_effort in the JointLimits (a URDF's <limit effort=...> has it)"),
    "home_pose": ("no named home", "JointSpace.from_limits(limits, home=READY)"),
    "reach_map": ("no reach map", "space.with_reach_map(reach_map)"),
    "forward_kinematics": ("no kinematics", KINEMATICS_FIX),
    "inverse_kinematics": ("no kinematics", KINEMATICS_FIX),
    "dynamics_model": ("no dynamics model", "space.with_dynamics(model), where torque(q, qd, qdd) predicts the torque "
                                            "(and forward(q, qd, tau) -> qdd simulates effort-commanded kinds)"),
    "collision_model": ("no collision checker", "space.with_collision(scene) or space.with_self_collision(model), where "
                                                "in_collision(q) -> bool"),
    "camera": ("no camera", "nothing: a camera measures the output, the generator never needs one"),
}


class MissingRung(ValueError):
    """Something a kind (or a call) needs is not in this space. The message names the input, its rung and the fix."""

    def __init__(self, who: str, need: str, detail: str = ""):
        rung = ENTRY_LEVEL_OF_NEED.get(need)
        what, fix = GAPS[need]
        self.who, self.need, self.rung, self.fix = who, need, rung, fix
        where = f"rung {ENTRY_LEVELS.index(rung) + 1} of the ladder, {RUNG_TITLES[rung]}" if rung else "beside the ladder"
        super().__init__(f"{who} needs {NEED_WORDS[need]} ({where}); this space has {what}{detail}. Add it: {fix}")


@dataclass(frozen=True)
class Status:
    """Whether something runs on a space: available, fallback (runs, a stated way) or locked (and the fix)."""

    state: str  # "available" | "fallback" | "locked"
    missing: tuple[str, ...] = ()
    why: str = ""
    fix: str = ""
    unchecked: tuple[str, ...] = ()

    def __str__(self) -> str:
        text = self.state + (f": {self.why}" if self.why else "") + (f" ({self.fix})" if self.fix else "")
        if self.unchecked:
            text += f" [not checked here: {', '.join(n.replace('_', ' ') for n in self.unchecked)}]"
        return text


# ====================================================================== the reach map, as a space reads it


@dataclass(frozen=True, eq=False)
class ReachSet:
    """A reach map as kinds read it: every joint solution, which are valid, and the map's own grid step."""

    configurations: np.ndarray  # (M, J)
    valid: np.ndarray  # (M,) bool
    grid_step: np.ndarray  # (J,) rad

    @classmethod
    def from_reach_map(cls, reach_map: ReachMapLike) -> ReachSet:
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


# ====================================================================== small helpers


def _finite(values: Any) -> bool:
    return values is not None and bool(np.all(np.isfinite(np.asarray(values, float))))


def _number(value: float) -> str:
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _deg(values: Any, unit: str = "°") -> str:
    """Radians as degrees for a person: one number when every joint agrees, else a list."""
    texts = [_number(math.degrees(float(value))) for value in np.atleast_1d(values)]
    return (texts[0] if len(set(texts)) == 1 else "[" + ", ".join(texts) + "]") + unit


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


def _with_acceleration(limits: JointLimits, supplied: Any, *, who: str) -> tuple[JointLimits, tuple[str, ...]]:
    """Fill a missing acceleration limit with ``supplied`` (or tighten a stated one), or refuse naming the fix."""
    stated = limits.max_acceleration
    if supplied is None:
        if not _finite(stated):
            raise MissingRung(who, "acceleration_limits", detail=" and every kind solves its motion against one; the "
                              "generator will not guess the value the drives can actually do")
        return limits, ()
    value = np.broadcast_to(np.asarray(supplied, float), (len(limits.joint_names),)).copy()
    if not _finite(stated):
        filled = value if stated is None else np.where(np.isfinite(stated), stated, value)
        return limits.replace(max_acceleration=filled), (f"max_acceleration {_deg(value, '°/s²')} supplied by the caller",)
    tightened = np.minimum(stated, value)  # WHY the minimum: a stated value may tighten a known limit, never loosen it
    return limits.replace(max_acceleration=tightened), (f"max_acceleration tightened to {_deg(tightened, '°/s²')}",)


def _pose(value: Any) -> np.ndarray | None:
    """A forward-kinematics answer as a (4, 4) pose: an array, an xyz, ``(success, pose)``, or bam_reach's FKResult."""
    if value is None:
        return None
    if hasattr(value, "success") and hasattr(value, "pose"):
        if not value.success:
            return None
        value = value.pose
    if isinstance(value, tuple) and len(value) in (2, 3) and isinstance(value[0], (bool, np.bool_)):
        if not value[0]:
            return None
        value = value[1]
    array = np.asarray(value, float)
    if array.shape == (3,):
        pose = np.eye(4)
        pose[:3, 3] = array
        return pose
    if array.shape != (4, 4):
        raise TypeError(f"forward kinematics returned shape {array.shape}; expected a 4x4 pose or an xyz")
    return array


def _solution(value: Any, count: int) -> np.ndarray | None:
    """An inverse-kinematics answer as joint positions: an array, None, ``(success, q)``, or bam_reach's IKResult."""
    if value is None:
        return None
    if hasattr(value, "success") and hasattr(value, "joint_positions"):
        if not value.success:
            return None
        value = value.joint_positions
    if isinstance(value, tuple) and len(value) in (2, 3) and isinstance(value[0], (bool, np.bool_)):
        if not value[0]:
            return None
        value = value[1]
    array = np.asarray(value, float).reshape(-1)
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        return None
    return array


def _slot_record(model: Any) -> dict[str, Any] | None:
    """An injected model as a recipe sees it: its name and cache key (data, not configuration: only fingerprinted)."""
    if model is None:
        return None
    return {"type": type(model).__name__, "name": str(getattr(model, "name", "") or type(model).__name__),
            "cache_key": getattr(model, "cache_key", None)}


# ====================================================================== the space


@dataclass(frozen=True, eq=False)
class JointSpace:
    """Everything a kind may read about the robot. Build it with a ``from_*`` constructor and add rungs with ``with_*``."""

    limits: JointLimits  # the effective limits: every trajectory is generated inside these, and carries them
    home: np.ndarray  # where a kind starts when it is given no start
    rate_hz: float = DEFAULT_RATE_HZ  # WHY on the space: every kind of one session shares one clock
    named_home: bool = False  # True when the caller named it (the catalog's home_pose)
    kinematics: Any = None
    dynamics: Any = None
    collision: Any = None  # the scene / environment
    self_collision: Any = None
    reach: ReachSet | None = None
    source: str = "joint limits"  # what it was built from, in words
    assumptions: tuple[str, ...] = ()  # every value supplied rather than read

    # ------------------------------------------------------------------ rung 1

    @classmethod
    def from_limits(cls, limits: JointLimits, *, home: Sequence[float] | None = None, max_acceleration: Any = None,
                    rate_hz: float = DEFAULT_RATE_HZ, source: str = "joint limits") -> JointSpace:
        """Bare robot_msgs JointLimits (positions, velocities, accelerations; jerk and effort when known)."""
        limits, assumptions = _with_acceleration(limits, max_acceleration, who=f"JointSpace.from_limits ({source})")
        for bound in ("min_position", "max_position"):
            if not _finite(getattr(limits, bound)):
                raise MissingRung(f"JointSpace.from_limits ({source})", "joint_limits",
                                  detail=f" ({bound} is not finite: a continuous joint needs .within((low, high)))")
        if not _finite(limits.max_velocity):
            raise MissingRung(f"JointSpace.from_limits ({source})", "velocity_limits")
        return cls._build(limits, home=home, rate_hz=rate_hz, source=source, assumptions=assumptions)

    @classmethod
    def from_description(cls, description: DescriptionLike, *, max_acceleration: Any = None, home: Sequence[float] | None = None,
                         joints: Sequence[str] | None = None, rate_hz: float = DEFAULT_RATE_HZ) -> JointSpace:
        """A robot description's limits (anything with ``joint_limits()``).

        WHY it refuses without ``max_acceleration``: a URDF has position and velocity limits but never an acceleration
        limit, every kind needs one, and the generator will not guess the value the drives can actually do."""
        name = str(getattr(description, "name", "") or "the description")
        limits = description.joint_limits(None if joints is None else tuple(joints))
        limits, assumptions = _with_acceleration(limits, max_acceleration, who=f"JointSpace.from_description ({name}'s URDF)")
        if not (_finite(limits.min_position) and _finite(limits.max_position)):
            raise MissingRung(f"JointSpace.from_description ({name})", "joint_limits",
                              detail=" (a continuous joint: tighten it with .within((low, high)))")
        return cls._build(limits, home=home, rate_hz=rate_hz, source=f"the description {name!r}", assumptions=assumptions)

    @classmethod
    def _build(cls, limits: JointLimits, *, home: Any, rate_hz: float, source: str, assumptions: tuple[str, ...],
               reach: ReachSet | None = None, **slots: Any) -> JointSpace:
        bounds = k.Bounds.from_limits(limits)
        named = home is not None
        if home is None:
            home = bounds.centre
            if reach is not None:
                # WHY nearest the middle of the reachable set: a stable, explainable default (v6's)
                reachable = reach.reachable
                home = reachable[int(np.argmin(np.max(np.abs(reachable - reachable.mean(axis=0)), axis=1)))]
        home = np.array(home, float)
        if home.shape != (len(limits.joint_names),):
            raise ValueError(f"home has shape {home.shape}, expected ({len(limits.joint_names)},)")
        if np.any(home < bounds.low - 1e-12) or np.any(home > bounds.high + 1e-12):
            raise ValueError(f"home {np.round(np.degrees(home), 2).tolist()}° is outside the position limits")
        if reach is not None and not reach.nearest_is_reachable(home):
            raise ValueError(f"home {np.round(np.degrees(home), 2).tolist()}° is not in the reach map's reachable set")
        return cls(limits=limits, home=home, rate_hz=float(rate_hz), named_home=named, reach=reach, source=source,
                   assumptions=tuple(assumptions), **slots)

    def _slots(self) -> dict[str, Any]:
        return {"kinematics": self.kinematics, "dynamics": self.dynamics, "collision": self.collision,
                "self_collision": self.self_collision}

    def within(self, box: JointLimits | tuple[Sequence[float], Sequence[float]]) -> JointSpace:
        """Positions (and any speeds the box states) tightened to ``box``, a JointLimits or a ``(low, high)`` pair; never loosened."""
        limits, changes = self.limits, {}
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
        return JointSpace._build(tightened, home=self.home if self.named_home else None, rate_hz=self.rate_hz, source=self.source,
                                 assumptions=(*self.assumptions, note), reach=self.reach, **self._slots())

    # ------------------------------------------------------------------ the injected rungs

    def with_kinematics(self, kinematics: Kinematics) -> JointSpace:
        """Rung 2: forward and inverse kinematics (duck-typed on bam_reach's ``Kinematics``): unlocks the Cartesian kinds."""
        missing = [name for name in ("forward", "inverse") if not callable(getattr(kinematics, name, None))]
        if missing or not hasattr(kinematics, "joint_names"):
            raise TypeError(f"kinematics needs forward(q) -> 4x4 pose, inverse(pose, seed) -> q | None and joint_names; "
                            f"{type(kinematics).__name__} lacks {missing or ['joint_names']}")
        if tuple(kinematics.joint_names) != self.joint_names:
            raise ValueError(f"the kinematics' joints {tuple(kinematics.joint_names)} are not this space's {self.joint_names}")
        return replace(self, kinematics=kinematics, source=f"{self.source} + kinematics")

    def with_dynamics(self, model: Dynamics) -> JointSpace:
        """Rung 3: a model that predicts torque, ``torque(q, qd, qdd)`` (pin_utils' ``inverse_dynamics`` is read too)."""
        if not (callable(getattr(model, "torque", None)) or callable(getattr(model, "inverse_dynamics", None))):
            raise TypeError("a dynamics model needs torque(q, qd, qdd) (or pin_utils' inverse_dynamics(q, velocity, acceleration))")
        return replace(self, dynamics=model, source=f"{self.source} + dynamics")

    def with_collision(self, scene: CollisionChecker) -> JointSpace:
        """The scene (environment): ``in_collision(q) -> bool``. Every kind's output is then checked against it."""
        _need_checker(scene, "scene")
        return replace(self, collision=scene, source=f"{self.source} + a collision scene")

    def with_self_collision(self, model: CollisionChecker) -> JointSpace:
        """A self-collision model: ``in_collision(q) -> bool``. Every kind's output is then checked against it."""
        _need_checker(model, "self-collision model")
        return replace(self, self_collision=model, source=f"{self.source} + self-collision")

    def with_reach_map(self, reach_map: ReachMapLike, *, home: Sequence[float] | None = None) -> JointSpace:
        """Rung 2's optional cache: the reachable set. Every centred kind then stays inside the largest box around where it
        starts that holds no unreachable configuration (``room``), and ``workspace_sweep`` tours the set itself."""
        names = tuple(reach_map.joint_names)
        if names != self.joint_names:
            raise ValueError(f"the reach map's joints {names} are not this space's {self.joint_names}")
        reach = ReachSet.from_reach_map(reach_map)
        if home is None and self.named_home:
            home = self.home
        space = JointSpace._build(self.limits, home=home, rate_hz=self.rate_hz, source=f"{self.source} + a reach map",
                                  assumptions=self.assumptions, reach=reach, **self._slots())
        if home is None:
            return replace(space, assumptions=(*space.assumptions, f"home: the reachable configuration nearest the middle "
                                                                    f"of the reach map, {_deg(space.home)}"))
        return space

    # ------------------------------------------------------------------ what kinds read

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.limits.joint_names)

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    def bounds(self) -> k.Bounds:
        return k.Bounds.from_limits(self.limits)

    def joint_index(self, which: str | int | None) -> int:
        """A joint by name or index; ``None`` is allowed on a 1-DOF robot only."""
        if which is None:
            if self.dof == 1:
                return 0
            raise ValueError(f"which joint? this robot has {self.joint_names}: pass joint=<name>")
        if isinstance(which, str):
            if which not in self.joint_names:
                raise ValueError(f"no joint {which!r}; joints are {self.joint_names}")
            return self.joint_names.index(which)
        if not 0 <= int(which) < self.dof:
            raise ValueError(f"joint index {which} out of range for {self.joint_names}")
        return int(which)

    def joint_indices(self, which: Sequence[str | int] | str | None) -> list[int]:
        """Several joints; ``None`` is every joint."""
        if which is None:
            return list(range(self.dof))
        if isinstance(which, (str, int)):
            return [self.joint_index(which)]
        return [self.joint_index(name) for name in which]

    def start_at(self, start: Any = None) -> np.ndarray:
        """Where a kind starts: ``None`` (home), a configuration, or a JointTrajectory (where it ends). Refused outside the limits."""
        if start is None:
            q = self.home.copy()
        elif isinstance(start, JointTrajectory):
            if tuple(start.joint_names) != self.joint_names:
                raise ValueError(f"start is a trajectory of joints {start.joint_names}; this space has {self.joint_names}")
            q = np.array(start.positions[-1], float)
        else:
            q = np.array(start, float).reshape(-1)
            if q.shape != (self.dof,):
                raise ValueError(f"start has {q.size} values; this space has {self.dof} joints {self.joint_names}")
        bounds = self.bounds
        if np.any(q < bounds.low - 1e-12) or np.any(q > bounds.high + 1e-12):
            raise ValueError(f"start {np.round(np.degrees(q), 3).tolist()}° is outside the position limits "
                             f"{np.round(np.degrees(bounds.low), 2).tolist()}..{np.round(np.degrees(bounds.high), 2).tolist()}°")
        return q

    def state_at(self, start: Any = None) -> State:
        """Where a moving-start kind starts: a JointTrajectory's last sample (q, qd, qdd), a State, a ``{"q", "qd", "qdd"}``
        dict (a recipe's), or a configuration / None at rest."""
        if isinstance(start, JointTrajectory):
            q = self.start_at(start)
            last = len(start.t) - 1
            return State(q, np.array(start.velocities[last], float), np.array(start.accelerations[last], float))
        if isinstance(start, State):
            return State(self.start_at(start.q), np.array(start.qd, float), np.array(start.qdd, float))
        if isinstance(start, Mapping) and "q" in start:
            return State(self.start_at(start["q"]), np.array(start.get("qd", np.zeros(self.dof)), float),
                         np.array(start.get("qdd", np.zeros(self.dof)), float))
        return State.rest(self.start_at(start))

    def room(self, centre: np.ndarray) -> k.Bounds:
        """The box a kind may use around ``centre``: the limit box, or with a reach map the largest box around ``centre``
        that holds no unreachable configuration.

        WHY here and not in each kind (all five API proposals converged on it): safe by construction. Every centred kind
        asks the space for its room, so a reach map keeps all of them inside the reachable set at once, and no kind needs to
        know what a reach map is."""
        if self.reach is None:
            return self.bounds
        return k.inscribed_box(centre, self.reach.unreachable, self.bounds)

    # ------------------------------------------------------------------ kinematics

    @property
    def position_only(self) -> bool:
        """True when the kinematics places a point, not a pose (``orientation_constraint`` "none", e.g. a planar five-bar)."""
        constraint = getattr(self.kinematics, "orientation_constraint", "full")
        return str(getattr(constraint, "value", constraint)).lower() == "none"

    def forward(self, q: Any) -> np.ndarray:
        """The tool pose (4, 4) in the base frame at ``q``."""
        if self.kinematics is None:
            raise MissingRung("forward kinematics", "forward_kinematics")
        pose = _pose(self.kinematics.forward(np.asarray(q, float)))
        if pose is None:
            raise ValueError(f"the kinematics has no tool pose at {np.round(np.degrees(np.asarray(q, float)), 3).tolist()}°")
        return pose

    def inverse(self, pose: Any, seed: Any = None) -> np.ndarray | None:
        """Joint positions that put the tool at ``pose`` (a 4x4, or an xyz), seeded at ``seed``; None when there is none."""
        if self.kinematics is None:
            raise MissingRung("inverse kinematics", "inverse_kinematics")
        target = np.asarray(pose, float)
        if target.shape == (3,):
            matrix = np.eye(4)
            matrix[:3, 3] = target
            target = matrix
        return _solution(self.kinematics.inverse(target, None if seed is None else np.asarray(seed, float)), self.dof)

    def tool_points(self, trajectory: JointTrajectory) -> np.ndarray:
        """(N, 3) tool positions along ``trajectory`` (forward kinematics of every sample)."""
        return np.array([self.forward(q)[:3, 3] for q in np.asarray(trajectory.positions)])

    # ------------------------------------------------------------------ dynamics

    def torque(self, q: Any, qd: Any, qdd: Any) -> np.ndarray:
        """The joint torque the dynamics model predicts for one state."""
        if self.dynamics is None:
            raise MissingRung("torque prediction", "dynamics_model")
        function = getattr(self.dynamics, "torque", None) or self.dynamics.inverse_dynamics
        return np.asarray(function(np.asarray(q, float), np.asarray(qd, float), np.asarray(qdd, float)), float)

    def torques(self, trajectory: JointTrajectory) -> np.ndarray:
        """(N, J) the torque the dynamics model predicts along ``trajectory`` (its torque feasibility)."""
        return np.array([self.torque(q, qd, qdd) for q, qd, qdd in
                         zip(trajectory.positions, trajectory.velocities, trajectory.accelerations, strict=True)])

    def predict(self, start: Any, effort: np.ndarray, *, start_velocity: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(q, qd, qdd), each (N, J): the motion ``effort`` (N, J) produces from ``start`` at rest, by the model's
        ``forward(q, qd, tau) -> qdd`` (semi-implicit Euler at ``rate_hz``). For effort-commanded kinds (catalog lane B)."""
        forward = getattr(self.dynamics, "forward", None)
        if self.dynamics is None or not callable(forward):
            raise MissingRung("predicting an effort-commanded motion", "dynamics_model",
                              detail="" if self.dynamics is None else " with no forward(q, qd, tau) -> qdd")
        effort = np.asarray(effort, float)
        dt = 1.0 / self.rate_hz
        q = np.empty_like(effort)
        qd = np.empty_like(effort)
        qdd = np.empty_like(effort)
        position = np.array(start, float)
        velocity = np.zeros(self.dof) if start_velocity is None else np.array(start_velocity, float)
        for index, tau in enumerate(effort):
            q[index], qd[index] = position, velocity
            qdd[index] = np.asarray(forward(position, velocity, tau), float)
            velocity = velocity + qdd[index] * dt
            position = position + velocity * dt
        return q, qd, qdd

    # ------------------------------------------------------------------ collision

    @property
    def checks_collision(self) -> bool:
        return self.collision is not None or self.self_collision is not None

    def in_collision(self, q: Any) -> bool:
        """True when the scene or the self-collision model says ``q`` collides (False when neither is injected)."""
        q = np.asarray(q, float)
        return any(bool(model.in_collision(q)) for model in (self.collision, self.self_collision) if model is not None)

    def collision_with(self, q: Any) -> str:
        """Which injected checker ``q`` collides with, in words ("" when none)."""
        q = np.asarray(q, float)
        for model, words in ((self.collision, "the scene"), (self.self_collision, "itself")):
            if model is not None and bool(model.in_collision(q)):
                return f"{words} ({_slot_record(model)['name']})"
        return ""

    # ------------------------------------------------------------------ what it knows, graded

    def has(self, need: str) -> bool:
        """Whether this space knows ``need`` (a word of the catalog's needs vocabulary)."""
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
        if need == "collision_model":
            return self.checks_collision
        if need in NOT_INPUTS:
            return False
        raise ValueError(f"unknown need {need!r}; the catalog's needs are {tuple(ENTRY_LEVEL_OF_NEED)}")

    def status(self, needs: Iterable[str], *, better_with: Mapping[str, str] | None = None, unchecked: Iterable[str] = (),
               reach_from_kinematics: bool = False) -> Status:
        """Grade anything that ``needs`` these inputs here: available, fallback (a soft need missing) or locked.

        WHY ``reach_from_kinematics`` (catalog rows only): on Zach's ladder a reach map is a cache of the kinematics rung, so
        for a row (is the input there?) kinematics covers it; a kind that reads the reachable set asks for the map itself."""
        better_with = dict(better_with or {})
        needs = list(dict.fromkeys([*needs, *better_with]))  # a soft need is still a need: missing, it falls back
        needs = [need for need in needs if need not in NOT_INPUTS]
        skipped = tuple(need for need in unchecked if not self.has(need))

        def known(need: str) -> bool:
            return self.has(need) or (need == "reach_map" and reach_from_kinematics and self.kinematics is not None)

        missing = [need for need in needs if not known(need)]
        hard = [need for need in missing if need not in better_with]
        if hard:
            shown = [need for need in hard if not (need == "reach_map" and reach_from_kinematics and
                                                   {"forward_kinematics", "inverse_kinematics"} & set(hard))]
            gaps = [GAPS[need] for need in shown]
            return Status("locked", tuple(hard), "; ".join(dict.fromkeys(why for why, _ in gaps)),
                          "; ".join(dict.fromkeys(fix for _, fix in gaps)), skipped)
        if missing:
            why, fix = GAPS[missing[0]]
            return Status("fallback", tuple(missing), f"{better_with[missing[0]]} ({why})", fix, skipped)
        return Status("available", (), "", "", skipped)

    # ------------------------------------------------------------------ as a recipe records it

    def record(self) -> dict[str, Any]:
        """What decides the motion, as JSON-ready data. Injected models are data, not configuration: only their names and
        cache keys go in, so a rebuild is handed the same models and a different one is refused by the fingerprint."""
        body = {"format": SPACE_FORMAT, "joint_names": list(self.joint_names), "rate_hz": self.rate_hz,
                "limits": limits_to_json(self.limits), "home": [float(value) for value in self.home], "named_home": self.named_home,
                "reach_map": None if self.reach is None else self.reach.fingerprint(),
                **{slot: _slot_record(model) for slot, model in self._slots().items()}}
        fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
        return {**body, "source": self.source, "fingerprint": fingerprint}

    @property
    def fingerprint(self) -> str:
        """16 hex characters over every limit, the home, the rate and the injected models' names (never the source words)."""
        return self.record()["fingerprint"]

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> JointSpace:
        """The space a ``record()`` describes, when it is joint limits alone; a space with injected models is refused."""
        injected = [slot for slot in ("reach_map", "kinematics", "dynamics", "collision", "self_collision") if record.get(slot)]
        if injected:
            raise ValueError(f"this recipe was made on a space with {', '.join(injected)} ({record.get('source', '')}); "
                             "those are data, not text: pass the space, rebuild(text, space)")
        space = cls.from_limits(limits_from_json(record["limits"]), home=record["home"] if record.get("named_home") else None,
                                rate_hz=record["rate_hz"], source=record.get("source", "joint limits"))
        if space.fingerprint != record["fingerprint"]:
            raise ValueError(f"the recipe's space does not rebuild to the same fingerprint ({space.fingerprint} != {record['fingerprint']})")
        return space

    # ------------------------------------------------------------------ printing

    def describe(self) -> str:
        """Each rung known or missing, angles in degrees."""
        names = self.joint_names
        joints = ", ".join(names) if len(names) <= 3 else f"{names[0]} .. {names[-1]}"
        limits = self.limits
        lines = [f"JointSpace of {len(names)} joint{'s' * (len(names) != 1)} ({joints}) from {self.source}, {self.rate_hz:g} Hz"]
        pairs = [f"{_number(math.degrees(a))}..{_number(math.degrees(b))}" for a, b in zip(limits.min_position, limits.max_position)]
        known = [f"position {(pairs[0] if len(set(pairs)) == 1 else ', '.join(pairs))}°", f"velocity {_deg(limits.max_velocity, '°/s')}",
                 f"acceleration {_deg(limits.max_acceleration, '°/s²')}",
                 f"jerk {_deg(limits.max_jerk, '°/s³')}" if _finite(limits.max_jerk) else "no jerk limit"]
        lines.append(f"  1 joint limits   known    {'; '.join(known)}")
        rows = (("kinematics", self.kinematics, "with_kinematics(kinematics)"), ("dynamics", self.dynamics, "with_dynamics(model)"),
                ("torque limits", limits.max_effort if _finite(limits.max_effort) else None, "state max_effort in the limits"))
        for index, (title, model, fix) in enumerate(rows, start=2):
            if model is None:
                lines.append(f"  {index} {title:<14} missing  {fix}")
            else:
                detail = f"effort {np.round(model, 2).tolist()} N·m" if title == "torque limits" else _slot_record(model)["name"]
                lines.append(f"  {index} {title:<14} known    {detail}")
            if title == "kinematics":
                lines.append("    reach map        " + (f"known    {int(self.reach.valid.sum())} of {len(self.reach.configurations)} "
                                                      f"configurations reachable on a {_deg(self.reach.grid_step)} grid"
                                                      if self.reach is not None else "missing  an optional cache · with_reach_map(map)"))
        for slot, words in (("collision", "scene"), ("self_collision", "self-collision")):
            model = getattr(self, slot)
            lines.append(f"  beside the ladder {words:<15} " + ("known    " + _slot_record(model)["name"] if model is not None
                                                                 else f"missing  with_{slot}({'scene' if slot == 'collision' else 'model'})"))
        lines.append(f"  home             {_deg(self.home)} ({'named' if self.named_home else 'the box centre' if self.reach is None else 'nearest the middle of the reachable set'})")
        lines += [f"  supplied         {assumption}" for assumption in self.assumptions]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()

    def __repr__(self) -> str:
        return f"JointSpace({self.dof} joints, {self.source}, {self.fingerprint})"


def _need_checker(model: Any, words: str) -> None:
    if not callable(getattr(model, "in_collision", None)):
        raise TypeError(f"a {words} needs in_collision(q) -> bool (and optionally distance(q) -> float)")


__all__ = ["CollisionChecker", "DescriptionLike", "Dynamics", "JointSpace", "Kinematics", "MissingRung", "ReachMapLike", "ReachSet",
           "State", "Status", "limits_from_json", "limits_to_json"]
