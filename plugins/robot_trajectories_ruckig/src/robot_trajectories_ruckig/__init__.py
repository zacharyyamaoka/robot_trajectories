"""Ruckig Community as a robot_trajectories time parameterization: ``TimeParameterization("ruckig", ...)`` times one leg as a
jerk-limited S-curve.

    law = TimeParameterization("ruckig", peak_speed_fraction=0.8, options={"jerk_fraction": 0.5, "synchronization": "phase"})
    joint_waypoints(space, waypoints, time_parameterization=law)   # every leg rest-to-rest, an S-curve each
    law.leg(q0, q1, space.limits, space.rate_hz, end_state=BoundaryState(velocity, acceleration))   # a leg that ends moving

Registered through the entry point ``robot_trajectories.time_parameterizations``
(``ruckig = "robot_trajectories_ruckig:RuckigBackend"``); robot_trajectories never imports it (its leaf rule). Zach,
2026-11-18: "for simple things you keep the implementation yourself but the other things other repos can implement them...
like ruckig (and if you have the professional version we can support that as a drop-in backend as well)". This is the
Community half; ``robot-trajectories-ruckig-pro`` is the drop-in.

What it serves:
- **one leg** (a path of two points) from any start state to any end state (position, velocity, acceleration), time-optimal
  under the velocity, acceleration and jerk limits it is handed (Berscheid & Kroger 2021, Ruckig);
- the duration lands on the sample grid (Ruckig's own ``DurationDiscretization.Discrete``), so the last sample is exactly the
  end state, never a held copy of it;
- ``law.move_s`` and ``law.min_move_s`` as a floor on the duration (Ruckig's ``minimum_duration``);
- ``law.options``: ``acceleration_fraction`` and ``jerk_fraction`` (of the stated limits, default 1), ``synchronization``
  (``"phase"``, the default: a rest-to-rest multi-joint leg stays a straight line in joint space, as the catalog row asks;
  ``"time"``, ``"time_if_necessary"``, ``"none"``), and ``max_jerk`` (rad/s^3, one number or one per joint) only when the
  JointLimits state none.

What it refuses (``can_service``, before Ruckig is called, and again in ``time_parameterize``):
- **more than two path points.** WHY (PEP 0002 §6; bam_traj_gen ``ruckig_backend.py``): Ruckig Community computes
  intermediate waypoints on Ruckig's cloud servers (its native module carries a ``CloudClient``; the switch is
  ``Calculator::use_waypoints_trajectory``: non-empty ``intermediate_positions``). This package never sets
  ``intermediate_positions``, so no motion leaves the machine. Waypoints are Ruckig Pro's, locally:
  ``robot-trajectories-ruckig-pro``.
- a leg without a jerk limit (the S-curve is defined by it), an unknown option (a typo is not ignored), a velocity or
  acceleration fraction outside (0, 1].
- a leg that leaves the position limits. WHY: Community ignores ``min_position`` / ``max_position`` (only Pro enforces them),
  so the samples are checked here and the refusal names Pro.

Installed beside ``robot-trajectories-ruckig-pro`` it refuses to load: both own the ``ruckig`` module and register the time
law ``ruckig``, so which binary ran would depend on install order. Install one of the two.
"""

from __future__ import annotations

import math
from importlib import metadata
from typing import Any

import numpy as np
import ruckig
from robot_msgs.trajectory import JointLimits, JointTrajectory

NAME = "ruckig"
EDITION = "community"
PRO_PACKAGE = "robot-trajectories-ruckig-pro"
PRO_DISTRIBUTION = "ruckig-pro"
COMMUNITY_DISTRIBUTION = "ruckig"
SYNCHRONIZATION = {"phase": ruckig.Synchronization.Phase, "time": ruckig.Synchronization.Time,
                   "time_if_necessary": ruckig.Synchronization.TimeIfNecessary, "none": ruckig.Synchronization.No}
OPTIONS = ("acceleration_fraction", "jerk_fraction", "max_jerk", "synchronization")
END_TOLERANCE = 1e-6  # WHY: Ruckig reaches its target to ~1e-10; a larger miss is a bug, not rounding, and is refused
LIMIT_TOLERANCE = 1e-9  # relative, on the samples: Ruckig's profiles sit on their limits exactly


def waypoint_refusal(count: int) -> str:
    """Why Community never times a path of ``count`` > 2 points, naming the drop-in that does."""
    return (f"Ruckig Community times one leg (two path points), not a path of {count}: Community sends intermediate "
            f"waypoints to Ruckig's cloud servers, so they are never handed to it. Install {PRO_PACKAGE} (the local Ruckig Pro "
            f"build, a drop-in for the time parameterization 'ruckig') instead of this package for waypoints, or time each leg "
            f"on its own (joint_waypoints / legs_through stop at every point)")


def _refuse_beside_pro() -> None:
    try:
        metadata.distribution(PRO_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return
    raise RuntimeError(f"the {PRO_DISTRIBUTION!r} distribution is installed beside {COMMUNITY_DISTRIBUTION!r}: both own the "
                       f"'ruckig' module, so which one runs depends on install order. Install exactly one of "
                       f"robot-trajectories-ruckig (Community) or {PRO_PACKAGE} (Pro)")


class RuckigBackend:
    """The time parameterization ``ruckig`` on Ruckig Community: one state-to-state leg, never intermediate waypoints."""

    name = NAME
    edition = EDITION
    needs = ("jerk_limits",)

    def __init__(self) -> None:
        _refuse_beside_pro()

    # ------------------------------------------------------------------ the seam

    def can_service(self, path_points: Any, limits: JointLimits, law: Any, start_state: Any = None, end_state: Any = None) -> str:
        path = np.asarray(path_points, float)
        if path.ndim != 2 or len(path) < 2:
            return f"a path is (M, J) with M >= 2 points, got shape {path.shape}"
        if len(path) > 2:
            return waypoint_refusal(len(path))
        try:
            settings(limits, law)
        except ValueError as error:
            return str(error)
        return ""

    def time_parameterize(self, path_points: Any, limits: JointLimits, rate_hz: float, law: Any, *, start_state: Any = None,
                          end_state: Any = None) -> JointTrajectory:
        # WHY checked again here: a caller may skip can_service; the waypoint refusal must hold on every route in
        refusal = self.can_service(path_points, limits, law, start_state, end_state)
        if refusal:
            raise ValueError(refusal)
        path = np.asarray(path_points, float)
        t, q, qd, qdd = solve_leg(path[0], path[1], limits, rate_hz, law, start_state, end_state)
        outside = position_violation(q, limits)
        if outside:
            raise ValueError(f"{outside}: Ruckig Community does not enforce position limits (only Pro does: {PRO_PACKAGE}); "
                             "lower the boundary speeds or move the point")
        return JointTrajectory(t=t, joint_names=tuple(limits.joint_names), positions=q, velocities=qd, accelerations=qdd,
                               limits=limits)


# ====================================================================== the leg


def settings(limits: JointLimits, law: Any) -> dict[str, Any]:
    """Ruckig's limits and synchronization from the JointLimits handed in and the law's options, or a ValueError saying why not."""
    options = dict(law.options) if law is not None else {}
    unknown = sorted(set(options) - set(OPTIONS))
    if unknown:
        raise ValueError(f"time parameterization 'ruckig' has no option {unknown} (it takes {OPTIONS})")
    dof = len(limits.joint_names)
    fractions = {}
    for key in ("acceleration_fraction", "jerk_fraction"):
        value = float(options.get(key, 1.0))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"time parameterization 'ruckig': {key} must lie in (0, 1], got {value}")
        fractions[key] = value
    sync = options.get("synchronization", "phase")
    if sync not in SYNCHRONIZATION:
        raise ValueError(f"time parameterization 'ruckig': synchronization must be one of {tuple(SYNCHRONIZATION)}, got {sync!r}")
    velocity = _finite_positive(limits.max_velocity, dof)
    acceleration = _finite_positive(limits.max_acceleration, dof)
    if velocity is None or acceleration is None:
        raise ValueError("time parameterization 'ruckig' needs finite, positive max_velocity and max_acceleration in the "
                         "JointLimits")
    jerk = _finite_positive(limits.max_jerk, dof)
    if jerk is None and "max_jerk" in options:
        jerk = _finite_positive(np.broadcast_to(np.asarray(options["max_jerk"], float), (dof,)), dof)
    if jerk is None:
        # WHY refused, never defaulted: a jerk-limited S-curve is defined by its jerk limit, and the drives' value is not ours to guess
        raise ValueError("a jerk-limited S-curve needs a jerk limit: state max_jerk in the JointLimits (the space's jerk_limits "
                         "rung) or TimeParameterization('ruckig', ..., options={'max_jerk': ...}) in rad/s^3")
    return {"max_velocity": velocity, "max_acceleration": acceleration * fractions["acceleration_fraction"],
            "max_jerk": jerk * fractions["jerk_fraction"], "synchronization": SYNCHRONIZATION[sync]}


def solve_leg(q0: np.ndarray, q1: np.ndarray, limits: JointLimits, rate_hz: float, law: Any, start_state: Any,
              end_state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One Ruckig leg q0 -> q1 between the two states, sampled at ``rate_hz`` from t = 0, the last sample the end state."""
    values = settings(limits, law)
    dof = len(q0)
    v0, a0 = _state(start_state, dof)
    v1, a1 = _state(end_state, dof)
    dt = 1.0 / rate_hz
    inp = ruckig.InputParameter(dof)
    inp.current_position, inp.current_velocity, inp.current_acceleration = q0.tolist(), v0.tolist(), a0.tolist()
    inp.target_position, inp.target_velocity, inp.target_acceleration = q1.tolist(), v1.tolist(), a1.tolist()
    inp.max_velocity = values["max_velocity"].tolist()
    inp.max_acceleration = values["max_acceleration"].tolist()
    inp.max_jerk = values["max_jerk"].tolist()
    # WHY Discrete: the duration is a whole number of samples, so the last sample is the end state itself (a moving end
    # cannot be "held" past its time the way a rest-to-rest leg can)
    inp.duration_discretization = ruckig.DurationDiscretization.Discrete
    floor = max(float(law.min_move_s), float(law.move_s or 0.0), 2 * dt)
    inp.minimum_duration = floor
    # WHY never intermediate_positions (nor per-section inputs): with them set, Community calls Ruckig's cloud API
    assert not inp.intermediate_positions
    otg = ruckig.Ruckig(dof, dt)
    trajectory = ruckig.Trajectory(dof)
    # WHY a Time fallback: Ruckig 0.19.4 with Phase synchronization and a Discrete duration can miss a moving end state (seen
    # 2026-11-18 on one joint, 0 -> -0.165 rad ending at 0.8 rad/s and -1 rad/s^2: the end 1 mrad and 1.3 mrad/s off), while
    # Time synchronization lands it exactly. Ruckig itself falls back to Time when phase synchronization is impossible.
    candidates = [values["synchronization"]]
    if values["synchronization"] != ruckig.Synchronization.Time:
        candidates.append(ruckig.Synchronization.Time)
    for sync in candidates:
        inp.synchronization = sync
        calculate(otg, inp, trajectory, "Ruckig")
        if end_miss(trajectory, q1, v1) <= END_TOLERANCE:
            break
    duration = float(trajectory.duration)
    steps = int(round(duration * rate_hz))
    if steps < 2 or abs(steps * dt - duration) > 1e-9:
        raise ValueError(f"Ruckig's duration {duration:.9f} s is not a whole number of {rate_hz:g} Hz samples")
    t = np.arange(steps + 1) / rate_hz
    rows = [trajectory.at_time(min(float(time), duration)) for time in t]
    q = np.array([row[0] for row in rows], float).reshape(len(t), dof)
    qd = np.array([row[1] for row in rows], float).reshape(len(t), dof)
    qdd = np.array([row[2] for row in rows], float).reshape(len(t), dof)
    snap_ends(q, qd, qdd, (q0, v0, a0), (q1, v1, a1))
    return t, q, qd, qdd


def calculate(otg: Any, inp: Any, trajectory: Any, who: str) -> None:
    """Ruckig's offline calculation, its refusals as a ValueError in Ruckig's own words."""
    try:
        result = otg.calculate(inp, trajectory)
    except ruckig.RuckigError as error:  # WHY reworded, not swallowed: Ruckig's own reason (e.g. a target speed over the limit)
        raise ValueError(f"{who} refused this leg: {' '.join(str(error).split())}") from error
    if result not in (ruckig.Result.Working, ruckig.Result.Finished):
        raise ValueError(f"{who} refused this leg ({result.name}): the boundary states may be unreachable together under these limits")


def end_miss(trajectory: Any, position: np.ndarray, velocity: np.ndarray) -> float:
    """How far Ruckig's solution ends from the end state (the larger of the position and velocity misses)."""
    q, qd, _ = trajectory.at_time(float(trajectory.duration))
    return max(float(np.max(np.abs(np.asarray(q) - position))), float(np.max(np.abs(np.asarray(qd) - velocity))))


def snap_ends(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, first: tuple, last: tuple) -> None:
    """Write the boundary states exactly into the first and last samples, after checking Ruckig is within END_TOLERANCE of them."""
    for row, (position, velocity, acceleration), which in ((0, first, "start"), (-1, last, "end")):
        miss = max(float(np.max(np.abs(q[row] - position))), float(np.max(np.abs(qd[row] - velocity))))
        if miss > END_TOLERANCE:
            raise ValueError(f"Ruckig's {which} sample misses the {which} state by {miss:.3g} (position rad or velocity rad/s)")
        q[row], qd[row], qdd[row] = position, velocity, acceleration


def position_violation(q: np.ndarray, limits: JointLimits) -> str:
    """The first joint whose samples leave the position limits, in words (degrees), or ""."""
    for bound, sign in (("min_position", -1.0), ("max_position", 1.0)):
        values = getattr(limits, bound)
        if values is None:
            continue
        excess = sign * (q - np.asarray(values, float)[None, :])
        if np.any(excess > 1e-9):
            row, joint = np.unravel_index(int(np.argmax(excess)), excess.shape)
            return (f"{limits.joint_names[joint]} reaches {math.degrees(q[row, joint]):.3f}°, past its {bound.replace('_', ' ')} "
                    f"{math.degrees(float(values[joint])):.3f}°")
    return ""


def _state(state: Any, dof: int) -> tuple[np.ndarray, np.ndarray]:
    if state is None:
        return np.zeros(dof), np.zeros(dof)
    return (np.broadcast_to(np.asarray(state.velocity, float), (dof,)).copy(),
            np.broadcast_to(np.asarray(state.acceleration, float), (dof,)).copy())


def _finite_positive(values: Any, dof: int) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, float).reshape(-1)
    if array.shape != (dof,) or not np.all(np.isfinite(array)) or not np.all(array > 0):
        return None
    return array


__all__ = ["EDITION", "NAME", "OPTIONS", "PRO_PACKAGE", "RuckigBackend", "position_violation", "settings", "solve_leg",
           "waypoint_refusal"]
