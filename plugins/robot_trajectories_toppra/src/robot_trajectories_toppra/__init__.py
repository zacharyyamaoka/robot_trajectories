"""TOPP-RA as a robot_trajectories time law: ``TimeLaw("toppra", ...)`` finds the fastest timing along a fixed joint path.

    law = TimeLaw("toppra", peak_speed_fraction=1.0, options={"acceleration_fraction": 0.8})
    law.along(path, space.limits, space.rate_hz)          # a dense joint path (M, J), followed point by point
    cartesian_path(space, [p1, p2], time_law=law)          # the IK path of a tool line, timed time-optimally
    time_optimal_path(space, path)                         # the catalog row's kind

Registered through the entry point ``robot_trajectories.time_laws`` (``toppra = "robot_trajectories_toppra:ToppraBackend"``);
robot_trajectories never imports it (its leaf rule).

What it does (Pham & Pham 2018, TOPP-RA; prior-art note 2026-11-18 §5): the path is fixed and only its timing is chosen.
The path points become a cubic spline q(s) through every point, evenly spaced in s (so ``path_progress``, the fractional
point index, is s rescaled);
TOPP-RA (``toppra.algorithm.TOPPRA`` with ``JointVelocityConstraint`` and ``JointAccelerationConstraint``) returns the
fastest path speed s-dot at grid points, rest to rest; the samples are the constant-path-acceleration profile between them
(``ParametrizeConstAccel``'s, computed here so every sample's s is known):

    q = q(s),  q-dot = q'(s) s-dot,  q-ddot = q''(s) s-dot^2 + q'(s) s-ddot.

Then the whole profile is slowed uniformly, by the least factor >= 1 that (a) puts the last sample exactly on the rate's grid,
(b) keeps every joint inside its velocity and acceleration limits on a fine check grid (TOPP-RA holds the limits at its grid
points; between them the spline's curvature can push a joint a little over), and (c) honours ``law.move_s`` /
``law.min_move_s`` as a floor. Slowing by c divides velocities by c and accelerations by c^2, so it never breaks a limit.

``law.options``: ``acceleration_fraction`` (of the stated limit, default 1) and ``grid_points`` (TOPP-RA's grid, default 200,
the catalog row's number; the path's own points are always on it).

Refused (``can_service``): a start or end that moves (this backend times rest to rest; minimum-jerk's quintic or Ruckig
serve moving boundaries), an unknown option, and **the torque-limited mode**: TOPP-RA can bound joint torques
(``JointTorqueConstraint`` / ``SecondOrderConstraint(inv_dyn, ...)``), but that needs the space's inverse dynamics, and the
time-law seam hands a backend only the path, the JointLimits (``max_effort`` included), the rate and the law, never the
space. That is a core change (``NEEDED_CORE_CHANGE``), proposed, not worked around.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import warnings

import numpy as np
from robot_msgs.trajectory import JointLimits, JointTrajectory

with warnings.catch_warnings():  # WHY: toppra 0.6.11's docstrings hold invalid escapes, a SyntaxWarning on 3.12+ at first compile
    warnings.simplefilter("ignore", SyntaxWarning)
    import toppra
    import toppra.algorithm
    import toppra.constraint

NAME = "toppra"
OPTIONS = ("acceleration_fraction", "grid_points")
TORQUE_WORDS = ("torque", "effort", "dynamics", "inverse_dynamics", "max_effort")
GRID_POINTS = 200  # WHY 200: the catalog row time_optimal_path's own default
CHECK_PER_INTERVAL = 8  # samples per TOPP-RA interval on the limit check (both ends of each interval included)
NEEDED_CORE_CHANGE = (
    "the time-law seam passes a backend (path_points, limits, rate_hz, law, start_state, end_state) and never the JointSpace, "
    "so a backend cannot read space.dynamics. Proposed (a barrier, core files time_law.py + the path kinds): TimeLaw.leg / "
    "TimeLaw.along take dynamics= (or space=), _by_backend passes it to time_parameterize as a keyword when the backend's "
    "needs name dynamics_model, and refuses with MissingRung('dynamics_model') when the space has none; the kinds that time "
    "paths (legs_through, cartesian.follow, joint_waypoints, cartesian_path, time_optimal_path) pass space.dynamics. The "
    "torque bound itself is already handed in: JointLimits.max_effort")

logging.getLogger("toppra").setLevel(logging.ERROR)  # WHY: toppra logs every solve at INFO/WARNING; refusals here say why


class ToppraBackend:
    """The time law ``toppra``: time-optimal along a fixed joint path under velocity and acceleration limits, rest to rest."""

    name = NAME
    needs = ("velocity_limits", "acceleration_limits")

    # ------------------------------------------------------------------ the seam

    def can_service(self, path_points: Any, limits: JointLimits, law: Any, start_state: Any = None, end_state: Any = None) -> str:
        path = np.asarray(path_points, float)
        if path.ndim != 2 or len(path) < 2:
            return f"a path is (M, J) with M >= 2 points, got shape {path.shape}"
        if not (_at_rest(start_state) and _at_rest(end_state)):
            return ("TOPP-RA here times a path rest to rest; a start or end that moves takes TimeLaw.minimum_jerk (the quintic "
                    "through both states) or the 'ruckig' time law")
        try:
            settings(limits, law)
        except ValueError as error:
            return str(error)
        return ""

    def time_parameterize(self, path_points: Any, limits: JointLimits, rate_hz: float, law: Any, *, start_state: Any = None,
                          end_state: Any = None) -> JointTrajectory:
        refusal = self.can_service(path_points, limits, law, start_state, end_state)
        if refusal:
            raise ValueError(refusal)
        path = np.asarray(path_points, float)
        t, q, qd, qdd, progress = time_optimal(path, limits, rate_hz, law)
        outside = _position_violation(q, limits)
        if outside:
            raise ValueError(f"the spline through the path points leaves the position limits ({outside}); add points where the "
                             "path turns near a limit")
        return JointTrajectory(t=t, joint_names=tuple(limits.joint_names), positions=q, velocities=qd, accelerations=qdd,
                               limits=limits, channels={"path_progress": progress})


# ====================================================================== the profile


def settings(limits: JointLimits, law: Any) -> dict[str, Any]:
    """TOPP-RA's limits and grid from the JointLimits handed in and the law's options, or a ValueError saying why not."""
    options = dict(law.options) if law is not None else {}
    torque = sorted(key for key in options if any(word in key for word in TORQUE_WORDS))
    if torque:
        raise ValueError(f"TOPP-RA's torque-limited mode ({torque}) is not served: it needs the space's dynamics, a core change: "
                         f"{NEEDED_CORE_CHANGE}")
    unknown = sorted(set(options) - set(OPTIONS))
    if unknown:
        raise ValueError(f"time law 'toppra' has no option {unknown} (it takes {OPTIONS})")
    fraction = float(options.get("acceleration_fraction", 1.0))
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"time law 'toppra': acceleration_fraction must lie in (0, 1], got {fraction}")
    grid = options.get("grid_points", GRID_POINTS)
    if isinstance(grid, bool) or not isinstance(grid, (int, np.integer)) or grid < 2:
        raise ValueError(f"time law 'toppra': grid_points must be an int >= 2, got {grid!r}")
    dof = len(limits.joint_names)
    velocity, acceleration = _finite_positive(limits.max_velocity, dof), _finite_positive(limits.max_acceleration, dof)
    if velocity is None or acceleration is None:
        raise ValueError("time law 'toppra' needs finite, positive max_velocity and max_acceleration in the JointLimits")
    return {"max_velocity": velocity, "max_acceleration": acceleration * fraction, "grid_points": int(grid)}


def time_optimal(path: np.ndarray, limits: JointLimits, rate_hz: float, law: Any) -> tuple[np.ndarray, ...]:
    """(t, q, qd, qdd, progress) along ``path`` (M, J), rest to rest, sampled at ``rate_hz`` from t = 0; progress = s, 0..M-1."""
    values = settings(limits, law)
    count, dof = path.shape
    floor = max(float(law.min_move_s), float(law.move_s or 0.0)) if law is not None else 0.1
    if float(np.max(np.abs(path - path[0]))) <= 1e-12:  # nowhere to go: a hold of the floor's length
        steps = max(2, int(math.ceil(floor * rate_hz - 1e-9)))
        zeros = np.zeros((steps + 1, dof))
        return (np.arange(steps + 1) / rate_hz, np.tile(path[0], (steps + 1, 1)), zeros, zeros.copy(),
                np.linspace(0.0, count - 1.0, steps + 1))
    # WHY s scaled to a rough duration, not the point index: toppra's solvers cap the squared path speed and the path
    # acceleration at 1e4 (toppra.constants MAXX, MAXU), so an index-scaled s (path speed in the hundreds of points per second)
    # is reported uncontrollable even on a straight line. With s ~ seconds both stay O(1/T).
    steps_between = np.abs(np.diff(path, axis=0))
    rough_s = max(float(np.sum(np.max(steps_between / values["max_velocity"], axis=1))),
                  math.sqrt(float(np.sum(np.max(steps_between / values["max_acceleration"], axis=1)))), 1e-3)
    s_points = np.linspace(0.0, rough_s, count)
    spline = toppra.SplineInterpolator(s_points, path)
    grid = np.union1d(np.linspace(0.0, rough_s, values["grid_points"]), s_points)
    velocity = toppra.constraint.JointVelocityConstraint(values["max_velocity"])
    acceleration = toppra.constraint.JointAccelerationConstraint(
        values["max_acceleration"], discretization_scheme=toppra.constraint.DiscretizationType.Interpolation)
    instance = toppra.algorithm.TOPPRA([velocity, acceleration], spline, gridpoints=grid, solver_wrapper="seidel")
    instance.compute_parameterization(0.0, 0.0)
    data = instance.problem_data
    if data.return_code != toppra.algorithm.ParameterizationReturnCode.Ok or data.sd_vec is None:
        raise ValueError(f"TOPP-RA could not time this path ({data.return_code}): the limits admit no rest-to-rest timing along it")
    s_grid = np.asarray(data.gridpoints, float).reshape(-1)
    sd = np.maximum(np.asarray(data.sd_vec, float).reshape(-1), 0.0)
    sd[0] = sd[-1] = 0.0  # WHY exact: rest to rest is the request; the solver returns ~1e-12 at the ends
    ds = np.diff(s_grid)
    interval_s = 2.0 * ds / np.maximum(sd[:-1] + sd[1:], 1e-12)
    times = np.concatenate([[0.0], np.cumsum(interval_s)])
    sdd = (sd[1:] ** 2 - sd[:-1] ** 2) / (2.0 * ds)
    duration = float(times[-1])

    def evaluate(tau: np.ndarray) -> tuple[np.ndarray, ...]:
        """The constant-path-acceleration profile at times ``tau`` (TOPP-RA's own time): s, q, qd, qdd."""
        tau = np.clip(tau, 0.0, duration)
        index = np.clip(np.searchsorted(times, tau, side="right") - 1, 0, len(ds) - 1)
        local = tau - times[index]
        s = np.clip(s_grid[index] + sd[index] * local + 0.5 * sdd[index] * local**2, s_grid[index], s_grid[index + 1])
        s_dot = np.maximum(sd[index] + sdd[index] * local, 0.0)
        slope, curvature = spline(s, 1), spline(s, 2)
        return s, spline(s), slope * s_dot[:, None], curvature * (s_dot**2)[:, None] + slope * sdd[index][:, None]

    # The limit check: every interval at CHECK_PER_INTERVAL points with that interval's own s-ddot (both ends included).
    fractions = np.linspace(0.0, 1.0, CHECK_PER_INTERVAL)
    worst_velocity = worst_acceleration = 0.0
    for start in range(0, len(ds), 512):
        rows = np.arange(start, min(start + 512, len(ds)))
        local = (interval_s[rows][:, None] * fractions[None, :]).reshape(-1)
        index = np.repeat(rows, CHECK_PER_INTERVAL)
        s = np.clip(s_grid[index] + sd[index] * local + 0.5 * sdd[index] * local**2, s_grid[index], s_grid[index + 1])
        s_dot = np.maximum(sd[index] + sdd[index] * local, 0.0)
        slope, curvature = spline(s, 1), spline(s, 2)
        qd_check = slope * s_dot[:, None]
        qdd_check = curvature * (s_dot**2)[:, None] + slope * sdd[index][:, None]
        worst_velocity = max(worst_velocity, float(np.max(np.abs(qd_check) / values["max_velocity"])))
        worst_acceleration = max(worst_acceleration, float(np.max(np.abs(qdd_check) / values["max_acceleration"])))
    # WHY slowed, never clipped: one factor c >= 1 on the clock keeps the path and divides qd by c, qdd by c^2
    # WHY the 1e-9 margin on the limits only: the floor (min_move_s, move_s) is a whole number of samples already, and a
    # margin on it would push a 0.1 s leg to 51 samples
    needed = max(1.0, worst_velocity * (1.0 + 1e-9), math.sqrt(worst_acceleration) * (1.0 + 1e-9), floor / duration)
    steps = max(2, int(math.ceil(needed * duration * rate_hz - 1e-9)))
    scale = steps / (rate_hz * duration)
    t = np.arange(steps + 1) / rate_hz
    s, q, qd, qdd = evaluate(t / scale)
    qd, qdd = qd / scale, qdd / scale**2
    progress = s * ((count - 1.0) / rough_s)  # the fractional point index (the points are evenly spaced in s)
    progress[0], progress[-1] = 0.0, count - 1.0
    q[0], q[-1] = path[0], path[-1]
    qd[0] = qd[-1] = 0.0
    return t, q, qd, qdd, progress


def _at_rest(state: Any) -> bool:
    return state is None or not (np.any(np.asarray(state.velocity) != 0.0) or np.any(np.asarray(state.acceleration) != 0.0))


def _finite_positive(values: Any, dof: int) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, float).reshape(-1)
    if array.shape != (dof,) or not np.all(np.isfinite(array)) or not np.all(array > 0):
        return None
    return array


def _position_violation(q: np.ndarray, limits: JointLimits) -> str:
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


__all__ = ["NAME", "NEEDED_CORE_CHANGE", "OPTIONS", "ToppraBackend", "settings", "time_optimal"]
