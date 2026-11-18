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

``law.options``: ``acceleration_fraction`` (of the stated limit, default 1), ``grid_points`` (TOPP-RA's grid, default 200,
the catalog row's number; the path's own points are always on it) and ``effort_fraction`` (the torque-limited mode, below).

**Torque-limited mode** (``effort_fraction`` in (0, 1]): the space's dynamics model's torque stays within that fraction of
``max_effort`` at every sample. ``needs(law)`` then names ``dynamics_model`` and ``torque_limits``, so the core hands the
space's model in (``time_parameterize(..., dynamics=space.dynamics)``) or refuses with ``MissingRung`` first. TOPP-RA gets a
``SecondOrderConstraint`` whose inverse dynamics is the model's own ``torque(q, qd, qdd)`` (or pin_utils'
``inverse_dynamics``), with -limit <= tau <= limit. Two guarantees are made here rather than assumed:

- **Inside the limit at every sample.** TOPP-RA reads the model by three probes and scales every velocity term with the
  squared path speed: exact for rigid-body form, ``M(q) qdd + qd' C(q) qd + g(q)``, wrong for friction (viscous, Coulomb).
  So each re-timing adds, as the constraint's ``custom_term``, what the model's torque is beyond that reading at the speeds
  the last timing ran (``_mismatch``: a sequential linearisation, zero for a rigid-body model). The model's torque is then
  checked at every output sample; after ``EFFORT_SETTLE_TRIES`` the bound is also tightened by any measured excess (between
  grid points, the slow-down). ``EFFORT_TRIES`` at most, then refused by name. On lane B's CAN 11 stand-in it settles in
  one to four tries (measured 2026-11-18).
- **Never faster than the kinematic timing.** A torque bound only removes options, so its TOPP-RA optimum is never shorter;
  but the uniform slow-down for between-grid peaks is not monotone, so the kinematic duration is also a floor. When the
  kinematic timing already keeps the torque inside the limit, it is the answer, sample for sample.

Holding still somewhere on the path already over the limit (gravity alone) is refused, naming the point.

Refused (``can_service``): a start or end that moves (this backend times rest to rest; minimum-jerk's quintic or Ruckig
serve moving boundaries), and an unknown or out-of-range option.
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
OPTIONS = ("acceleration_fraction", "grid_points", "effort_fraction")
GRID_POINTS = 200  # WHY 200: the catalog row time_optimal_path's own default
CHECK_PER_INTERVAL = 8  # samples per TOPP-RA interval on the limit check (both ends of each interval included)
EFFORT_TRIES = 12  # WHY 12: the CAN 11 stand-in (Coulomb + viscous) settles in two to five; the rest is headroom
EFFORT_SETTLE_TRIES = 2  # tries that only re-linearise (the mismatch) before the bound is also tightened by the measured excess
EFFORT_STEP_MARGIN = 0.05  # tighten 5 % past the measured excess, so the next try lands inside rather than on the limit

logging.getLogger("toppra").setLevel(logging.ERROR)  # WHY: toppra logs every solve at INFO/WARNING; refusals here say why


class ToppraBackend:
    """The time law ``toppra``: time-optimal along a fixed joint path under velocity and acceleration limits, rest to rest."""

    name = NAME

    def needs(self, law: Any) -> tuple[str, ...]:
        """Velocity and acceleration limits; in the torque-limited mode (``effort_fraction``) also the torque limits and the
        space's dynamics model (the core hands it in, or refuses with ``MissingRung``)."""
        kinematic = ("velocity_limits", "acceleration_limits")
        return (*kinematic, "torque_limits", "dynamics_model") if torque_limited(law) else kinematic

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
                          end_state: Any = None, dynamics: Any = None) -> JointTrajectory:
        refusal = self.can_service(path_points, limits, law, start_state, end_state)
        if refusal:
            raise ValueError(refusal)
        path = np.asarray(path_points, float)
        t, q, qd, qdd, progress = time_optimal(path, limits, rate_hz, law, dynamics=dynamics)
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
    effort_limit = None
    if "effort_fraction" in options:
        effort = options["effort_fraction"]
        if isinstance(effort, bool) or not isinstance(effort, (int, float, np.integer, np.floating)) or not 0.0 < effort <= 1.0:
            raise ValueError(f"time law 'toppra': effort_fraction must lie in (0, 1], got {effort!r}")
        max_effort = _finite_positive(limits.max_effort, dof)
        if max_effort is None:
            raise ValueError("time law 'toppra': effort_fraction needs a finite, positive max_effort in the JointLimits")
        effort_limit = max_effort * float(effort)
    return {"max_velocity": velocity, "max_acceleration": acceleration * fraction, "grid_points": int(grid),
            "effort_limit": effort_limit}


def torque_limited(law: Any) -> bool:
    """Whether the law asks for the torque-limited mode (an ``effort_fraction`` option)."""
    return law is not None and "effort_fraction" in dict(law.options)


def time_optimal(path: np.ndarray, limits: JointLimits, rate_hz: float, law: Any, *, dynamics: Any = None) -> tuple[np.ndarray, ...]:
    """(t, q, qd, qdd, progress) along ``path`` (M, J), rest to rest, sampled at ``rate_hz`` from t = 0; progress = s, 0..M-1.
    In the torque-limited mode (``effort_fraction``), ``dynamics`` is the space's model (the module docstring)."""
    values = settings(limits, law)
    count, dof = path.shape
    floor = max(float(law.min_move_s), float(law.move_s or 0.0)) if law is not None else 0.1
    limit = values["effort_limit"]
    torque = None if limit is None else torque_of(dynamics)
    if limit is not None:
        _refuse_static_overload(path, torque, limit, limits.joint_names)
    if float(np.max(np.abs(path - path[0]))) <= 1e-12:  # nowhere to go: a hold of the floor's length
        steps = max(2, int(math.ceil(floor * rate_hz - 1e-9)))
        zeros = np.zeros((steps + 1, dof))
        return (np.arange(steps + 1) / rate_hz, np.tile(path[0], (steps + 1, 1)), zeros, zeros.copy(),
                np.linspace(0.0, count - 1.0, steps + 1))
    kinematic, profile = _solve(path, values, rate_hz, floor)
    if limit is None:
        return kinematic
    excess = _effort_excess(kinematic, torque, limit)
    if np.all(excess <= 0.0):
        return kinematic  # WHY: the torque bound does not bind along this path, so the kinematic optimum is the answer
    # WHY the kinematic duration as a floor: a torque bound only removes options, so the result is never faster than it
    at_least = max(floor, float(kinematic[0][-1]))
    bound = limit.copy()
    for attempt in range(EFFORT_TRIES):
        # WHY the mismatch at the last speed profile: TOPP-RA scales every velocity term of the model with the squared path
        # speed (exact for rigid-body terms, wrong for friction); adding what the model says beyond that, at the speeds the last
        # timing ran, makes TOPP-RA's torque the model's own near that timing (sequential linearisation)
        mismatch = _mismatch(torque, *profile)
        try:
            timed, profile = _solve(path, values, rate_hz, at_least, effort=(torque, bound, mismatch))
        except ValueError as error:
            raise ValueError(f"TOPP-RA found no timing along this path that keeps the model's torque within effort_fraction x "
                             f"max_effort = {np.round(limit, 4).tolist()} N·m ({error}); near where the path turns against gravity "
                             "the model's friction leaves too little torque: raise effort_fraction or max_effort, or move the "
                             "path") from error
        excess = _effort_excess(timed, torque, limit)
        if np.all(excess <= 0.0):
            return timed
        if attempt >= EFFORT_SETTLE_TRIES:
            # WHY tightened by what the model measured: what remains is between grid points and from the slow-down
            bound = bound - np.maximum(excess, 0.0) * (1.0 + EFFORT_STEP_MARGIN)
            if np.any(bound <= 0.0):
                break
    joint = int(np.argmax(excess / limit))
    raise ValueError(f"TOPP-RA could not keep the model's torque on {limits.joint_names[joint]} within {limit[joint]:.4f} N·m "
                     f"(effort_fraction x max_effort) along this path in {EFFORT_TRIES} tries (still {excess[joint]:.4f} N·m "
                     "over): the model is far from TOPP-RA's form along it, or the bound is too tight")


def _mismatch(torque: Any, spline: Any, grid: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """(G, J): at each grid point, the model's torque at path speed ``speed`` (path acceleration 0) minus what TOPP-RA's
    SecondOrderConstraint reads from the model's three probes (c = tau(q, 0, 0), b = tau(q, q', q'') - c, so b sd^2 + c).
    Zero for a rigid-body model; the friction terms for the rest."""
    q, slope, curvature = spline(grid), spline(grid, 1), spline(grid, 2)
    zeros = np.zeros(q.shape[1])
    rows = []
    for point, tangent, bend, path_speed in zip(q, slope, curvature, speed, strict=True):
        at_rest = torque(point, zeros, zeros)
        read = (torque(point, tangent, bend) - at_rest) * path_speed**2 + at_rest
        rows.append(torque(point, tangent * path_speed, bend * path_speed**2) - read)
    return np.array(rows)


def torque_of(dynamics: Any) -> Any:
    """The model's inverse dynamics as ``tau(q, qd, qdd) -> (J,)``: its ``torque`` (or pin_utils' ``inverse_dynamics``)."""
    function = getattr(dynamics, "torque", None) or getattr(dynamics, "inverse_dynamics", None)
    if not callable(function):
        raise ValueError("time law 'toppra' with effort_fraction needs the space's dynamics, a model with torque(q, qd, qdd) "
                         "(the core hands it in as dynamics=space.dynamics)")

    def tau(q: Any, qd: Any, qdd: Any) -> np.ndarray:
        return np.asarray(function(np.asarray(q, float), np.asarray(qd, float), np.asarray(qdd, float)), float).reshape(-1)

    return tau


def _refuse_static_overload(path: np.ndarray, torque: Any, limit: np.ndarray, joint_names: Any) -> None:
    """Holding still at some path point already needs more than the limit (gravity alone): no timing can help, say where."""
    zeros = np.zeros(path.shape[1])
    static = np.array([np.abs(torque(point, zeros, zeros)) for point in path])
    point, joint = np.unravel_index(int(np.argmax(static / limit)), static.shape)
    if static[point, joint] > limit[joint]:
        raise ValueError(f"holding still at path point {point} already needs {static[point, joint]:.4f} N·m on "
                         f"{joint_names[joint]}, over effort_fraction x max_effort = {limit[joint]:.4f} N·m: no timing keeps the "
                         "torque inside it")


def _effort_excess(timed: tuple[np.ndarray, ...], torque: Any, limit: np.ndarray) -> np.ndarray:
    """Per joint, the model's worst |torque| over every output sample minus the limit (<= 0 everywhere: inside it)."""
    _, q, qd, qdd, _ = timed
    worst = np.max(np.abs(np.array([torque(*state) for state in zip(q, qd, qdd, strict=True)])), axis=0)
    return worst - limit


def _solve(path: np.ndarray, values: dict[str, Any], rate_hz: float, floor: float,
           effort: tuple[Any, np.ndarray, np.ndarray] | None = None) -> tuple[tuple[np.ndarray, ...], tuple[Any, ...]]:
    """TOPP-RA along ``path`` under the velocity and acceleration limits (and, with ``effort`` = (tau, bound, mismatch),
    -bound <= tau + mismatch <= bound by a SecondOrderConstraint on the model's inverse dynamics, ``mismatch`` (G, J) its
    custom term on the grid), sampled at the rate, at least ``floor`` long. Returns the samples (t, q, qd, qdd, progress) and
    the profile (spline, grid, path speed on the grid as sampled) the next mismatch is read at."""
    count, dof = path.shape
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
    constraints = [velocity, acceleration]
    if effort is not None:
        tau, bound, mismatch = effort
        both_ways = np.vstack([np.eye(dof), -np.eye(dof)])  # F w <= g with w = tau: tau <= bound and -tau <= bound
        limit_g = np.concatenate([bound, bound])

        def beyond_form(_path: Any, s: float) -> np.ndarray:  # TOPP-RA calls it at each grid point (``_mismatch``)
            return np.array([np.interp(s, grid, mismatch[:, joint]) for joint in range(dof)])

        # WHY the model's own inverse dynamics, not a fitted M, C, g: the model is injected by shape (torque(q, qd, qdd)), and
        # TOPP-RA reads a, b, c from it by its three probes; what those miss (friction) comes in as the custom term
        constraints.append(toppra.constraint.SecondOrderConstraint(
            tau, lambda _q: both_ways, lambda _q: limit_g, dof, custom_term=beyond_form,
            discretization_scheme=toppra.constraint.DiscretizationType.Interpolation))
    instance = toppra.algorithm.TOPPRA(constraints, spline, gridpoints=grid, solver_wrapper="seidel")
    instance.compute_parameterization(0.0, 0.0)
    data = instance.problem_data
    if data.return_code != toppra.algorithm.ParameterizationReturnCode.Ok or data.sd_vec is None:
        within = " and the torque bound" if effort is not None else ""
        raise ValueError(f"TOPP-RA could not time this path ({data.return_code}): the limits{within} admit no rest-to-rest timing "
                         "along it")
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
    return (t, q, qd, qdd, progress), (spline, s_grid, sd / scale)


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


__all__ = ["NAME", "OPTIONS", "ToppraBackend", "settings", "time_optimal", "torque_limited", "torque_of"]
