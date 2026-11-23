"""WaypointTrajectoryGenerator: points and joint limits in, a JointTrajectory out. ``Path x waypoint trajectory generator ->
JointTrajectory`` (Zach's refined spec, 2026-11-18). Trajectory generation is the stage; every kind is a ``TrajectoryGenerator``
(``trajectory_generator.py``: anything that outputs a robot_msgs Trajectory), and a waypoint trajectory generator is the one the
kinds hand their legs' timing to (MoveIt calls this role a time parameterization: a path and limits in, a timed trajectory out).

A sparse point set (the backend picks the curve between, as Ruckig does) and a dense path followed exactly (TOPP-RA, the
built-ins' ``along``) are the same type: following the path exactly is a property of a backend, not a separate class.

A path kind (a move, waypoints, the speed ladder, a Cartesian line) says WHERE the joints go; its ``waypoint_generator`` says
how each leg is timed:

    WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.6)  # peaks at 60 % of top speed, no joint over 60 % of its own
    WaypointTrajectoryGenerator.minimum_jerk(move_s=1.0)               # each leg takes 1 s (longer if a limit needs it)
    WaypointTrajectoryGenerator.trapezoid(peak_speed_fraction=0.8)     # accelerate, cruise, decelerate (a quarter of the leg a ramp)
    WaypointTrajectoryGenerator("ruckig", peak_speed_fraction=1.0)     # a plugin backend, when a package provides one (lane E)

Formerly ``TimeLaw``, then ``TimeParameterization`` (both renamed 2026-11-23); the old names still import, from ``time_law.py``
and ``time_parameterization.py``, as the same objects.

Inside every limit by construction: a leg never runs faster than its law asks, and never faster than the velocity AND
acceleration limits allow. When the acceleration limit binds, the leg peaks lower than asked and ``leg_timing`` says so (as
the pendulum ladder's index marks a ``clamped`` rung).

Backends by role (Zach, 2026-11-18: "for simple things you keep the implementation yourself but the other things other repos
can implement them ... like ruckig (and if you have the professional version we can support that as a drop in backend)").
Plugins are cut by what they do, not by how hard they are (prior-art note, 2026-11-18, section 7): a **waypoint trajectory
generator** takes points and limits and returns a timed trajectory (minimum-jerk, trapezoid, Ruckig, TOPP-RA); a **planner**
searches for a collision-free path; an **optimizer** chooses path and timing together. This module is the waypoint trajectory
generator's role: its ``shape`` names a ``WaypointTrajectoryGeneratorBackend``. Minimum-jerk and trapezoid are built in;
anything else is registered by another package under the entry-point group ``robot_trajectories.waypoint_trajectory_generators``
(``plugins.py``; the older groups ``robot_trajectories.time_parameterizations`` and ``robot_trajectories.time_laws`` are still
read, and ``robot_trajectories.planners`` and ``robot_trajectories.optimizers`` are reserved for the later roles). Every
backend hands back a robot_msgs JointTrajectory,
so whatever it speaks inside (Ruckig's InputParameter, TOPP-RA's spline), what comes out plays with everything else here.

The space's dynamics reach a backend that names ``dynamics_model`` in its ``needs`` (TOPP-RA's torque-limited mode): callers
hand ``dynamics=space.dynamics`` to ``leg`` / ``along``; without one the call is a ``MissingRung``. The built-in laws never read it.

Boundary states: a leg starts and ends at rest unless a start or end state (velocity, acceleration) is given. Minimum-jerk
then solves the quintic through both states (the minimum-jerk solution with those boundary conditions); trapezoid refuses
by name; a backend takes them as ``start_state`` / ``end_state``. That is what a velocity-matched sync (a moving target)
needs: a leg that ends moving with the target.

Ruckig Community must never be handed intermediate waypoints: Community sends those to Ruckig's cloud servers. A Ruckig
backend refuses a path of more than two points unless the local Pro build is verified (bam_traj_gen's
``ruckig_backend.require_local_ruckig_pro``: the ``ruckig-pro`` distribution, never beside Community, from the local checkout,
with the Pro-only API and no cloud markers in the native module), through ``can_service``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, TypeVar

import numpy as np
from robot_msgs.trajectory import JointLimits, JointTrajectory

from . import kernels as k
from .space import MissingRung

BUILT_IN = ("minimum_jerk", "trapezoid")
# The catalog's timing rows a plugin backend serves (lane E), named so a missing one is refused with the fix.
PLUGIN_SHAPES = {"ruckig": "jerk_limited_scurve", "toppra": "time_optimal_path"}
NOT_YET = ("jerk_limited_scurve", "polynomial_move", "time_optimal_path")  # catalog ids with no built-in law
WAYPOINT_TRAJECTORY_GENERATOR_GROUP = "robot_trajectories.waypoint_trajectory_generators"
TRAPEZOID_RAMP = 0.25  # WHY a quarter: Pilz PTP's and the textbook's default blend; a fraction, so the profile is one shape


# ====================================================================== the backend seam


class BoundaryState(NamedTuple):
    """A leg's velocity and acceleration at one end (each (J,)); ``None`` for a state means at rest."""

    velocity: np.ndarray
    acceleration: np.ndarray


class WaypointTrajectoryGeneratorBackend(Protocol):
    """The waypoint trajectory generator's role: points and limits in, a timed JointTrajectory out. ``name`` is the shape it
    serves (``WaypointTrajectoryGenerator(name, ...)``).

    ``time_parameterize(path_points, limits, rate_hz, law, *, start_state=None, end_state=None) -> JointTrajectory``:

    - ``path_points`` (M, J), M >= 2: the joint path, in order. M = 2 is one move; a dense M is a geometric path to follow
      (e.g. from inverse kinematics along a tool line); a backend that cannot follow a dense path exactly refuses it.
    - ``limits``: the space's JointLimits with ``max_velocity`` already scaled by ``law.peak_speed_fraction`` (acceleration
      and jerk as stated).
    - ``law``: the WaypointTrajectoryGenerator (``move_s``, ``min_move_s``, ``options`` for backend-specific settings).
    - ``start_state`` / ``end_state``: a ``BoundaryState`` (velocity, acceleration) or None (at rest, the default). A backend
      that cannot end moving refuses a non-rest state (through ``can_service``), never ignores it.
    - returns a JointTrajectory sampled at ``rate_hz`` from t = 0, positions, velocities and accelerations filled, starting
      at ``path_points[0]`` in ``start_state`` and ending at ``path_points[-1]`` in ``end_state``, passing every path point in
      order, inside ``limits``. Optional channel ``"path_progress"``: per sample, the fractional index into ``path_points``
      (Cartesian kinds then name the tool point commanded at every sample).

    Optional: ``can_service(path_points, limits, law, start_state, end_state) -> str`` returns "" when it can, else why not
    (MoveIt's canServiceRequest): a Ruckig Community backend returns a refusal for more than two points (they would go to
    Ruckig's cloud). ``needs``: a tuple in the catalog's needs vocabulary, or a method ``needs(law)`` returning one when they
    depend on the law (TOPP-RA: ``dynamics_model`` and ``torque_limits`` in its torque-limited mode only).

    **The space's dynamics** (``needs`` naming ``dynamics_model``): the caller hands ``dynamics=space.dynamics`` to
    ``WaypointTrajectoryGenerator.leg`` / ``WaypointTrajectoryGenerator.along`` (``legs_through``, ``cartesian.follow`` and the
    path kinds do), and the backend is called with ``time_parameterize(..., dynamics=model)``: the model as injected
    (``torque(q, qd, qdd)``, or pin_utils' ``inverse_dynamics``). No model, or ``torque_limits`` named and no finite
    ``max_effort``: ``MissingRung``, before the backend runs. The torque bound is the ``max_effort`` already in ``limits``; the
    caller then holds the model's torque along the result to it. A backend that does not name ``dynamics_model`` is never
    handed a model (its signature needs no ``dynamics``). The kind that called it guards the result anyway (limits, jumps,
    rest, collisions)."""

    name: str

    def time_parameterize(self, path_points: np.ndarray, limits: JointLimits, rate_hz: float, law: WaypointTrajectoryGenerator,
                          *, start_state: BoundaryState | None = None,
                          end_state: BoundaryState | None = None) -> JointTrajectory: ...


def backend_needs(serving: Any, law: WaypointTrajectoryGenerator) -> tuple[str, ...]:
    """What a backend needs for this law: its ``needs`` tuple, or what its ``needs(law)`` method returns (none when absent)."""
    needs = getattr(serving, "needs", ())
    return tuple(needs(law) if callable(needs) else needs)


@dataclass(frozen=True)
class BackendInfo:
    name: str
    backend: Any
    provider: str  # the distribution that registered it ("robot_trajectories" for the built-ins)


_BACKENDS: dict[str, list[BackendInfo]] = {}


def register_waypoint_trajectory_generator(backend: Any, *, provider: str = "robot_trajectories") -> None:
    """Register a waypoint trajectory generator's backend (what an entry point in
    ``robot_trajectories.waypoint_trajectory_generators``, or in the older ``robot_trajectories.time_parameterizations`` or
    ``robot_trajectories.time_laws``, loads; callable directly too)."""
    name = getattr(backend, "name", None)
    if not isinstance(name, str) or not name.isidentifier() or not callable(getattr(backend, "time_parameterize", None)):
        raise TypeError(f"a waypoint trajectory generator's backend needs a name (an identifier) and "
                        f"time_parameterize(path_points, limits, rate_hz, law); {backend!r} from {provider} has neither")
    entries = _BACKENDS.setdefault(name, [])
    if not any(entry.provider == provider for entry in entries):
        entries.append(BackendInfo(name, backend, provider))


def waypoint_trajectory_generators() -> dict[str, tuple[BackendInfo, ...]]:
    """Every registered waypoint trajectory generator's name with its providers (more than one provider for a name is refused
    at use)."""
    return {name: tuple(entries) for name, entries in _BACKENDS.items()}


def backend(name: str) -> Any:
    """The one backend serving ``name``, or a refusal that names the fix.

    WHY refuse two providers instead of ranking them (bam_traj_gen ruckig_backend.require_local_ruckig_pro): Ruckig Community
    and Ruckig Pro both import as ``ruckig``, so which one runs would depend on install order, and Community's intermediate
    waypoints call a cloud API. A Pro plugin is a drop-in for the same name; installing both is ambiguous, so it is refused."""
    entries = _BACKENDS.get(name)
    if not entries:
        from . import plugins

        plugins.load()
        entries = _BACKENDS.get(name)
    if not entries:
        row = PLUGIN_SHAPES.get(name)
        raise ValueError(f"waypoint trajectory generator {name!r} has no backend here (built in: {BUILT_IN}; from plugins: "
                         f"{tuple(sorted(set(_BACKENDS) - set(BUILT_IN)))}): install a package that registers a {name!r} "
                         f"backend in the {WAYPOINT_TRAJECTORY_GENERATOR_GROUP!r} entry-point group"
                         + (f" (catalog row {row}; docs/KINDS_TODO.md lane E)" if row else ""))
    if len(entries) > 1:
        raise ValueError(f"waypoint trajectory generator {name!r} is provided by {[entry.provider for entry in entries]}: "
                         "ambiguous (a drop-in replaces, it does not sit beside); uninstall all but one")
    return entries[0].backend


# ====================================================================== the law


@dataclass(frozen=True)
class LegTiming:
    """How one leg was timed: its duration, what bounded it, and the peak speed it reaches."""

    duration_s: float
    bound: str  # "speed" (the law's own number), "move_s", "acceleration", "velocity" or "min_move_s"
    peak_speed_fraction: float  # the worst joint's peak speed over its own top speed, as generated

    @property
    def clamped(self) -> bool:
        return self.bound in ("acceleration", "velocity", "min_move_s")


@dataclass(frozen=True)
class Timed:
    """A timed path: samples at the rate, and (when the backend says) each sample's progress along the path, 0..1."""

    arrays: k.Arrays
    progress: np.ndarray | None
    timing: LegTiming | None


@dataclass(frozen=True)
class WaypointTrajectoryGenerator:
    """Points and joint limits in, a JointTrajectory out: the shape of each leg's progress s(t) (a backend's name), and either
    its peak speed or its duration. ``leg`` times one move, ``along`` a path of points; the kinds hand it their legs."""

    shape: str = "minimum_jerk"
    peak_speed_fraction: float | None = None  # no joint's peak passes this fraction of its own top speed
    move_s: float | None = None  # a fixed time per leg instead
    min_move_s: float = 0.1  # WHY 0.1 s: ten 100 Hz plan ticks, the ladder's MIN_MOVE_S; a shorter leg is not smooth on the rig
    ramp_fraction: float = TRAPEZOID_RAMP  # trapezoid only: the fraction of the leg spent accelerating (and decelerating)
    options: tuple[tuple[str, Any], ...] = field(default=())  # backend-specific settings, as sorted (key, value) pairs

    def __post_init__(self) -> None:
        if isinstance(self.options, Mapping):
            object.__setattr__(self, "options", tuple(sorted((str(key), value) for key, value in self.options.items())))
        if self.shape not in BUILT_IN:
            backend(self.shape)  # refuses an unknown shape now, naming the plugin to install
        if (self.peak_speed_fraction is None) == (self.move_s is None):
            raise ValueError("a waypoint trajectory generator takes peak_speed_fraction= or move_s=, exactly one")
        if self.peak_speed_fraction is not None and not 0.0 < self.peak_speed_fraction <= 1.0:
            raise ValueError(f"peak_speed_fraction must lie in (0, 1], got {self.peak_speed_fraction}")
        if self.move_s is not None and self.move_s <= 0.0:
            raise ValueError(f"move_s must be positive, got {self.move_s}")
        if not 0.0 < self.ramp_fraction <= 0.5:
            raise ValueError(f"ramp_fraction must lie in (0, 0.5], got {self.ramp_fraction}")

    @classmethod
    def minimum_jerk(cls, *, peak_speed_fraction: float | None = None, move_s: float | None = None,
                     min_move_s: float = 0.1) -> WaypointTrajectoryGenerator:
        """Minimum-jerk legs x = x0 + d (10 s^3 - 15 s^4 + 6 s^5), peaking at 1.875 d / T."""
        return cls("minimum_jerk", peak_speed_fraction=peak_speed_fraction, move_s=move_s, min_move_s=min_move_s)

    @classmethod
    def trapezoid(cls, *, peak_speed_fraction: float | None = None, move_s: float | None = None,
                  ramp_fraction: float = TRAPEZOID_RAMP, min_move_s: float = 0.1) -> WaypointTrajectoryGenerator:
        """Accelerate for ``ramp_fraction`` of the leg, cruise, decelerate as long (a triangle at 0.5): Pilz PTP's profile.
        Its acceleration steps at the ramp ends (unlimited jerk); minimum-jerk is the smooth one."""
        return cls("trapezoid", peak_speed_fraction=peak_speed_fraction, move_s=move_s, min_move_s=min_move_s,
                   ramp_fraction=ramp_fraction)

    @property
    def built_in(self) -> bool:
        return self.shape in BUILT_IN

    # ------------------------------------------------------------------ the normalised profile s(tau), tau in [0, 1]

    def profile(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(s, ds/dtau, d2s/dtau2) of a built-in shape: progress 0 -> 1, at rest at both ends."""
        tau = np.clip(np.asarray(tau, float), 0.0, 1.0)
        if self.shape == "minimum_jerk":
            return (10 * tau**3 - 15 * tau**4 + 6 * tau**5, 30 * tau**2 - 60 * tau**3 + 30 * tau**4, 60 * tau - 180 * tau**2 + 120 * tau**3)
        if self.shape != "trapezoid":
            raise ValueError(f"{self.shape!r} is a plugin backend: it has no built-in profile")
        ramp = self.ramp_fraction
        acceleration = 1.0 / (ramp * (1.0 - ramp))
        cruise = acceleration * ramp
        rising, falling = tau < ramp, tau > 1.0 - ramp
        s = np.where(rising, 0.5 * acceleration * tau**2,
                     np.where(falling, 1.0 - 0.5 * acceleration * (1.0 - tau) ** 2, 0.5 * acceleration * ramp**2 + cruise * (tau - ramp)))
        ds = np.where(rising, acceleration * tau, np.where(falling, acceleration * (1.0 - tau), cruise))
        dds = np.where(rising, acceleration, np.where(falling, -acceleration, 0.0))
        return s, ds, dds

    def peaks(self) -> tuple[float, float]:
        """(peak ds/dtau, peak |d2s/dtau2|) of a built-in shape: the multiples of d / T and d / T^2 a straight leg peaks at."""
        if self.shape == "minimum_jerk":
            return k.MIN_JERK_PEAK_VELOCITY, k.MIN_JERK_PEAK_ACCELERATION
        ramp = self.ramp_fraction
        return 1.0 / (1.0 - ramp), 1.0 / (ramp * (1.0 - ramp))

    # ------------------------------------------------------------------ timing

    def leg_timing(self, q0: np.ndarray, q1: np.ndarray, bounds: Any) -> LegTiming:
        """The leg's duration: the law's, raised until every joint is inside its velocity and acceleration limits."""
        distance = np.abs(np.asarray(q1, float) - np.asarray(q0, float))
        if not np.any(distance > 1e-12):
            return LegTiming(self.min_move_s, "min_move_s", 0.0)
        if self.shape == "minimum_jerk":
            # v6's lines, unchanged (tests/test_equivalence.py holds the motion bit for bit)
            by_velocity = float(np.max(k.MIN_JERK_PEAK_VELOCITY * distance / bounds.max_velocity))
            by_acceleration = float(np.max(np.sqrt(k.MIN_JERK_PEAK_ACCELERATION * distance / bounds.max_acceleration)))
            if self.peak_speed_fraction is not None:
                # WHY per joint: "60 % of top speed" on a multi-joint leg means no joint passes 60 % of its own top speed.
                wanted, why = float(np.max(k.MIN_JERK_PEAK_VELOCITY * distance / (self.peak_speed_fraction * bounds.max_velocity))), "speed"
            else:
                wanted, why = float(self.move_s), "move_s"
            return self._settle(wanted, why, by_velocity, by_acceleration, distance * k.MIN_JERK_PEAK_VELOCITY, bounds)
        peak_v, peak_a = self.peaks()
        return self.timing_for(peak_v * distance, peak_a * distance, bounds)

    def timing_for(self, velocity: np.ndarray, acceleration: np.ndarray, bounds: Any) -> LegTiming:
        """A path's duration from its per-joint peaks at T = 1 (``velocity`` = max |dq/dtau|, ``acceleration`` = max
        |d2q/dtau2|): the law's, raised until every joint is inside its limits. A straight leg is ``leg_timing``."""
        velocity, acceleration = np.asarray(velocity, float), np.asarray(acceleration, float)
        if not np.any(velocity > 1e-12):
            return LegTiming(self.min_move_s, "min_move_s", 0.0)
        by_velocity = float(np.max(velocity / bounds.max_velocity))
        by_acceleration = float(np.max(np.sqrt(acceleration / bounds.max_acceleration)))
        if self.peak_speed_fraction is not None:
            wanted, why = float(np.max(velocity / (self.peak_speed_fraction * bounds.max_velocity))), "speed"
        else:
            wanted, why = float(self.move_s), "move_s"
        return self._settle(wanted, why, by_velocity, by_acceleration, velocity, bounds)

    def _settle(self, wanted: float, why: str, by_velocity: float, by_acceleration: float, velocity: np.ndarray,
                bounds: Any) -> LegTiming:
        # WHY the law wins ties: a leg bounded exactly by what was asked is not "clamped"
        candidates = [(wanted, why), (by_velocity, "velocity"), (by_acceleration, "acceleration"), (self.min_move_s, "min_move_s")]
        duration, bound = max(candidates, key=lambda item: (item[0], item[1] == why))
        if duration <= wanted * (1 + 1e-12):
            bound = why
        # WHY rounded up to the ms (pendulum_ladder.move_time): the peak never exceeds what was asked
        duration = math.ceil(duration * 1000.0 - 1e-6) / 1000.0
        peak = float(np.max(velocity / duration / bounds.max_velocity))
        return LegTiming(duration, bound, peak)

    # ------------------------------------------------------------------ one leg, and a path

    def __call__(self, points: np.ndarray, limits: JointLimits, rate_hz: float, *, start_state: BoundaryState | None = None,
                 end_state: BoundaryState | None = None, dynamics: Any = None) -> JointTrajectory:
        """Points (M, J) and joint limits in, a JointTrajectory out. Two points are one leg (``leg``, boundary states allowed);
        more are a path timed rest to rest (``along``, with each sample's ``path_progress`` in points, as the built-in backend
        records it).

        WHY callable (Zach, 2026-11-23: "All TrajectoryGenerators output Trajectories"): it makes every
        WaypointTrajectoryGenerator a ``TrajectoryGenerator`` in the literal isinstance sense, not only in prose."""
        points = np.asarray(points, float)
        if points.ndim != 2 or len(points) < 2:
            raise ValueError(f"points are (M, J) with M >= 2, got shape {points.shape}")
        joint_limits = _joint_limits(limits)
        if len(points) == 2:
            arrays = self.leg(points[0], points[1], limits, rate_hz, start_state=start_state, end_state=end_state,
                              dynamics=dynamics)
            progress = None
        else:
            if not (_at_rest(start_state) and _at_rest(end_state)):
                raise ValueError("a path of more than two points is timed rest to rest; hand boundary states to a two-point leg")
            timed = self.along(points, limits, rate_hz, dynamics=dynamics)
            arrays = timed.arrays
            progress = None if timed.progress is None else timed.progress * (len(points) - 1)
        channels = {} if progress is None else {"path_progress": progress}
        return JointTrajectory(t=arrays.t, joint_names=tuple(joint_limits.joint_names), positions=arrays.q,
                               velocities=arrays.qd, accelerations=arrays.qdd, limits=joint_limits, channels=channels)

    def leg(self, q0: np.ndarray, q1: np.ndarray, limits: Any, rate_hz: float, *, label: str = "move",
            start_state: BoundaryState | None = None, end_state: BoundaryState | None = None, dynamics: Any = None) -> k.Arrays:
        """One leg q0 -> q1 under this law, at rest at both ends unless ``start_state`` / ``end_state`` say otherwise.
        ``limits``: the space's JointLimits (or kernels.Bounds). ``dynamics``: the space's model (``space.dynamics``), handed to
        a backend that needs it (TOPP-RA's torque mode); the built-in laws never read it."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        moving = not (_at_rest(start_state) and _at_rest(end_state))
        if moving and self.shape == "minimum_jerk":
            return self._quintic(q0, q1, limits, rate_hz, label, start_state, end_state)
        if moving and self.shape == "trapezoid":
            raise ValueError("a trapezoid leg is rest-to-rest; a leg that starts or ends moving takes "
                             "WaypointTrajectoryGenerator.minimum_jerk (the quintic through both states) or a backend that serves "
                             "boundary states")
        if self.shape == "minimum_jerk":
            timing = self.leg_timing(q0, q1, limits)
            return k.min_jerk(q0, q1, limits, rate_hz, duration_s=timing.duration_s, label=label)
        if self.shape == "trapezoid":
            timing = self.leg_timing(q0, q1, limits)
            t, s, ds, dds = self._sampled(timing.duration_s, rate_hz)
            delta = (q1 - q0)[None, :]
            return k.Arrays(t, q0 + delta * s[:, None], delta * ds[:, None], delta * dds[:, None], label)
        timed = self._by_backend(np.stack([q0, q1]), limits, rate_hz, label, start_state, end_state, dynamics)
        return timed.arrays

    def _quintic(self, q0: np.ndarray, q1: np.ndarray, limits: Any, rate_hz: float, label: str,
                 start_state: BoundaryState | None, end_state: BoundaryState | None) -> k.Arrays:
        """The minimum-jerk leg between two states: the quintic matching position, velocity and acceleration at both ends, over
        the shortest duration that keeps every joint inside its limits. The law's speed caps the joints the way it caps a
        rest-to-rest leg, but never below the boundary speeds themselves; ``move_s`` is a floor on the duration."""
        zeros = np.zeros_like(q0)
        v0, a0 = (zeros, zeros) if start_state is None else (np.asarray(start_state.velocity, float), np.asarray(start_state.acceleration, float))
        v1, a1 = (zeros, zeros) if end_state is None else (np.asarray(end_state.velocity, float), np.asarray(end_state.acceleration, float))
        top = np.asarray(limits.max_velocity, float)
        cap = top if self.peak_speed_fraction is None else np.minimum(top, np.maximum(self.peak_speed_fraction * top, np.maximum(np.abs(v0), np.abs(v1))))
        floor = max(self.min_move_s, self.move_s or 0.0)
        tau = np.linspace(0.0, 1.0, 401)
        duration = floor
        for _ in range(600):  # WHY a 2 % search from the floor up: the peak of a quintic with moving ends is not monotonic in T
            steps = max(2, int(math.ceil(duration * rate_hz - 1e-9)))
            total = steps / rate_hz
            _, qd, qdd = _quintic_samples(q0, v0, a0, q1, v1, a1, total, tau * total)
            if np.all(np.abs(qd) <= cap * (1 - 1e-9)) and np.all(np.abs(qdd) <= np.asarray(limits.max_acceleration) * (1 - 1e-9)):
                t = np.arange(steps + 1) / rate_hz
                q, qd, qdd = _quintic_samples(q0, v0, a0, q1, v1, a1, total, t)
                q[0], q[-1] = q0, q1
                return k.Arrays(t, q, qd, qdd, label)
            duration *= 1.02
        raise ValueError(f"no minimum-jerk leg from {np.round(np.degrees(q0), 2).tolist()}° to {np.round(np.degrees(q1), 2).tolist()}° "
                         "between those boundary states fits inside the limits; lower the boundary speeds or move the point")

    def along(self, path: np.ndarray, limits: Any, rate_hz: float, *, label: str = "path",
              progress_grid: np.ndarray | None = None, dynamics: Any = None) -> Timed:
        """A dense joint path (M, J), followed from its first point to its last at rest at both ends under this law.
        ``dynamics``: the space's model, handed to a backend that needs it (as ``leg``).

        Built-in shapes: the path is q(s) on ``progress_grid`` (default uniform 0..1); its derivatives set the duration (every
        joint inside its velocity and acceleration limits, 1 % kept for interpolation), positions are cubic Hermite in s,
        velocities q'(s) s-dot, accelerations q''(s) s-dot^2 + q'(s) s-ddot. ``progress`` is s at every sample."""
        path = np.asarray(path, float)
        if path.ndim != 2 or len(path) < 2:
            raise ValueError(f"a path is (M, J) with M >= 2 points, got shape {path.shape}")
        if not self.built_in:
            return self._by_backend(path, limits, rate_hz, label, None, None, dynamics)
        grid = np.linspace(0.0, 1.0, len(path)) if progress_grid is None else np.asarray(progress_grid, float)
        if len(path) == 2:
            slope = np.tile(path[1] - path[0], (2, 1))
            curvature = np.zeros_like(path)
        else:
            slope = np.gradient(path, grid, axis=0, edge_order=2)
            curvature = np.gradient(slope, grid, axis=0, edge_order=2)
        tau = np.linspace(0.0, 1.0, 2001)
        s, ds, dds = self.profile(tau)
        slope_at, curvature_at = _interp_rows(s, grid, slope), _interp_rows(s, grid, curvature)
        velocity = np.max(np.abs(slope_at * ds[:, None]), axis=0)
        acceleration = np.max(np.abs(curvature_at * ds[:, None] ** 2 + slope_at * dds[:, None]), axis=0)
        timing = self.timing_for(1.01 * velocity, 1.01 * acceleration, limits)
        t, s, sdot, sddot = self._sampled(timing.duration_s, rate_hz)
        q, slope_at = _hermite(path, slope, grid, s)
        curvature_at = _interp_rows(s, grid, curvature)
        q[0], q[-1] = path[0], path[-1]
        qd = slope_at * sdot[:, None]
        qdd = curvature_at * sdot[:, None] ** 2 + slope_at * sddot[:, None]
        return Timed(k.Arrays(t, q, qd, qdd, label), s, timing)

    def _sampled(self, duration_s: float, rate_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The kernels' grid (a whole number of samples, ending at or after ``duration_s``) and the profile on it."""
        steps = max(2, int(math.ceil(duration_s * rate_hz - 1e-9)))
        t = np.arange(steps + 1) / rate_hz
        total = t[-1]
        s, ds, dds = self.profile(t / total)
        return t, s, ds / total, dds / total**2

    def _by_backend(self, path: np.ndarray, limits: Any, rate_hz: float, label: str, start_state: BoundaryState | None,
                    end_state: BoundaryState | None, dynamics: Any = None) -> Timed:
        joint_limits = _joint_limits(limits)
        if self.peak_speed_fraction is not None:
            joint_limits = joint_limits.replace(max_velocity=joint_limits.max_velocity * self.peak_speed_fraction)
        serving = backend(self.shape)
        needs = backend_needs(serving, self)
        handed: dict[str, Any] = {}
        if "dynamics_model" in needs:
            if dynamics is None:
                # WHY refused here, before the backend runs: a torque-limited timing without the model would be a guess
                raise MissingRung(f"waypoint trajectory generator {self.shape!r}{_mode(self)}", "dynamics_model",
                                  detail=" (none was handed to the waypoint trajectory generator: a kind passes dynamics=space.dynamics)")
            handed["dynamics"] = dynamics
        if "torque_limits" in needs and not _finite_positive(joint_limits.max_effort):
            raise MissingRung(f"waypoint trajectory generator {self.shape!r}{_mode(self)}", "torque_limits")
        refusal = serving.can_service(path, joint_limits, self, start_state, end_state) if hasattr(serving, "can_service") else ""
        if refusal:
            raise ValueError(f"waypoint trajectory generator {self.shape!r} refuses this request: {refusal}")
        out = serving.time_parameterize(path, joint_limits, rate_hz, self, start_state=start_state, end_state=end_state, **handed)
        if not isinstance(out, JointTrajectory) or out.velocities is None or out.accelerations is None:
            raise ValueError(f"backend {self.shape!r} broke the contract: return a JointTrajectory with velocities and accelerations")
        t = np.asarray(out.t, float)
        if not np.allclose(t, np.arange(len(t)) / rate_hz, atol=1e-9):
            raise ValueError(f"backend {self.shape!r} broke the contract: sample at {rate_hz:g} Hz from t = 0")
        q = np.array(out.positions, float)
        for which, row, state in (("start", 0, start_state), ("end", -1, end_state)):
            wanted = np.zeros(path.shape[1]) if state is None else np.asarray(state.velocity, float)
            if not np.allclose(np.asarray(out.velocities)[row], wanted, atol=1e-9):
                raise ValueError(f"backend {self.shape!r} broke the contract: its {which} velocity is not the {which} state's")
            if not np.allclose(q[row], path[row], atol=1e-9):
                raise ValueError(f"backend {self.shape!r} broke the contract: it does not {which} at the path's {which}")
        q[0], q[-1] = path[0], path[-1]
        if "dynamics" in handed and _finite_positive(joint_limits.max_effort):
            _hold_to_max_effort(self.shape, handed["dynamics"], q, out, joint_limits)
        progress = out.channels.get("path_progress")
        progress = None if progress is None else np.asarray(progress, float) / (len(path) - 1)
        return Timed(k.Arrays(np.arange(len(t)) / rate_hz, q, np.asarray(out.velocities, float), np.asarray(out.accelerations, float),
                              label), progress, None)

    # ------------------------------------------------------------------ as configuration

    def to_config(self) -> dict[str, Any]:
        # WHY "time_law" through both renames: recipes and arm_motion_bench's frozen datasets hold this key (registry.py)
        config: dict[str, Any] = {"time_law": self.shape}
        if self.peak_speed_fraction is not None:
            config["peak_speed_fraction"] = self.peak_speed_fraction
        if self.move_s is not None:
            config["move_s"] = self.move_s
        config["min_move_s"] = self.min_move_s
        if self.shape == "trapezoid":
            config["ramp_fraction"] = self.ramp_fraction
        if self.options:
            config["options"] = dict(self.options)
        return config

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> WaypointTrajectoryGenerator:
        values = dict(config)
        return cls(values.pop("time_law"), **values)

    def __repr__(self) -> str:
        parts = [f"peak_speed_fraction={self.peak_speed_fraction!r}" if self.peak_speed_fraction is not None else f"move_s={self.move_s!r}"]
        if self.min_move_s != 0.1:
            parts.append(f"min_move_s={self.min_move_s!r}")
        if self.shape == "trapezoid" and self.ramp_fraction != TRAPEZOID_RAMP:
            parts.append(f"ramp_fraction={self.ramp_fraction!r}")
        if self.options:
            parts.append(f"options={dict(self.options)!r}")
        if self.built_in:
            return f"WaypointTrajectoryGenerator.{self.shape}({', '.join(parts)})"
        return f"WaypointTrajectoryGenerator({self.shape!r}, {', '.join(parts)})"


def legs_through(start: np.ndarray, waypoints: Sequence[np.ndarray], law: WaypointTrajectoryGenerator, limits: Any, rate_hz: float, *,
                 dwell_s: float, label: str, dynamics: Any = None) -> list[k.Arrays]:
    """Rest-to-rest legs from ``start`` through each waypoint under ``law``, with a dwell at each: the house leg of every
    waypoints tape (the rack runner's ``q:move_s:dwell_s;...``; v6's Move, unchanged). ``dynamics``: the space's model, for a
    law whose backend needs it (pass ``space.dynamics``)."""
    pieces: list[k.Arrays] = []
    here = np.asarray(start, float)
    for index, target in enumerate(waypoints, start=1):
        target = np.asarray(target, float)
        pieces.append(law.leg(here, target, limits, rate_hz, label=f"{label}: leg {index}", dynamics=dynamics))
        if dwell_s > 0:
            pieces.append(k.hold(target, dwell_s, rate_hz, label=f"{label}: dwell"))
        here = target
    return pieces


# ====================================================================== built-in backends (the same Protocol a plugin meets)


@dataclass(frozen=True)
class BuiltInBackend:
    """Minimum-jerk or trapezoid as a WaypointTrajectoryGeneratorBackend: a two-point path is one leg, a dense path is followed as
    q(s)."""

    name: str

    def can_service(self, path_points: np.ndarray, limits: JointLimits, law: WaypointTrajectoryGenerator,
                    start_state: BoundaryState | None, end_state: BoundaryState | None) -> str:
        moving = not (_at_rest(start_state) and _at_rest(end_state))
        if moving and self.name == "trapezoid":
            return "a trapezoid is rest-to-rest"
        if moving and len(path_points) > 2:
            return "a dense path is followed rest-to-rest; boundary states apply to a two-point leg"
        return ""

    def time_parameterize(self, path_points: np.ndarray, limits: JointLimits, rate_hz: float, law: WaypointTrajectoryGenerator, *,
                          start_state: BoundaryState | None = None, end_state: BoundaryState | None = None) -> JointTrajectory:
        refusal = self.can_service(path_points, limits, law, start_state, end_state)
        if refusal:
            raise ValueError(refusal)
        # WHY the law re-made without the speed fraction: the limits handed in are already scaled by it
        plain = WaypointTrajectoryGenerator(self.name, peak_speed_fraction=1.0 if law.move_s is None else None, move_s=law.move_s,
                                     min_move_s=law.min_move_s, ramp_fraction=law.ramp_fraction)
        path = np.asarray(path_points, float)
        if len(path) == 2:
            arrays = plain.leg(path[0], path[1], limits, rate_hz, start_state=start_state, end_state=end_state)
            progress = None
        else:
            timed = plain.along(path, limits, rate_hz)
            arrays, progress = timed.arrays, timed.progress * (len(path) - 1)
        channels = {} if progress is None else {"path_progress": progress}
        return JointTrajectory(t=arrays.t, joint_names=tuple(limits.joint_names), positions=arrays.q, velocities=arrays.qd,
                               accelerations=arrays.qdd, limits=limits, channels=channels)


for _name in BUILT_IN:
    register_waypoint_trajectory_generator(BuiltInBackend(_name))


# ====================================================================== the renames of 2026-11-23: the old keywords

F = TypeVar("F", bound=Callable[..., Any])
# WHY (Zach, 2026-11-23): "time law" was an agent's coinage he never used, so the role became a time parameterization (MoveIt's
# name), and then, the same day, a waypoint trajectory generator: "lets give it the name TrajectoryGenerator ... these ones are
# specially WaypointTrajectoryGenerators. All TrajectoryGenerators output Trajectories". The older keywords are deprecated
# aliases, accepted silently: arm_motion_bench calls kinds with them, and a recipe still records these parameters under the
# OLDEST key (registry.RECORDED_AS), so its 21 frozen dataset files rebuild unchanged.
# Every name each renamed keyword has had, oldest first; the last is today's.
KEYWORD_NAMES = (("time_law", "time_parameterization", "waypoint_generator"),
                 ("approach_time_law", "approach_time_parameterization", "approach_waypoint_generator"))
RENAMED_KEYWORDS = {old: names[-1] for names in KEYWORD_NAMES for old in names[:-1]}  # each older name -> today's


def renamed_keywords(**old_to_new: str) -> Callable[[F], F]:
    """Accept each old keyword (a deprecated alias) as its new one: the same call, silently. Several old names may map to one
    new name; more than one name of the same parameter at once is a ``TypeError``. ``functools.wraps`` keeps
    ``inspect.signature`` (and a kind's ``parameters``) on the new names."""
    olds_of = {new: [old for old, to in old_to_new.items() if to == new] for new in old_to_new.values()}

    def decorate(function: F) -> F:
        @functools.wraps(function)
        def call(*args: Any, **kwargs: Any) -> Any:
            for new, olds in olds_of.items():
                given = [name for name in (new, *olds) if name in kwargs]
                if len(given) > 1:
                    spelled = " and ".join(f"{name}=" for name in given)
                    raise TypeError(f"{function.__name__}() got {spelled}: one parameter under {len(given)} of its names "
                                    f"(the others are aliases); pass {new}= alone")
                if given and given[0] != new:
                    kwargs[new] = kwargs.pop(given[0])
            return function(*args, **kwargs)

        return call  # type: ignore[return-value]

    return decorate


# ====================================================================== helpers


def _finite_positive(values: Any) -> bool:
    return values is not None and bool(np.all(np.isfinite(np.asarray(values, float)))) and bool(np.all(np.asarray(values, float) > 0))


def _mode(law: WaypointTrajectoryGenerator) -> str:
    return f" with options {dict(law.options)}" if law.options else ""


def _hold_to_max_effort(shape: str, dynamics: Any, q: np.ndarray, out: JointTrajectory, limits: JointLimits) -> None:
    """The model's torque along a backend's result, held to ``max_effort`` at every sample: the caller's own check of a backend
    that was handed the dynamics, as it checks the ends and the rate (an independent ruler, never the backend's word)."""
    function = getattr(dynamics, "torque", None) or dynamics.inverse_dynamics
    velocities, accelerations = np.asarray(out.velocities, float), np.asarray(out.accelerations, float)
    limit = np.asarray(limits.max_effort, float)
    for row in range(len(q)):
        torque = np.abs(np.asarray(function(q[row], velocities[row], accelerations[row]), float).reshape(-1))
        if np.any(torque > limit * (1 + 1e-6)):
            joint = int(np.argmax(torque / limit))
            raise ValueError(f"backend {shape!r} broke the contract: at sample {row} the model needs {torque[joint]:.4f} N·m on "
                             f"{limits.joint_names[joint]}, over its max_effort {limit[joint]:.4f} N·m")


def _at_rest(state: BoundaryState | None) -> bool:
    return state is None or not (np.any(np.asarray(state.velocity) != 0.0) or np.any(np.asarray(state.acceleration) != 0.0))


def _quintic_samples(q0, v0, a0, q1, v1, a1, total: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The quintic through (q0, v0, a0) at 0 and (q1, v1, a1) at ``total``, at times ``t``: (q, qd, qdd), each (N, J)."""
    d, T = q1 - q0, total
    c3 = (20 * d - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * T**2) / (2 * T**3)
    c4 = (-30 * d + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * T**2) / (2 * T**4)
    c5 = (12 * d - 6 * (v1 + v0) * T + (a1 - a0) * T**2) / (2 * T**5)
    t = np.asarray(t, float)[:, None]
    q = q0 + v0 * t + 0.5 * a0 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    qd = v0 + a0 * t + 3 * c3 * t**2 + 4 * c4 * t**3 + 5 * c5 * t**4
    qdd = a0 + 6 * c3 * t + 12 * c4 * t**2 + 20 * c5 * t**3
    return q, qd, qdd


def _joint_limits(limits: Any) -> JointLimits:
    if isinstance(limits, JointLimits):
        return limits
    return JointLimits(joint_names=tuple(limits.joint_names), min_position=limits.low, max_position=limits.high,
                       max_velocity=limits.max_velocity, max_acceleration=limits.max_acceleration)


def _interp_rows(x: np.ndarray, grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(x, grid, values[:, column]) for column in range(values.shape[1])], axis=1)


def _hermite(points: np.ndarray, slopes: np.ndarray, grid: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Hermite through ``points`` with ``slopes`` on ``grid``, at ``x``: (values, first derivatives)."""
    index = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, len(grid) - 2)
    h = (grid[index + 1] - grid[index])[:, None]
    u = ((x - grid[index])[:, None]) / h
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    d00, d10, d01, d11 = 6 * u**2 - 6 * u, 3 * u**2 - 4 * u + 1, -6 * u**2 + 6 * u, 3 * u**2 - 2 * u
    p0, p1, m0, m1 = points[index], points[index + 1], slopes[index], slopes[index + 1]
    values = h00 * p0 + h10 * h * m0 + h01 * p1 + h11 * h * m1
    derivatives = (d00 * p0 + d01 * p1) / h + d10 * m0 + d11 * m1
    return values, derivatives


__all__ = ["BUILT_IN", "KEYWORD_NAMES", "RENAMED_KEYWORDS", "WAYPOINT_TRAJECTORY_GENERATOR_GROUP", "BackendInfo", "BoundaryState",
           "LegTiming", "Timed", "WaypointTrajectoryGenerator", "WaypointTrajectoryGeneratorBackend", "backend", "backend_needs",
           "legs_through", "register_waypoint_trajectory_generator", "renamed_keywords", "waypoint_trajectory_generators"]
