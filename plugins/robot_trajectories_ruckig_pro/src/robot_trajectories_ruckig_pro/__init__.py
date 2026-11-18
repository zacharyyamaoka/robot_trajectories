"""Ruckig Pro, the local build, as a drop-in for the robot_trajectories time law ``ruckig``: waypoints and position limits.

    law = TimeLaw("ruckig", peak_speed_fraction=1.0)
    law.along(path, space.limits, space.rate_hz)            # a path of M points, passed through in order (Pro's waypoints)
    law.leg(q0, q1, space.limits, space.rate_hz, end_state=...)   # one leg, as Community does, with position limits enforced
    cartesian_path(space, [p1, p2], time_law=law)            # the IK path of a tool line, every IK point a waypoint

Registered through ``robot_trajectories.time_laws`` under the same name as Community
(``ruckig = "robot_trajectories_ruckig_pro:RuckigProBackend"``). Zach, 2026-11-18: "...like ruckig (and if you have the
professional version we can support that as a drop-in backend as well)".

**Install exactly one of** ``robot-trajectories-ruckig`` (Community) **or** this package. Both register ``ruckig``, and the
``ruckig`` / ``ruckig-pro`` distributions own the same ``ruckig`` module, so which binary ran would depend on install order:
robot_trajectories refuses two providers of one time law, and this backend refuses to load beside the ``ruckig``
distribution.

**Provenance, verified at load** (a copy of bam_ws ``bam_traj_gen/ruckig_backend.py`` ``require_local_ruckig_pro``, fail
closed): the ``ruckig-pro`` distribution at 0.19.4; the Community ``ruckig`` distribution absent; installed from a local
``file://`` checkout (its ``direct_url.json``) that is the pinned one, ``~/bam_ws/src/external/ruckig-pro-v0.19.4-obfuscated-code``
(or ``$ROBOT_TRAJECTORIES_RUCKIG_PRO_CHECKOUT``, which must carry the same folder name); the loaded module's
``__version__`` matches; the Pro-only ``WaypointsCalculator`` is present; the native module belongs to the distribution; and it
carries no cloud-client markers. Any failure raises at load, so the plugin is reported (``robot_trajectories.plugins.errors()``)
and the time law ``ruckig`` is not registered. It never falls back to Community or to the cloud.

What it serves:
- a **path of M >= 2 points**: M = 2 is one leg (from any start state to any end state, its duration on the sample grid by
  Ruckig's ``DurationDiscretization.Discrete``); M > 2 makes the inner points Ruckig Pro's ``intermediate_positions``, solved
  locally: the motion passes through every point, in order (checked here to 1e-6 rad), with the path between them and the
  timing chosen together. Pro takes no global minimum or discrete duration with waypoints, so a rest-to-rest waypoint path is
  slowed uniformly onto the grid (velocities / c, accelerations / c^2: never past a limit), and a moving end gets a
  per-section minimum on its last section instead;
- **position limits** (``min_position`` / ``max_position``, enforced by Pro);
- ``law.options`` as Community's: ``acceleration_fraction``, ``jerk_fraction``, ``synchronization`` (two-point legs; default
  ``"phase"``), ``max_jerk`` (only when the JointLimits state none);
- ``path_progress``: the fractional point index at every sample (the sample projected onto the chord it is passing).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import ruckig
from robot_msgs.trajectory import JointLimits, JointTrajectory

NAME = "ruckig"
EDITION = "pro"
COMMUNITY_PACKAGE = "robot-trajectories-ruckig"
RUCKIG_PRO_DISTRIBUTION = "ruckig-pro"
RUCKIG_PRO_VERSION = "0.19.4"
RUCKIG_PRO_CHECKOUT_NAME = "ruckig-pro-v0.19.4-obfuscated-code"
CHECKOUT_ENV = "ROBOT_TRAJECTORIES_RUCKIG_PRO_CHECKOUT"
DEFAULT_CHECKOUT = Path.home() / "bam_ws" / "src" / "external" / RUCKIG_PRO_CHECKOUT_NAME
_CLOUD_BACKEND_MARKERS = (b"cloud api", b"api.ruckig", b"cloudclient")  # bam_traj_gen's list, unchanged
OPTIONS = ("acceleration_fraction", "jerk_fraction", "max_jerk", "synchronization")
END_TOLERANCE = 1e-6  # WHY: Ruckig reaches its target to ~1e-10; a larger miss is a bug, not rounding, and is refused
WAYPOINT_TOLERANCE = 1e-6  # rad: how close the motion must pass every intermediate point


# ====================================================================== provenance (bam_traj_gen's check, copied)


@dataclass(frozen=True, slots=True)
class RuckigProInfo:
    """Verified provenance of the loaded local Pro extension."""

    version: str
    source_checkout: Path
    module_path: Path
    module_sha256: str


def expected_checkout() -> Path:
    """The Pro checkout the distribution must come from: the pinned workspace one, or ``$ROBOT_TRAJECTORIES_RUCKIG_PRO_CHECKOUT``."""
    candidate = Path(os.environ.get(CHECKOUT_ENV) or DEFAULT_CHECKOUT).expanduser()
    if candidate.name != RUCKIG_PRO_CHECKOUT_NAME or not candidate.is_dir():
        raise RuntimeError(f"the local Ruckig Pro checkout must be a folder named {RUCKIG_PRO_CHECKOUT_NAME!r}; {candidate} is not "
                           f"(set {CHECKOUT_ENV} to it)")
    return candidate.resolve()


def _direct_url_path(distribution: metadata.Distribution) -> Path:
    text = distribution.read_text("direct_url.json")
    if text is None:
        raise RuntimeError("ruckig-pro has no direct_url.json; install it from the local Pro checkout, not a registry or a prebuilt wheel")
    try:
        direct_url = json.loads(text)
        parsed = urlparse(direct_url["url"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("ruckig-pro direct_url.json is invalid") from error
    if parsed.scheme != "file":
        raise RuntimeError(f"ruckig-pro must resolve from a local file path; found {direct_url.get('url')!r}")
    return Path(unquote(parsed.path)).resolve()


@lru_cache(maxsize=1)
def require_local_ruckig_pro() -> RuckigProInfo:
    """Provenance of the loaded ``ruckig`` module, or a RuntimeError before any trajectory is computed.

    WHY it rejects an environment holding the Community distribution even beside ``ruckig-pro`` (bam_traj_gen): both own the
    same ``ruckig`` extension file, so installation order would select the behaviour, and Community's waypoints go to the cloud."""
    try:
        distribution = metadata.distribution(RUCKIG_PRO_DISTRIBUTION)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("the local Ruckig Pro build (distribution 'ruckig-pro') is not installed; the Community ruckig package "
                           "would use the cloud API for intermediate waypoints") from error
    try:
        metadata.distribution("ruckig")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("both the 'ruckig' and 'ruckig-pro' distributions are installed: they provide the same native module, "
                           f"so the backend is ambiguous; install exactly one of {COMMUNITY_PACKAGE} or robot-trajectories-ruckig-pro")
    if distribution.version != RUCKIG_PRO_VERSION:
        raise RuntimeError(f"ruckig-pro {RUCKIG_PRO_VERSION} is required; found {distribution.version}")
    expected = expected_checkout()
    source = _direct_url_path(distribution)
    if source != expected:
        raise RuntimeError(f"ruckig-pro resolved from the wrong source: {source}; expected {expected}")
    if getattr(ruckig, "__version__", None) != RUCKIG_PRO_VERSION:
        raise RuntimeError("the loaded ruckig module's version does not match the local Pro distribution")
    if not hasattr(ruckig, "WaypointsCalculator"):
        raise RuntimeError("the loaded ruckig module lacks the Pro-only WaypointsCalculator API")
    module_path = Path(ruckig.__file__).resolve()
    root = Path(distribution.locate_file("")).resolve()
    try:
        relative = module_path.relative_to(root)
    except ValueError as error:
        raise RuntimeError("the loaded ruckig module is outside the installed ruckig-pro distribution") from error
    files = distribution.files
    if files is None or relative not in {Path(file) for file in files}:
        raise RuntimeError("the loaded ruckig module is not owned by the installed ruckig-pro distribution")
    module_bytes = module_path.read_bytes()
    lowered = module_bytes.lower()
    found = tuple(marker.decode("ascii") for marker in _CLOUD_BACKEND_MARKERS if marker in lowered)
    if found:
        raise RuntimeError("the loaded ruckig native module contains cloud-backend markers: " + ", ".join(found))
    return RuckigProInfo(distribution.version, source, module_path, hashlib.sha256(module_bytes).hexdigest())


# ====================================================================== the backend


class RuckigProBackend:
    """The time law ``ruckig`` on the verified local Ruckig Pro build: legs and waypoint paths, position limits enforced."""

    name = NAME
    edition = EDITION
    needs = ("jerk_limits",)

    def __init__(self) -> None:
        # WHY at construction: the entry point builds the backend at load, so an unverified build is never registered
        self.provenance = require_local_ruckig_pro()

    def can_service(self, path_points: Any, limits: JointLimits, law: Any, start_state: Any = None, end_state: Any = None) -> str:
        path = np.asarray(path_points, float)
        if path.ndim != 2 or len(path) < 2:
            return f"a path is (M, J) with M >= 2 points, got shape {path.shape}"
        try:
            settings(limits, law)
        except ValueError as error:
            return str(error)
        return ""

    def time_parameterize(self, path_points: Any, limits: JointLimits, rate_hz: float, law: Any, *, start_state: Any = None,
                          end_state: Any = None) -> JointTrajectory:
        require_local_ruckig_pro()  # WHY again: the cached check is free, and no motion is computed without it
        refusal = self.can_service(path_points, limits, law, start_state, end_state)
        if refusal:
            raise ValueError(refusal)
        path = np.asarray(path_points, float)
        t, q, qd, qdd, progress = solve(path, limits, rate_hz, law, start_state, end_state)
        channels = {} if progress is None else {"path_progress": progress}
        return JointTrajectory(t=t, joint_names=tuple(limits.joint_names), positions=q, velocities=qd, accelerations=qdd,
                               limits=limits, channels=channels)


def settings(limits: JointLimits, law: Any) -> dict[str, Any]:
    """Ruckig's limits from the JointLimits handed in and the law's options, or a ValueError saying why not."""
    options = dict(law.options) if law is not None else {}
    unknown = sorted(set(options) - set(OPTIONS))
    if unknown:
        raise ValueError(f"time law 'ruckig' has no option {unknown} (it takes {OPTIONS})")
    dof = len(limits.joint_names)
    fractions = {}
    for key in ("acceleration_fraction", "jerk_fraction"):
        value = float(options.get(key, 1.0))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"time law 'ruckig': {key} must lie in (0, 1], got {value}")
        fractions[key] = value
    sync_names = {"phase": ruckig.Synchronization.Phase, "time": ruckig.Synchronization.Time,
                  "time_if_necessary": ruckig.Synchronization.TimeIfNecessary, "none": ruckig.Synchronization.No}
    sync = options.get("synchronization", "phase")
    if sync not in sync_names:
        raise ValueError(f"time law 'ruckig': synchronization must be one of {tuple(sync_names)}, got {sync!r}")
    velocity = _finite_positive(limits.max_velocity, dof)
    acceleration = _finite_positive(limits.max_acceleration, dof)
    if velocity is None or acceleration is None:
        raise ValueError("time law 'ruckig' needs finite, positive max_velocity and max_acceleration in the JointLimits")
    jerk = _finite_positive(limits.max_jerk, dof)
    if jerk is None and "max_jerk" in options:
        jerk = _finite_positive(np.broadcast_to(np.asarray(options["max_jerk"], float), (dof,)), dof)
    if jerk is None:
        raise ValueError("a jerk-limited S-curve needs a jerk limit: state max_jerk in the JointLimits (the space's jerk_limits "
                         "rung) or TimeLaw('ruckig', ..., options={'max_jerk': ...}) in rad/s^3")
    low = None if limits.min_position is None else np.asarray(limits.min_position, float)
    high = None if limits.max_position is None else np.asarray(limits.max_position, float)
    return {"max_velocity": velocity, "max_acceleration": acceleration * fractions["acceleration_fraction"],
            "max_jerk": jerk * fractions["jerk_fraction"], "synchronization": sync_names[sync],
            "synchronization_given": "synchronization" in options, "min_position": low, "max_position": high}


def solve(path: np.ndarray, limits: JointLimits, rate_hz: float, law: Any, start_state: Any,
          end_state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """(t, q, qd, qdd, progress) through every point of ``path``, sampled at ``rate_hz`` from t = 0, the last sample the end state."""
    values = settings(limits, law)
    count, dof = path.shape
    v0, a0 = _state(start_state, dof)
    v1, a1 = _state(end_state, dof)
    dt = 1.0 / rate_hz
    floor = max(float(law.min_move_s), float(law.move_s or 0.0), 2 * dt)
    moving_end = bool(np.any(v1 != 0.0) or np.any(a1 != 0.0))
    inp = ruckig.InputParameter(dof)
    inp.current_position, inp.current_velocity, inp.current_acceleration = path[0].tolist(), v0.tolist(), a0.tolist()
    inp.target_position, inp.target_velocity, inp.target_acceleration = path[-1].tolist(), v1.tolist(), a1.tolist()
    inp.max_velocity = values["max_velocity"].tolist()
    inp.max_acceleration = values["max_acceleration"].tolist()
    inp.max_jerk = values["max_jerk"].tolist()
    if values["min_position"] is not None:
        inp.min_position = values["min_position"].tolist()
    if values["max_position"] is not None:
        inp.max_position = values["max_position"].tolist()
    if count == 2 or values["synchronization_given"]:
        inp.synchronization = values["synchronization"]
    if count == 2:
        # WHY Discrete for one leg: the duration is a whole number of samples, so the last sample is the end state itself
        inp.duration_discretization = ruckig.DurationDiscretization.Discrete
        inp.minimum_duration = floor
    else:
        inp.intermediate_positions = [row.tolist() for row in path[1:-1]]  # solved locally: provenance verified at load
    otg = ruckig.Ruckig(dof, dt, count + 2)
    trajectory = ruckig.Trajectory(dof, count + 2)
    _calculate(otg, inp, trajectory)
    if count == 2 and inp.synchronization != ruckig.Synchronization.Time and _end_miss(trajectory, path[-1], v1) > END_TOLERANCE:
        # WHY a Time fallback (as the Community plugin): Ruckig 0.19.4 with Phase synchronization and a Discrete duration can
        # miss a moving end state; Time synchronization lands it exactly, and is Ruckig's own fallback when phase is impossible
        inp.synchronization = ruckig.Synchronization.Time
        _calculate(otg, inp, trajectory)
    duration = float(trajectory.duration)
    scale = 1.0
    if count > 2:
        steps = max(2, int(math.ceil(max(duration, floor) * rate_hz - 1e-9)))
        if not moving_end:
            # WHY slowed uniformly, not re-solved: Pro takes no global minimum or discrete duration with waypoints, and one
            # factor c >= 1 on the clock keeps the path and divides qd by c, qdd by c^2 (inside every limit still)
            scale = steps / (rate_hz * duration)
        else:
            duration = _land_last_section(otg, inp, trajectory, path, steps * dt)
    steps = int(round(duration * scale * rate_hz))
    if steps < 2 or abs(steps * dt - duration * scale) > 1e-9:
        raise ValueError(f"Ruckig Pro's duration {duration * scale:.9f} s is not a whole number of {rate_hz:g} Hz samples")
    t = np.arange(steps + 1) / rate_hz
    rows = [trajectory.at_time(min(float(time) / scale, duration)) for time in t]
    q = np.array([row[0] for row in rows], float).reshape(len(t), dof)
    qd = np.array([row[1] for row in rows], float).reshape(len(t), dof) / scale
    qdd = np.array([row[2] for row in rows], float).reshape(len(t), dof) / scale**2
    for row, (position, velocity, acceleration), which in ((0, (path[0], v0, a0), "start"), (-1, (path[-1], v1, a1), "end")):
        miss = max(float(np.max(np.abs(q[row] - position))), float(np.max(np.abs(qd[row] - velocity))))
        if miss > END_TOLERANCE:
            raise ValueError(f"Ruckig Pro's {which} sample misses the {which} state by {miss:.3g} (position rad or velocity rad/s)")
        q[row], qd[row], qdd[row] = position, velocity, acceleration
    progress = None
    if count > 2:
        times = waypoint_times(trajectory, path) * scale
        progress = _progress(t, q, times, path)
    return t, q, qd, qdd, progress


def waypoint_times(trajectory: Any, path: np.ndarray) -> np.ndarray:
    """When the motion passes each path point (s from its start), checked to WAYPOINT_TOLERANCE, or a ValueError.

    Ruckig keeps one entry of ``intermediate_durations`` per waypoint it did not filter as redundant; a filtered one (a
    collinear point) is found by the closest approach after the previous point (bam_traj_gen ``_waypoint_times``'s idea)."""
    duration = float(trajectory.duration)
    marks = [float(value) for value in trajectory.intermediate_durations]
    times = [0.0]
    for index in range(1, len(path)):
        target = path[index]
        if index == len(path) - 1:
            found = duration
        elif len(marks) == len(path) - 1 and _miss(trajectory, marks[index - 1], target) <= WAYPOINT_TOLERANCE:
            found = marks[index - 1]
        else:
            found = _closest_after(trajectory, target, times[-1], duration)
        miss = _miss(trajectory, found, target)
        if miss > WAYPOINT_TOLERANCE:
            raise ValueError(f"Ruckig Pro did not pass path point {index} (closest {miss:.3g} rad away)")
        times.append(found)
    return np.asarray(times)


def _closest_after(trajectory: Any, target: np.ndarray, after: float, duration: float) -> float:
    grid = np.linspace(after, duration, 2001)
    distances = np.array([_miss(trajectory, time, target) for time in grid])
    best = int(np.argmin(distances))
    low, high = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(80):  # golden-section search on the distance, unimodal this close to the point
        a, b = high - golden * (high - low), low + golden * (high - low)
        if _miss(trajectory, a, target) < _miss(trajectory, b, target):
            high = b
        else:
            low = a
    return 0.5 * (low + high)


def _end_miss(trajectory: Any, position: np.ndarray, velocity: np.ndarray) -> float:
    q, qd, _ = trajectory.at_time(float(trajectory.duration))
    return max(float(np.max(np.abs(np.asarray(q) - position))), float(np.max(np.abs(np.asarray(qd) - velocity))))


def _miss(trajectory: Any, time: float, target: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(trajectory.at_time(time)[0], float) - target)))


def _progress(t: np.ndarray, q: np.ndarray, times: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Per sample: the chord it is passing (by the waypoint times) plus how far along it the sample projects, kept monotone."""
    index = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(path) - 2)
    chord = path[index + 1] - path[index]
    length = np.maximum(np.sum(chord * chord, axis=1), 1e-24)
    along = np.clip(np.sum((q - path[index]) * chord, axis=1) / length, 0.0, 1.0)
    progress = np.maximum.accumulate(index + along)
    progress[0], progress[-1] = 0.0, len(path) - 1.0
    return progress


def _land_last_section(otg: Any, inp: Any, trajectory: Any, path: np.ndarray, total: float) -> float:
    """A moving end with waypoints: stretch the last section (Pro's per-section minimum) until the whole lasts ``total``."""
    for _ in range(12):
        times = waypoint_times(trajectory, path)
        sections = np.diff(times)
        missing = total - float(trajectory.duration)
        if abs(missing) <= 1e-10:
            return float(trajectory.duration)
        minimums = [0.0] * len(sections)
        minimums[-1] = float(sections[-1] + missing)
        inp.per_section_minimum_duration = minimums
        _calculate(otg, inp, trajectory)
    raise ValueError(f"Ruckig Pro could not land a waypoint path that ends moving on the sample grid ({total:.6f} s); "
                     "time it to rest, or end it moving with a two-point leg")


def _calculate(otg: Any, inp: Any, trajectory: Any) -> None:
    try:
        result = otg.calculate(inp, trajectory)
    except ruckig.RuckigError as error:  # WHY reworded, not swallowed: Ruckig's own reason
        raise ValueError(f"Ruckig Pro refused this path: {' '.join(str(error).split())}") from error
    if result not in (ruckig.Result.Working, ruckig.Result.Finished):
        raise ValueError(f"Ruckig Pro refused this path ({result.name}): the states or the position limits admit no motion here")


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


__all__ = ["EDITION", "NAME", "OPTIONS", "RUCKIG_PRO_CHECKOUT_NAME", "RUCKIG_PRO_VERSION", "RuckigProBackend", "RuckigProInfo",
           "expected_checkout", "require_local_ruckig_pro", "settings", "solve", "waypoint_times"]
