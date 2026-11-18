# Vendored from bam_ws docs/trajectory_generator/shared/kernels.py at commit b5e1f6965 (origin/main, 2026-11-18): the
# shared math of the trajectory-generator babble round, which every proposal and v6 (the splice this package promotes)
# wrapped. Everything below this comment block is byte-identical to that file (sha256 9485aa020f3ec024...; the full hash
# is KERNELS_SHA256 in tests/test_equivalence.py), so the round's results hold here: the frozen check and v6's tapes.
# WHY vendored and not imported: this package owns its math (the leaf rule: numpy + robot_msgs only, no sys.path into
# bam_ws). A deliberate change edits below this block, updates KERNELS_SHA256 and says why in the commit message.
"""Reference trajectory math shared by all five API proposals (numpy + robot_msgs only).

WHY one shared kernel file: the babble round compares API *shapes*. If every variant re-derived the chirp or the
min-jerk leg, a difference in the generated motion could pass for a difference in the API. Every proposal wraps these
functions; none re-implements them.

Conventions: SI (rad, rad/s, rad/s^2, s), time on axis 0, arrays (N, J). Every segment starts and ends at rest unless
its docstring says otherwise, and every duration is a whole number of samples so pieces join on one uniform grid.
The amplitude of every periodic kind is solved, not guessed: the signal is linear in its amplitude, so it is built at
amplitude 1 and scaled so the worst of position room, velocity and acceleration lands exactly on ``margin``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from robot_msgs.trajectory import JointLimits, JointTrajectory

MIN_JERK_PEAK_VELOCITY = 1.875  # peak |qd| = 1.875 * distance / T
MIN_JERK_PEAK_ACCELERATION = 10.0 / math.sqrt(3.0)  # peak |qdd| = 5.7735 * distance / T^2
DEFAULT_RATE_HZ = 500.0


@dataclass(frozen=True)
class Arrays:
    """One piece of motion before it becomes a robot_msgs JointTrajectory."""

    t: np.ndarray  # (N,), starts at 0
    q: np.ndarray  # (N, J)
    qd: np.ndarray
    qdd: np.ndarray
    label: str = ""

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])


@dataclass(frozen=True)
class Bounds:
    """The numbers a kernel needs, read once from a robot_msgs JointLimits (positions may be a tightened box)."""

    joint_names: tuple[str, ...]
    low: np.ndarray
    high: np.ndarray
    max_velocity: np.ndarray
    max_acceleration: np.ndarray

    @classmethod
    def from_limits(cls, limits: JointLimits, *, default_acceleration: float | None = None) -> Bounds:
        count = len(limits.joint_names)

        def need(values, name: str, fallback: float | None = None) -> np.ndarray:
            if values is None:
                if fallback is None:
                    raise ValueError(f"JointLimits.{name} is required (a RobotDescription's URDF carries no acceleration: pass one)")
                return np.full(count, float(fallback))
            array = np.asarray(values, float)
            if not np.all(np.isfinite(array)):
                if fallback is None:
                    raise ValueError(f"JointLimits.{name} must be finite for generation, got {array}")
                array = np.where(np.isfinite(array), array, float(fallback))
            return array

        return cls(
            joint_names=tuple(limits.joint_names),
            low=need(limits.min_position, "min_position"),
            high=need(limits.max_position, "max_position"),
            max_velocity=need(limits.max_velocity, "max_velocity"),
            max_acceleration=need(limits.max_acceleration, "max_acceleration", default_acceleration),
        )

    @property
    def centre(self) -> np.ndarray:
        return 0.5 * (self.low + self.high)

    def shrunk(self, margin: float) -> Bounds:
        """Positions pulled in by ``margin`` of the range on each side (velocity and acceleration unchanged)."""
        span = self.high - self.low
        return Bounds(self.joint_names, self.low + margin * span, self.high - margin * span, self.max_velocity, self.max_acceleration)


def _samples(duration_s: float, rate_hz: float) -> np.ndarray:
    steps = max(2, int(math.ceil(duration_s * rate_hz - 1e-9)))
    return np.arange(steps + 1) / rate_hz


def hold(q: np.ndarray, duration_s: float, rate_hz: float = DEFAULT_RATE_HZ, *, label: str = "hold") -> Arrays:
    q = np.asarray(q, float)
    t = _samples(duration_s, rate_hz)
    zeros = np.zeros((len(t), q.size))
    return Arrays(t, np.tile(q, (len(t), 1)), zeros, zeros.copy(), label)


def min_jerk_duration(q0: np.ndarray, q1: np.ndarray, bounds: Bounds, *, speed: float = 1.0) -> float:
    """Shortest rest-to-rest min-jerk time that keeps every joint under ``speed`` x its velocity and acceleration limits."""
    distance = np.abs(np.asarray(q1, float) - np.asarray(q0, float))
    by_velocity = MIN_JERK_PEAK_VELOCITY * distance / (speed * bounds.max_velocity)
    by_acceleration = np.sqrt(MIN_JERK_PEAK_ACCELERATION * distance / (speed * bounds.max_acceleration))
    return float(max(np.max(by_velocity), np.max(by_acceleration), 0.0))


def min_jerk(q0, q1, bounds: Bounds, rate_hz: float = DEFAULT_RATE_HZ, *, speed: float = 1.0,
             duration_s: float | None = None, label: str = "move") -> Arrays:
    """Rest-to-rest quintic (minimum-jerk) leg, timed by the slowest joint unless ``duration_s`` is given."""
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    duration = duration_s if duration_s is not None else min_jerk_duration(q0, q1, bounds, speed=speed)
    duration = max(duration, 2.0 / rate_hz)
    t = _samples(duration, rate_hz)
    total = t[-1]
    s = (t / total)[:, None]
    delta = (q1 - q0)[None, :]
    position = 10 * s**3 - 15 * s**4 + 6 * s**5
    velocity = (30 * s**2 - 60 * s**3 + 30 * s**4) / total
    acceleration = (60 * s - 180 * s**2 + 120 * s**3) / total**2
    return Arrays(t, q0 + delta * position, delta * velocity, delta * acceleration, label)


def _fade(t: np.ndarray, total: float, fade_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raised-cosine in/out window and its two derivatives, so a periodic signal starts and ends at rest."""
    w = np.ones_like(t)
    dw = np.zeros_like(t)
    ddw = np.zeros_like(t)
    if fade_s <= 0:
        return w, dw, ddw
    for mask, local in ((t < fade_s, t), (t > total - fade_s, total - t)):
        x = local[mask] / fade_s
        sign = 1.0 if local is t else -1.0
        w[mask] = 0.5 - 0.5 * np.cos(np.pi * x)
        dw[mask] = sign * 0.5 * np.pi / fade_s * np.sin(np.pi * x)
        ddw[mask] = 0.5 * (np.pi / fade_s) ** 2 * np.cos(np.pi * x)
    return w, dw, ddw


def _scaled(centre: np.ndarray, joints: list[int], shape_q: np.ndarray, shape_qd: np.ndarray, shape_qdd: np.ndarray,
            bounds: Bounds, margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale unit-amplitude shapes per joint so the tightest of room / velocity / acceleration is ``margin`` of its limit."""
    count = len(shape_q)
    q = np.tile(centre, (count, 1))
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    for column, joint in enumerate(joints):
        room = min(centre[joint] - bounds.low[joint], bounds.high[joint] - centre[joint])
        ratios = (
            np.max(np.abs(shape_q[:, column])) / max(room, 1e-12),
            np.max(np.abs(shape_qd[:, column])) / bounds.max_velocity[joint],
            np.max(np.abs(shape_qdd[:, column])) / bounds.max_acceleration[joint],
        )
        gain = margin / max(max(ratios), 1e-12)
        q[:, joint] += gain * shape_q[:, column]
        qd[:, joint] = gain * shape_qd[:, column]
        qdd[:, joint] = gain * shape_qdd[:, column]
    return q, qd, qdd


def chirp(centre, joint: int, bounds: Bounds, rate_hz: float = DEFAULT_RATE_HZ, *, start_hz: float = 0.2, end_hz: float = 5.0,
          duration_s: float = 20.0, logarithmic: bool = True, fade_s: float = 1.0, margin: float = 0.8, label: str = "chirp") -> Arrays:
    """Swept sine on one joint (others held), faded in and out; amplitude solved against the joint's limits."""
    centre = np.asarray(centre, float)
    t = _samples(duration_s, rate_hz)
    total = t[-1]
    if logarithmic:
        ratio = end_hz / start_hz
        k = math.log(ratio) / total
        phase = 2 * np.pi * start_hz * (np.exp(k * t) - 1) / k
        omega = 2 * np.pi * start_hz * np.exp(k * t)
        omega_dot = k * omega
    else:
        phase = 2 * np.pi * (start_hz * t + 0.5 * (end_hz - start_hz) * t**2 / total)
        omega = 2 * np.pi * (start_hz + (end_hz - start_hz) * t / total)
        omega_dot = np.full_like(t, 2 * np.pi * (end_hz - start_hz) / total)
    x = np.sin(phase)
    dx = np.cos(phase) * omega
    ddx = -np.sin(phase) * omega**2 + np.cos(phase) * omega_dot
    w, dw, ddw = _fade(t, total, fade_s)
    shape = (x * w)[:, None], (dx * w + x * dw)[:, None], (ddx * w + 2 * dx * dw + x * ddw)[:, None]
    q, qd, qdd = _scaled(centre, [joint], *shape, bounds, margin)
    return Arrays(t, q, qd, qdd, label)


def multisine(centre, joints: list[int], bounds: Bounds, rate_hz: float = DEFAULT_RATE_HZ, *, band_hz: tuple[float, float] = (0.2, 3.0),
              period_s: float = 10.0, periods: int = 2, seed: int = 0, fade_s: float = 1.0, margin: float = 0.8,
              label: str = "multisine") -> Arrays:
    """Odd-harmonic Schroeder-phase multisine on several joints at once, each joint on its own lines (decorrelated)."""
    centre = np.asarray(centre, float)
    t = _samples(period_s * periods, rate_hz)
    total = t[-1]
    base = 1.0 / period_s
    lines = [k for k in range(1, int(band_hz[1] / base) + 1) if k * base >= band_hz[0] and k % 2 == 1]
    rng = np.random.default_rng(seed)
    w, dw, ddw = _fade(t, total, fade_s)
    shapes = [np.zeros((len(t), len(joints))) for _ in range(3)]
    for column in range(len(joints)):
        own = lines[column % 2::2] if len(joints) > 1 and len(lines) > 3 else lines
        count = len(own)
        offset = rng.uniform(0, 2 * np.pi)
        for index, k in enumerate(own):
            phi = -np.pi * index * (index + 1) / count + offset  # Schroeder phases: low crest factor
            omega = 2 * np.pi * k * base
            x = np.sin(omega * t + phi) / math.sqrt(count)
            dx = omega * np.cos(omega * t + phi) / math.sqrt(count)
            ddx = -omega**2 * np.sin(omega * t + phi) / math.sqrt(count)
            shapes[0][:, column] += x * w
            shapes[1][:, column] += dx * w + x * dw
            shapes[2][:, column] += ddx * w + 2 * dx * dw + x * ddw
    q, qd, qdd = _scaled(centre, list(joints), *shapes, bounds, margin)
    return Arrays(t, q, qd, qdd, label)


def step_train(centre, joint: int, bounds: Bounds, rate_hz: float = DEFAULT_RATE_HZ, *, steps_fraction: tuple[float, ...] = (0.5, -0.5, 0.25, -0.25, 0.9),
               dwell_s: float = 1.5, speed: float = 1.0, label: str = "step_train") -> list[Arrays]:
    """Fastest limit-respecting moves to centre + fraction x room, each followed by a dwell; ends back at centre.

    A true position step needs infinite acceleration; this is the step a position controller can be asked to track.
    """
    centre = np.asarray(centre, float)
    room = min(centre[joint] - bounds.low[joint], bounds.high[joint] - centre[joint])
    pieces: list[Arrays] = []
    here = centre.copy()
    for fraction in (*steps_fraction, 0.0):
        target = centre.copy()
        target[joint] += fraction * room
        pieces.append(min_jerk(here, target, bounds, rate_hz, speed=speed, label=f"{label}: step {math.degrees(fraction * room):+.1f}°"))
        pieces.append(hold(target, dwell_s, rate_hz, label=f"{label}: dwell"))
        here = target
    return pieces


def velocity_sweep(centre, joint: int, bounds: Bounds, rate_hz: float = DEFAULT_RATE_HZ, *, speeds_fraction: tuple[float, ...] = (0.1, 0.3, 0.6),
                   span_fraction: float = 0.8, dwell_s: float = 0.5, label: str = "velocity_sweep") -> list[Arrays]:
    """Constant-velocity legs across the joint's range, up and down at each speed (friction / Stribeck identification)."""
    centre = np.asarray(centre, float)
    room = min(centre[joint] - bounds.low[joint], bounds.high[joint] - centre[joint])
    low = centre.copy()
    high = centre.copy()
    low[joint] -= span_fraction * room
    high[joint] += span_fraction * room
    pieces = [min_jerk(centre, low, bounds, rate_hz, speed=0.5, label=f"{label}: to start"), hold(low, dwell_s, rate_hz, label=f"{label}: dwell")]
    here, there = low, high
    for fraction in speeds_fraction:
        for _ in range(2):
            pieces.append(_trapezoid(here, there, joint, fraction * bounds.max_velocity[joint], bounds.max_acceleration[joint] * 0.8, rate_hz,
                                     label=f"{label}: {math.degrees(fraction * bounds.max_velocity[joint]):.0f}°/s"))
            pieces.append(hold(there, dwell_s, rate_hz, label=f"{label}: dwell"))
            here, there = there, here
    pieces.append(min_jerk(here, centre, bounds, rate_hz, speed=0.5, label=f"{label}: home"))
    return pieces


def _trapezoid(q0, q1, joint: int, cruise: float, accel: float, rate_hz: float, *, label: str) -> Arrays:
    distance = float(q1[joint] - q0[joint])
    sign = math.copysign(1.0, distance)
    length = abs(distance)
    ramp = cruise / accel
    if accel * ramp**2 > length:  # never reaches cruise: triangle
        ramp = math.sqrt(length / accel)
        cruise = accel * ramp
    coast = max(0.0, (length - accel * ramp**2) / cruise)
    total = 2 * ramp + coast
    t = _samples(total, rate_hz)
    t = t * (total / t[-1])
    s = np.where(t < ramp, 0.5 * accel * t**2,
                 np.where(t < ramp + coast, 0.5 * accel * ramp**2 + cruise * (t - ramp),
                          length - 0.5 * accel * (total - t) ** 2))
    v = np.where(t < ramp, accel * t, np.where(t < ramp + coast, cruise, accel * (total - t)))
    a = np.where(t < ramp, accel, np.where(t < ramp + coast, 0.0, -accel))
    q = np.tile(np.asarray(q0, float), (len(t), 1))
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    q[:, joint] += sign * s
    qd[:, joint] = sign * v
    qdd[:, joint] = sign * a
    t = np.arange(len(t)) / rate_hz  # back onto the uniform grid (sub-sample timing error < 1/rate)
    return Arrays(t, q, qd, qdd, label)


def waypoint_chain(waypoints: np.ndarray, bounds: Bounds, rate_hz: float = DEFAULT_RATE_HZ, *, start=None, speed: float = 0.6,
                   dwell_s: float = 0.2, label: str = "waypoints") -> list[Arrays]:
    """Min-jerk legs through ``waypoints`` (M, J) with a dwell at each, starting from ``start`` (default the first waypoint)."""
    waypoints = np.asarray(waypoints, float)
    here = np.asarray(start, float) if start is not None else waypoints[0]
    pieces: list[Arrays] = []
    for index, target in enumerate(waypoints):
        if np.allclose(here, target):
            continue
        pieces.append(min_jerk(here, target, bounds, rate_hz, speed=speed, label=f"{label}: leg {index + 1}"))
        if dwell_s > 0:
            pieces.append(hold(target, dwell_s, rate_hz, label=f"{label}: dwell"))
        here = target
    return pieces


def random_waypoints(bounds: Bounds, count: int, *, seed: int = 0, margin: float = 0.1) -> np.ndarray:
    """``count`` uniform joint configurations inside the box shrunk by ``margin`` (seeded)."""
    box = bounds.shrunk(margin)
    rng = np.random.default_rng(seed)
    return rng.uniform(box.low, box.high, size=(count, len(bounds.joint_names)))


def serpentine_grid(bounds: Bounds, points_per_joint: int | tuple[int, ...], *, joints: list[int] | None = None, margin: float = 0.05,
                    rest=None) -> np.ndarray:
    """Boustrophedon grid over ``joints`` (others at ``rest``): neighbours differ in one joint, so every leg is short."""
    box = bounds.shrunk(margin)
    joints = list(range(len(bounds.joint_names))) if joints is None else list(joints)
    counts = (points_per_joint,) * len(joints) if isinstance(points_per_joint, int) else tuple(points_per_joint)
    axes = [np.linspace(box.low[j], box.high[j], n) for j, n in zip(joints, counts, strict=True)]
    rest = bounds.centre if rest is None else np.asarray(rest, float)
    rows: list[np.ndarray] = []

    def walk(level: int, prefix: list[float], forward: bool) -> None:
        values = axes[level] if forward else axes[level][::-1]
        for index, value in enumerate(values):
            here = prefix + [value]
            if level + 1 == len(axes):
                q = rest.copy()
                q[joints] = here
                rows.append(q)
            else:
                walk(level + 1, here, (index % 2 == 0) == forward)

    walk(0, [], True)
    return np.array(rows)


def nearest_order(configs: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbour visiting order (for reachable configurations that are not on a grid)."""
    remaining = list(range(len(configs)))
    here = np.asarray(start, float)
    order: list[int] = []
    while remaining:
        distances = [float(np.max(np.abs(configs[i] - here))) for i in remaining]
        pick = remaining.pop(int(np.argmin(distances)))
        order.append(pick)
        here = configs[pick]
    return configs[order]


def join(pieces: list[Arrays], bounds: Bounds, rate_hz: float = DEFAULT_RATE_HZ, *, transition_speed: float = 0.5) -> tuple[Arrays, list[Arrays]]:
    """Join pieces end to end on one grid. A piece that does not start where the last ended gets a min-jerk transition first.

    Returns the joined arrays and the list of pieces actually used (transitions included), so the caller can hand out either.
    """
    if not pieces:
        raise ValueError("join needs at least one piece")
    used: list[Arrays] = []
    for piece in pieces:
        if used and not np.allclose(used[-1].q[-1], piece.q[0], atol=1e-9):
            used.append(min_jerk(used[-1].q[-1], piece.q[0], bounds, rate_hz, speed=transition_speed, label="transition"))
        used.append(piece)
    t_parts, q_parts, qd_parts, qdd_parts = [], [], [], []
    offset = 0.0
    for index, piece in enumerate(used):
        keep = slice(None) if index == 0 else slice(1, None)
        t_parts.append(piece.t[keep] + offset)
        q_parts.append(piece.q[keep])
        qd_parts.append(piece.qd[keep])
        qdd_parts.append(piece.qdd[keep])
        offset += piece.t[-1]
    t = np.round(np.concatenate(t_parts) * rate_hz) / rate_hz
    return Arrays(t, np.concatenate(q_parts), np.concatenate(qd_parts), np.concatenate(qdd_parts), "joined"), used


def segment_starts(used: list[Arrays]) -> np.ndarray:
    """Start time of each used piece on the joined clock."""
    return np.concatenate([[0.0], np.cumsum([piece.t[-1] for piece in used])[:-1]])


def to_trajectory(arrays: Arrays, limits: JointLimits | None, joint_names: tuple[str, ...], *, channels: dict | None = None) -> JointTrajectory:
    """Wrap arrays as a robot_msgs JointTrajectory (the only trajectory type any proposal may return)."""
    return JointTrajectory(t=arrays.t, joint_names=tuple(joint_names), positions=arrays.q, velocities=arrays.qd,
                           accelerations=arrays.qdd, limits=limits, channels=channels or {})


def inscribed_box(centre, invalid: np.ndarray, bounds: Bounds, *, safety: float = 0.85) -> Bounds:
    """The largest box centred on ``centre`` (scaled from the limit box) that holds none of the ``invalid`` configurations.

    Exact for points: the box scale is the smallest L-infinity distance from the centre to an invalid configuration,
    normalised by the half-ranges; ``safety`` covers the gaps between a reach map's grid points.
    """
    centre = np.asarray(centre, float)
    half = np.minimum(centre - bounds.low, bounds.high - centre)
    if len(invalid) == 0:
        scale = 1.0
    else:
        scale = float(np.min(np.max(np.abs(np.asarray(invalid, float) - centre) / half, axis=1)))
    scale = min(1.0, scale) * safety
    return Bounds(bounds.joint_names, centre - scale * half, centre + scale * half, bounds.max_velocity, bounds.max_acceleration)


def reachable_tour(configs: np.ndarray, step: np.ndarray | float, start: np.ndarray) -> np.ndarray:
    """Visit every configuration of a grid-sampled reachable set moving only between grid neighbours (depth-first, with
    backtracking), so no leg jumps across an unreachable hole. Returns the waypoint list, revisits included."""
    configs = np.asarray(configs, float)
    step = np.broadcast_to(np.asarray(step, float), configs.shape[1])
    keys = np.round(configs / step).astype(int)
    index = {tuple(key): i for i, key in enumerate(keys)}
    first = int(np.argmin(np.max(np.abs(configs - np.asarray(start, float)) / step, axis=1)))
    seen = {first}
    path = [first]
    stack = [first]
    offsets = [np.eye(len(step), dtype=int)[j] * sign for j in range(len(step)) for sign in (1, -1)]
    while stack:
        here = stack[-1]
        nxt = next((index[tuple(keys[here] + o)] for o in offsets if tuple(keys[here] + o) in index and index[tuple(keys[here] + o)] not in seen), None)
        if nxt is None:
            stack.pop()
            if stack:
                path.append(stack[-1])
            continue
        seen.add(nxt)
        stack.append(nxt)
        path.append(nxt)
    return configs[path]
