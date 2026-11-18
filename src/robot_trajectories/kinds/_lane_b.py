"""Lane B's shared helpers (imported by lane B's kinds; registers nothing): effort profiles, a closed-loop prediction that
stops on an event, an amplitude solved against the limits, a seeded pose draw, a station-set builder and a faded dither.

WHY here and not in the core: KINDS_TODO.md lets a lane add one private helper beside its kinds and nothing else; every
function below is a pure function of a ``JointSpace`` and numbers.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .. import kernels as k
from ..segments import Piece

# A solved amplitude lands this fraction of the tightest of room, velocity and acceleration (kernels._scaled uses 0.8).
EFFORT_MARGIN = 0.9
# WHY two periods: with one, a joint a quarter period behind the others gets a small lobe on one side only (the fade swallows it);
# with two, every joint reaches the full amplitude on both sides inside the fade's flat middle
DITHER_PERIODS = 2


# ====================================================================== grid and gravity


def sample_times(duration_s: float, rate_hz: float) -> np.ndarray:
    """A whole number of samples on the space's clock, starting at 0 (kernels._samples' grid)."""
    return np.arange(max(2, int(math.ceil(duration_s * rate_hz - 1e-9))) + 1) / rate_hz


def hold_effort(space, q: np.ndarray) -> np.ndarray:
    """The torque that holds ``q`` still under the model (gravity and nothing else): ``torque(q, 0, 0)``."""
    return space.torque(q, np.zeros(space.dof), np.zeros(space.dof))


def gravity_scale(space, joint: int, q: np.ndarray) -> float:
    """G of a pendulum model ``G sin(q)`` about the bottom, read from the model without knowing its name: at ``q`` and a
    quarter turn on, the torques are G sin q and G cos q, so G is their hypotenuse. (A model that is not a pendulum has no
    such G: the number is then only a scale, and every torque kind still solves its amplitude against the limits.)"""
    here = hold_effort(space, q)[joint]
    turned = q.copy()
    turned[joint] += math.pi / 2
    return float(math.hypot(here, hold_effort(space, turned)[joint]))


def two_sided_room(space, joint: int, q: np.ndarray) -> float:
    """The distance from ``q`` to the nearer position limit of ``joint`` (the pendulum's smaller side)."""
    bounds = space.bounds
    return float(min(q[joint] - bounds.low[joint], bounds.high[joint] - q[joint]))


def torque_headroom(space, joint: int, hold: np.ndarray) -> float:
    """What the drive can add on ``joint`` on top of the torque that holds the pose (refused if it cannot even hold it)."""
    cap = float(np.asarray(space.limits.max_effort, float)[joint])
    over = np.flatnonzero(np.abs(hold) > np.asarray(space.limits.max_effort, float) + 1e-12)
    if len(over):
        raise ValueError(f"the drive cannot hold this pose: {space.joint_names[int(over[0])]} needs {abs(float(hold[over[0]])):.3f} N.m "
                         f"against its {float(np.asarray(space.limits.max_effort, float)[over[0]]):.3f} N.m limit")
    return cap - abs(float(hold[joint]))


# ====================================================================== predictions


def predict_until(space, start: np.ndarray, effort_at: Callable[[int, np.ndarray, np.ndarray], np.ndarray], samples: int, *,
                  stop: Callable[[int, np.ndarray, np.ndarray], bool] | None = None, start_velocity: np.ndarray | None = None):
    """``space.predict``'s scheme (semi-implicit Euler of the model's ``forward(q, qd, tau)``), with the torque allowed to
    depend on the predicted state (a relay, a gravity feed-forward on the moving pose) and the run allowed to stop on an
    event. Returns (q, qd, qdd, effort) of the samples run, the event's sample last.

    WHY the same scheme: the output must equal ``space.predict(start, effort)`` for the effort it returns (tests/test_lane_b.py
    holds that), so the guard's and the contract's reading of "effort and motion agree" is one reading."""
    forward = getattr(space.dynamics, "forward", None)
    if not callable(forward):
        from ..space import MissingRung

        raise MissingRung("predicting an effort-commanded motion", "dynamics_model", detail=" with no forward(q, qd, tau) -> qdd")
    dt = 1.0 / space.rate_hz
    position = np.array(start, float)
    velocity = np.zeros(space.dof) if start_velocity is None else np.array(start_velocity, float)
    q, qd, qdd, effort = [], [], [], []
    for index in range(samples):
        tau = np.asarray(effort_at(index, position, velocity), float)
        acceleration = np.asarray(forward(position, velocity, tau), float)
        q.append(position)
        qd.append(velocity)
        qdd.append(acceleration)
        effort.append(tau)
        if stop is not None and stop(index, position, velocity):
            break
        velocity = velocity + acceleration * dt
        position = position + velocity * dt
    return np.array(q), np.array(qd), np.array(qdd), np.array(effort)


def within_margin(space, joint: int, start: np.ndarray, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, margin: float) -> bool:
    """The prediction keeps ``joint`` inside ``margin`` of its room, speed and acceleration (the guard's three limits)."""
    bounds = space.bounds
    room = two_sided_room(space, joint, start)
    return bool(np.max(np.abs(q[:, joint] - start[joint])) <= margin * room
                and np.all(q >= bounds.low) and np.all(q <= bounds.high)
                and np.max(np.abs(qd[:, joint])) <= margin * bounds.max_velocity[joint]
                and np.max(np.abs(qdd[:, joint])) <= margin * bounds.max_acceleration[joint])


def solve_gain(space, joint: int, start: np.ndarray, build: Callable[[float], np.ndarray], *, margin: float = EFFORT_MARGIN,
               iterations: int = 11) -> tuple[float, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The largest gain in [0, 1] of the nominal profile ``build(gain)`` (an (N, J) effort) whose predicted motion stays inside
    ``margin`` of the joint's room, speed and acceleration; 1 when the nominal profile already does. Returns (gain, effort,
    (q, qd, qdd)). Bisection on the prediction itself: the model is not linear (friction, gravity), so no formula would do.

    WHY solved, not clipped silently: the catalog's amplitude (a fraction of gravity) is a pendulum figure; the guard refuses a
    profile whose predicted motion leaves the limits, so the profile is scaled to the largest one that does not, and the
    caller says so in the notes (never a silent change of the authored number)."""
    def run(gain: float):
        effort = build(gain)
        return effort, space.predict(start, effort)

    effort, motion = run(1.0)
    if within_margin(space, joint, start, *motion, margin):
        return 1.0, effort, motion
    low, high = 0.0, 1.0
    best = run(0.0)
    best_gain = 0.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        trial = run(middle)
        if within_margin(space, joint, start, *trial[1], margin):
            low, best, best_gain = middle, trial, middle
        else:
            high = middle
    return best_gain, best[0], best[1]


def effort_piece(space, t: np.ndarray, effort: np.ndarray, start: np.ndarray, label: str, *, motion=None) -> Piece:
    """A torque profile as a kernel piece: ``effort`` is the command, ``q, qd, qdd`` the model's prediction of it."""
    q, qd, qdd = motion if motion is not None else space.predict(start, effort)
    return Piece(t, q, qd, qdd, label, effort=effort)


def held_effort_piece(space, q: np.ndarray, effort: np.ndarray, t: np.ndarray, label: str) -> Piece:
    """A torque profile whose positions are held, not predicted (an output locked in a fixture, a rotor held at standstill)."""
    zeros = np.zeros((len(t), space.dof))
    return Piece(t, np.tile(q, (len(t), 1)), zeros, zeros.copy(), label, effort=effort)


def fade(t: np.ndarray, total_s: float, fade_s: float) -> np.ndarray:
    """The raised-cosine in/out window of kernels._fade (its value only)."""
    return k._fade(t, total_s, fade_s)[0]


# ====================================================================== position pieces


def slew(space, q0: np.ndarray, q1: np.ndarray, joint: int, speed_rad_s: float, label: str) -> k.Arrays:
    """A constant-speed slewed setpoint of one joint from ``q0`` to ``q1`` (a trapezoid kernel: ramp, cruise, ramp), the other
    joints as they are."""
    bounds = space.bounds
    cruise = min(speed_rad_s, float(bounds.max_velocity[joint]))
    return k._trapezoid(np.asarray(q0, float), np.asarray(q1, float), joint, cruise, 0.5 * float(bounds.max_acceleration[joint]),
                        space.rate_hz, label=label)


def faded_dither(space, q: np.ndarray, duration_s: float, amplitude_rad: float, label: str) -> k.Arrays:
    """A slow dither about ``q`` on every joint at once, ``DITHER_PERIODS`` periods per ``duration_s`` with each joint a quarter period
    behind the last, faded in and out so it starts and ends at rest (Coulomb friction is crossed both ways: it averages out)."""
    t = sample_times(duration_s, space.rate_hz)
    total = float(t[-1])
    w, dw, ddw = k._fade(t, total, 0.25 * total)
    omega = 2 * math.pi * DITHER_PERIODS / total
    phase = omega * t[:, None] + (math.pi / 2) * np.arange(space.dof)[None, :]
    x = np.sin(phase)
    dx = omega * np.cos(phase)
    ddx = -(omega**2) * np.sin(phase)
    w, dw, ddw = w[:, None], dw[:, None], ddw[:, None]
    shape = (x * w, dx * w + x * dw, ddx * w + 2 * dx * dw + x * ddw)
    # WHY a gain per joint that puts each joint's peak at the amplitude: the fade lowers the sine's own peak, by a different
    # amount for each phase
    peak = np.maximum(np.max(np.abs(shape[0]), axis=0), 1e-12)
    gain = amplitude_rad / peak
    return k.Arrays(t, q + gain * shape[0], gain * shape[1], gain * shape[2], label)


# ====================================================================== poses and stations


def draw_stations(space, start: np.ndarray, count: int, *, seed: int, margin: float = 0.1, pad_rad: float = 0.0,
                  admit: Callable[[np.ndarray], bool] | None = None, spread_pool: int = 6) -> list[np.ndarray]:
    """``count`` admitted configurations spread over the room around ``start``: a seeded uniform draw (inside the room shrunk by
    ``margin`` and by ``pad_rad``), kept where ``admit`` says yes, then farthest-point picked so they cover the room (the
    five-bar's ``weightless_poses`` rule, in joint space scaled by each joint's range)."""
    box = space.room(start).shrunk(margin)
    low, high = box.low + pad_rad, box.high - pad_rad
    if np.any(high <= low):
        raise ValueError("the room around the start is smaller than the dither it must hold")
    span = np.maximum(space.bounds.high - space.bounds.low, 1e-9)
    rng = np.random.default_rng(seed)
    pool: list[np.ndarray] = []
    wanted = max(count * spread_pool, count)
    for _ in range(200 * wanted):
        q = rng.uniform(low, high)
        if admit is None or admit(q):
            pool.append(q)
            if len(pool) >= wanted:
                break
    if len(pool) < count:
        raise ValueError(f"only {len(pool)} of {count} admitted configurations found in the room around the start "
                         f"({len(pool)} of {200 * wanted} draws passed)")
    chosen = [pool[0]]
    distance = np.array([np.linalg.norm((q - chosen[0]) / span) for q in pool])
    while len(chosen) < count:
        index = int(np.argmax(distance))
        chosen.append(pool[index])
        distance = np.minimum(distance, np.array([np.linalg.norm((q - pool[index]) / span) for q in pool]))
    return chosen


def tool_ok(space, q: np.ndarray) -> bool:
    """The kinematics has a tool pose at ``q`` (a five-bar's linkage closes there) and nothing collides."""
    try:
        space.forward(q)
    except ValueError:
        return False
    return not (space.checks_collision and space.in_collision(q))


def holdable(space, q: np.ndarray) -> bool:
    """The drive can hold ``q`` still (when the space states torque limits): the model's holding torque is inside them."""
    if not space.has("torque_limits") or space.dynamics is None:
        return True
    return bool(np.all(np.abs(hold_effort(space, q)) <= np.asarray(space.limits.max_effort, float)))


def fold_ok(space, q: np.ndarray, min_margin: float) -> bool:
    """A kinematics that reports a ``fold_margin(q)`` (how far a closed linkage is from folding) must clear ``min_margin``;
    one that reports none passes (its own forward refusing a folded pose is already tool_ok's test)."""
    margin = getattr(space.kinematics, "fold_margin", None)
    return True if not callable(margin) else bool(margin(q) >= min_margin)


def stations_entry(stations: list[np.ndarray]) -> dict[str, Any]:
    """The segment entry a pose set carries: its stations' configurations (rad, JSON)."""
    return {"stations": [[float(value) for value in q] for q in stations]}


def station_pieces(space, start: np.ndarray, stations: list[np.ndarray], law, *, label: str, dwell: Callable[[int, np.ndarray], k.Arrays]):
    """Legs under ``law`` from ``start`` through each station, a dwell piece at each labelled ``"<label>: station k"``;
    ``dwell(index, q)`` makes the dwell (a hold, a dither)."""
    pieces: list[k.Arrays] = []
    here = np.asarray(start, float)
    for index, q in enumerate(stations, start=1):
        if not np.allclose(here, q, atol=1e-12):
            pieces.append(law.leg(here, q, space.limits, space.rate_hz, label=f"{label}: move to station {index}"))
        pieces.append(dwell(index, q))
        here = q
    return pieces
