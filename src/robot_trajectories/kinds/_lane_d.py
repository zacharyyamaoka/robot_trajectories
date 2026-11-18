"""Lane D's shared helpers (standard tests, calibration pose sets, task cycles). Imported by lane D's kinds; registers nothing.

What is here, and only because two or more of the nine kinds need it:
- ``usable`` / ``line_clear`` / ``try_solve``: whether a configuration (or a joint-space line) is inside the room, has a tool pose
  and is free of collision; one inverse-kinematics answer checked the way ``cartesian.solve_points`` checks it, but returning
  None instead of raising (a candidate screen, not a refusal).
- ``pose_set``: the house shape of a pose set (KINDS_TODO "Three output shapes", 1): stations joined by ``TimeLaw`` legs with a
  dwell piece at each, labelled ``"<label>: station k"``, the stations' configurations in the segment entry.
- ``circle_path`` / ``plane_axes``: a tool circle for ``cartesian.follow`` (the private arc helper KINDS_TODO allows lane D).
- ``look_at_rotation`` / ``rotation_between`` / ``rotation_about``: the tool orientations of the calibration view sets.
- ``tool_speed_law`` / ``peak_tool_speed``: a trapezoid whose plateau IS a stated tool speed, and a measure of what came out.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import kernels as k
from ..cartesian import tool_target_hold
from ..segments import TOOL_TARGET, Motion, Piece
from ..time_law import TimeLaw

DEFAULT_LAW = TimeLaw.minimum_jerk(peak_speed_fraction=0.5)


# ====================================================================== is a configuration usable?


def forward_or_none(space: Any, q: np.ndarray) -> np.ndarray | None:
    """The tool pose at ``q``, or None where the kinematics has none (a linkage that does not close, a fold)."""
    try:
        return space.forward(q)
    except ValueError:
        return None


def usable(space: Any, q: np.ndarray, room: k.Bounds | None = None) -> bool:
    """Inside the room (the limit box, or the reach map's hole-free box), a tool pose exists, and no collision."""
    q = np.asarray(q, float)
    bounds = room if room is not None else space.bounds
    if np.any(q < bounds.low - 1e-9) or np.any(q > bounds.high + 1e-9):
        return False
    if space.kinematics is not None and forward_or_none(space, q) is None:
        return False
    return not (space.checks_collision and space.in_collision(q))


def line_clear(space: Any, a: np.ndarray, b: np.ndarray, room: k.Bounds | None = None, samples: int = 40) -> bool:
    """Every sampled configuration on the straight joint-space line a -> b is ``usable``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return all(usable(space, a + (b - a) * fraction, room) for fraction in np.linspace(0.0, 1.0, samples))


def try_solve(space: Any, pose: np.ndarray, seed: np.ndarray, room: k.Bounds | None = None, *, tolerance_m: float = 1e-4,
              tolerance_rad: float = 1e-3) -> np.ndarray | None:
    """Inverse kinematics at ``pose`` seeded at ``seed``, kept only if ``usable`` and forward kinematics lands on the pose within
    the tolerances (orientation too unless the kinematics is position only); None otherwise."""
    from ..cartesian import rotation_angle

    q = space.inverse(pose, seed)
    if q is None or not usable(space, q, room):
        return None
    reached = space.forward(q)
    if float(np.linalg.norm(reached[:3, 3] - pose[:3, 3])) > tolerance_m:
        return None
    if not space.position_only and rotation_angle(reached[:3, :3], pose[:3, :3]) > tolerance_rad:
        return None
    return q


def pose_of(rotation: np.ndarray, point: Sequence[float]) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, np.asarray(point, float)
    return pose


def nearest_order(start: np.ndarray, configurations: Sequence[np.ndarray]) -> list[int]:
    """A nearest-neighbour tour (largest joint distance) from ``start`` through every configuration: short legs, no optimum claimed."""
    remaining = list(range(len(configurations)))
    here, order = np.asarray(start, float), []
    while remaining:
        best = min(remaining, key=lambda index: float(np.max(np.abs(np.asarray(configurations[index]) - here))))
        order.append(best)
        remaining.remove(best)
        here = np.asarray(configurations[best], float)
    return order


# ====================================================================== the pose-set shape


def pose_set(space: Any, label: str, start: np.ndarray, stations: Sequence[np.ndarray], *, points: Sequence[Any] | None = None,
             dwell_s: float, time_law: TimeLaw | None, room: k.Bounds | None = None, via: np.ndarray | None = None,
             entry: dict[str, Any] | None = None, notes: Sequence[str] = ()) -> Motion:
    """Stations visited in order: a ``time_law`` leg to each, then a dwell of ``dwell_s`` labelled ``"<label>: station k"``.

    ``points``: the tool point commanded at each station (``tool_target`` is set on the dwell and on its leg's last sample, NaN
    between: the path between stations is not commanded). ``via``: a configuration every leg whose straight joint line is not
    ``usable`` is routed through (a rest there): a pose set built by sampling can have a fold between two good poses.
    WHY a dwell at every station and refused without one (KINDS_TODO rule 1): a station is where the measurement happens."""
    if dwell_s <= 0:
        raise ValueError(f"{label}: dwell_s must be positive: a pose set measures at its stations")
    law = time_law if time_law is not None else DEFAULT_LAW
    pieces: list[k.Arrays] = []
    here = np.asarray(start, float)
    for index, target in enumerate(stations, start=1):
        target = np.asarray(target, float)
        point = None if points is None or points[index - 1] is None else np.asarray(points[index - 1], float)
        legs = [(target, f"{label}: leg to station {index}")]
        if via is not None and space.kinematics is not None and not line_clear(space, here, target, room):
            legs.insert(0, (np.asarray(via, float), f"{label}: leg via the start to station {index}"))
        for goal, leg_label in legs:
            leg = law.leg(here, goal, space.limits, space.rate_hz, label=leg_label, dynamics=space.dynamics)
            tool = np.full((len(leg.t), 3), np.nan)
            if point is not None and goal is target:
                tool[-1] = point
            pieces.append(Piece(leg.t, leg.q, leg.qd, leg.qdd, leg.label, channels={TOOL_TARGET: tool}))
            here = goal
        rest = k.hold(target, dwell_s, space.rate_hz, label=f"{label}: station {index}")
        channels = tool_target_hold(point, len(rest.t)) if point is not None else {TOOL_TARGET: np.full((len(rest.t), 3), np.nan)}
        pieces.append(Piece(rest.t, rest.q, rest.qd, rest.qdd, rest.label, channels=channels))
    return Motion(pieces, tuple(notes), {"stations": [[float(value) for value in station] for station in stations], **(entry or {})})


# ====================================================================== geometry


def unit(vector: Sequence[float], what: str = "direction") -> np.ndarray:
    vector = np.asarray(vector, float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError(f"{what} must be a nonzero vector")
    return vector / norm


def plane_axes(plane: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane directions (u, v) from ``plane`` (two direction vectors; the second is made orthogonal to the first)."""
    if len(plane) != 2:
        raise ValueError(f"plane is two direction vectors (u, v), got {len(plane)}")
    u = unit(plane[0], "plane[0]")
    v = np.asarray(plane[1], float)
    v = v - np.dot(v, u) * u
    if np.linalg.norm(v) < 1e-9:
        raise ValueError("plane's two directions are parallel")
    return u, v / np.linalg.norm(v)


def circle_path(centre: np.ndarray, u: np.ndarray, v: np.ndarray, radius_m: float, rotation: np.ndarray, *, phase_rad: float,
                sweep_rad: float):
    """A tool circle for ``cartesian.follow``: progress s in [0, 1] -> the pose at angle ``phase_rad + s * sweep_rad`` on the
    circle of ``radius_m`` about ``centre`` in the plane spanned by u, v, the tool keeping ``rotation``. A negative sweep goes the other way."""
    centre = np.asarray(centre, float)

    def at(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, float)
        angle = phase_rad + s * sweep_rad
        poses = np.tile(np.eye(4), (len(s), 1, 1))
        poses[:, :3, :3] = rotation
        poses[:, :3, 3] = centre + radius_m * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v)
        return poses

    return at


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """The rotation matrix of ``angle`` (rad) about the unit ``axis`` (Rodrigues)."""
    axis = unit(axis, "axis")
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * cross @ cross


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking the unit vector ``a`` onto ``b``."""
    a, b = unit(a), unit(b)
    axis = np.cross(a, b)
    sine, cosine = float(np.linalg.norm(axis)), float(np.dot(a, b))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return rotation_about(np.cross(a, helper), math.pi)
    return rotation_about(axis, math.atan2(sine, cosine))


# ====================================================================== tool speed


def tool_speed_law(length_m: float, tool_speed_m_s: float) -> TimeLaw:
    """A trapezoid over ``length_m`` whose cruise plateau is ``tool_speed_m_s``: T = L / (v (1 - ramp)). WHY a trapezoid: a stated
    tool speed (a standard's, a feed rate) must be a plateau the run holds, not the peak of a bell. If the joint limits bind,
    the law's duration is raised, never the limits (``peak_tool_speed`` says what came out)."""
    if tool_speed_m_s <= 0 or length_m <= 0:
        raise ValueError(f"tool speed and length must be positive, got {tool_speed_m_s} m/s over {length_m} m")
    ramp = TimeLaw.trapezoid(move_s=1.0).ramp_fraction
    return TimeLaw.trapezoid(move_s=length_m / (tool_speed_m_s * (1.0 - ramp)))


def peak_tool_speed(piece: Any) -> float:
    """The peak tool speed (m/s) of a Cartesian ``Piece``, from its commanded ``tool_target`` points."""
    tool = np.asarray(piece.channels[TOOL_TARGET], float)
    step = np.linalg.norm(np.diff(tool, axis=0), axis=1)
    return float(np.max(step / np.diff(piece.t)))
