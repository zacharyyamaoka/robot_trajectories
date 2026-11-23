from __future__ import annotations

import math

import numpy as np

from .. import kernels as k
from ..registry import kind
from ..segments import Motion
from ..waypoint_trajectory_generator import BoundaryState, WaypointTrajectoryGenerator


def _cubic(q0, v0, q1, v1, total: float, t: np.ndarray):
    """The cubic through (q0, v0) at 0 and (q1, v1) at ``total``: (q, qd, qdd) at ``t`` (N, J)."""
    d = q1 - q0
    c2 = (3.0 * d - (2.0 * v0 + v1) * total) / total**2
    c3 = (-2.0 * d + (v0 + v1) * total) / total**3
    t = np.asarray(t, float)[:, None]
    return q0 + v0 * t + c2 * t**2 + c3 * t**3, v0 + 2.0 * c2 * t + 3.0 * c3 * t**2, 2.0 * c2 + 6.0 * c3 * t


@kind(use_cases=("tracking_eval",), starts="moving", ends="moving", needs=("joint_limits", "velocity_limits", "acceleration_limits"),
      catalog_id="polynomial_move")
def polynomial_move(space, target=None, *, order=5, duration_s=1.0, end_velocity_rad_s=None, end_acceleration_rad_s2=None,
                    label="polynomial move", start=None):
    """A cubic (``order=3``: position and velocity at both ends) or quintic (``order=5``: also acceleration) polynomial from the
    start state to ``target`` over ``duration_s``. The start is a ``State``: a configuration starts at rest, a trajectory that
    ended moving hands its exact state on. The end state is ``end_velocity_rad_s`` and ``end_acceleration_rad_s2`` (default
    rest), so a move can end moving into the next one (``chain`` joins a moving end to a start in that exact state).

    The duration is a floor: when a velocity or acceleration limit binds, the move is lengthened (the notes say so), as
    ``WaypointTrajectoryGenerator.minimum_jerk(move_s=...).leg(..., start_state=, end_state=)`` does for the quintic; the cubic is searched the same
    way. With the rest-to-rest defaults and order 5 this is the minimum-jerk leg. ``target=None`` moves half way to the top of the room.

    WHY ``ends="moving"`` even when the end state is rest: the end state is an argument, so the declaration must allow it; a rest
    end joins any kind that starts at rest, and the guard still holds the output to the limits."""
    if order not in (3, 5):
        raise ValueError(f"polynomial_move.order is 3 (cubic) or 5 (quintic), got {order}")
    if duration_s <= 0:
        raise ValueError(f"polynomial_move.duration_s must be positive, got {duration_s}")
    q0 = start.q
    if target is None:
        goal = q0 + 0.5 * (space.room(q0).high - q0)
    else:
        goal = np.asarray(target, float).reshape(-1)
        if goal.shape != (space.dof,):
            raise ValueError(f"target needs {space.dof} joint positions {space.joint_names}, got {goal.size}")
    zeros = np.zeros(space.dof)
    v1 = zeros if end_velocity_rad_s is None else np.broadcast_to(np.asarray(end_velocity_rad_s, float), (space.dof,)).copy()
    a1 = zeros if end_acceleration_rad_s2 is None else np.broadcast_to(np.asarray(end_acceleration_rad_s2, float), (space.dof,)).copy()
    notes: list[str] = []
    if order == 5:
        law = WaypointTrajectoryGenerator.minimum_jerk(move_s=duration_s)
        leg = law.leg(q0, goal, space.limits, space.rate_hz, label=f"{label}: quintic", start_state=BoundaryState(start.qd, start.qdd),
                      end_state=BoundaryState(v1, a1))
    else:
        bounds = space.bounds
        total, tau = duration_s, np.linspace(0.0, 1.0, 401)
        for _ in range(600):  # WHY a 2 % search from the floor up, as the quintic leg's: the peak is not monotonic in the duration
            steps = max(2, int(math.ceil(total * space.rate_hz - 1e-9)))
            span = steps / space.rate_hz
            _, qd, qdd = _cubic(q0, start.qd, goal, v1, span, tau * span)
            if np.all(np.abs(qd) <= bounds.max_velocity * (1 - 1e-9)) and np.all(np.abs(qdd) <= bounds.max_acceleration * (1 - 1e-9)):
                t = np.arange(steps + 1) / space.rate_hz
                q, qd, qdd = _cubic(q0, start.qd, goal, v1, span, t)
                q[0], q[-1] = q0, goal
                leg = k.Arrays(t, q, qd, qdd, f"{label}: cubic")
                break
            total *= 1.02
        else:
            raise ValueError("no cubic leg between those states fits inside the limits; lower the end velocity or move the point")
    if leg.t[-1] > duration_s + 1e-9:
        notes.append(f"{label}: {duration_s:.3f} s asked, {leg.t[-1]:.3f} s given (a velocity or acceleration limit binds)")
    return Motion([leg], notes)
