from __future__ import annotations

import numpy as np

from ..registry import LIMITS, kind
from ..segments import Motion
from ..space import State
from ..time_law import TimeLaw
from . import _lane_f as f


@kind(use_cases=("tracking_eval", "conveyor_pick_and_place"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model",), catalog_id="track_and_release", starts="moving")
def track_and_release(space, target=None, *, dwell_s=0.3, leave_m=0.05, leave_direction=(0.0, 0.0, 1.0), target_clock_s=0.0,
                      time_law=TimeLaw.minimum_jerk(peak_speed_fraction=0.3), tolerance_m=1e-4, label="track, then leave",
                      start=None):
    """The second half of a moving-target cycle: already moving with a point (``start`` is the state a trajectory ends in,
    moving: pass that trajectory, e.g. ``meet_moving_point``'s, the first half), keep the tool where it is in the point's frame for ``dwell_s``, then leave ``leave_m``
    along ``leave_direction`` (base frame; on a planar robot its plane's up) while braking to rest.

    - ``target``: as ``moving_target_track`` takes it, or None: the tool's own motion at the start, extrapolated at constant
      velocity (the only target a start is sure to move with; at rest, a fixed point); ``target_clock_s`` is where this trajectory's t = 0 falls on the
      target's clock (the end time of the trajectory it continues, when that one's t = 0 was the target's).
    - The leave is a tool path given in time: velocity V (1 - s) + D s'/T, so the tool starts moving with the point and rises
      off it before it stops (V T / 2 further along the point's way, ``leave_m`` along the direction); ``time_law`` sets how
      fast the rise is (its leg timing for the joint move it amounts to), and the duration grows until every joint is inside
      its limits.

    Refused: a start that is not moving with the target (its tool velocity more than 0.01 m/s off the target's), and the
    ``CannotCatch`` causes of the dwell (``target_leaves_window``, ``target_outruns_arm``, ``intercept_out_of_reach``)."""
    # WHY no clock shift for the default: it is derived from the start, on this trajectory's own clock
    moving = f.moving_target(f.following(space, start)) if target is None else f.on_clock(f.moving_target(target), target_clock_s)
    rate, notes = space.rate_hz, []
    direction = np.asarray(leave_direction, float)
    if np.linalg.norm(direction) < 1e-12:
        raise ValueError("leave_direction must be a nonzero direction")
    if leave_m < 0 or dwell_s < 0:
        raise ValueError(f"dwell_s and leave_m must be >= 0, got {dwell_s}, {leave_m}")
    direction = direction / np.linalg.norm(direction)
    steps, note = f.sample_of(dwell_s, rate, "dwell_s")
    notes += [note] if note else []
    f.check_window(moving, 0.0, steps / rate, "the dwell")
    here = space.forward(start.q)
    rotation = here[:3, :3]
    point, velocity, _, _ = f.target_motion(moving, np.array([0.0]))
    offset = here[:3, 3] - point[0]
    tool_velocity = (f.jacobian(space, start.q) @ start.qd)[:3]
    drift = float(np.linalg.norm(tool_velocity - velocity[0]))
    if drift > f.PACE_VELOCITY_TOLERANCE_M_S:
        raise ValueError(f"{label}: the start is not moving with the target: its tool velocity is {drift:.4f} m/s off the target's "
                         f"(tolerance {f.PACE_VELOCITY_TOLERANCE_M_S}); start it at the end of a trajectory that tracks this target, "
                         "with target_clock_s on that trajectory's clock")
    pieces = []
    end = start
    leave_from = here[:3, 3]
    leave_velocity = velocity[0]
    if steps > 0:
        path = f.chase(moving, np.arange(steps + 1) / rate, rotation, f.fixed_offset(offset))
        dwell = f.synced(space, path, start.q, label=f"{label}: synchronised dwell", first=start, tolerance_m=tolerance_m)
        pieces.append(dwell)
        end = State(dwell.q[-1], dwell.qd[-1], dwell.qdd[-1])
        leave_from, leave_velocity = path.points[-1], path.velocities[-1]
    goal = f.solve(space, f.posed(rotation, leave_from + leave_m * direction), end.q, tolerance_m=tolerance_m, cannot_catch=False,
                   what=f"{label}: {1000 * leave_m:.1f} mm along the leave direction")
    begin = steps / rate

    def make(count: int, stride: int):
        return f.coast(leave_from, leave_velocity, count, rate, rotation, begin_s=begin, displacement=leave_m * direction, stride=stride)

    first_try = max(int(np.ceil(time_law.min_move_s * rate)), f.law_steps(space, time_law, end.q, goal), f.braking_steps(space, end.qd))
    pieces.append(f.fitted(space, make, first_try, end.q, label=f"{label}: leave", first=end, tolerance_m=tolerance_m))
    return Motion(pieces, notes, {"assumes": moving.assumes, "sync": [0.0, begin]})
