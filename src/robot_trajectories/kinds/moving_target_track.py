from __future__ import annotations

import numpy as np

from ..registry import LIMITS, kind
from ..segments import Motion
from ..space import State
from ..time_law import TimeLaw
from . import _lane_f as f


@kind(use_cases=("tracking_eval", "conveyor_pick_and_place", "cycle_time"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model",), catalog_id="moving_target_track")
def moving_target_track(space, target=None, *, sync_start_s=1.0, sync_end_s=2.0, pose_in_target=None, orientation="hold",
                        time_law=TimeLaw.minimum_jerk(peak_speed_fraction=0.5), tolerance_m=1e-4, label="track a moving point",
                        start=None):
    """From rest, meet a point that moves at ``sync_start_s`` with its velocity matched, keep the tool on it to ``sync_end_s``,
    then coast to rest. Times are on this trajectory's clock (t = 0 its first sample), as the target's are.

    - ``target``: None (a default from the space: through the tool's start point at the window's middle, 5 cm/s along the
      way the arm moves the tool most easily there), a ``LinearMotion`` (``LinearMotion.from_sync(pose, sync_dir, sync_speed)``:
      constant velocity along a line) or a mapping of its fields, a robot_msgs ``PoseTrajectory`` (sampled poses, no rotation),
      or a fixed xyz / 4x4. The recipe keeps the target's data form, so ``rebuild`` makes the same samples. Its ``assumes`` goes in the segment entry with
      ``sync`` = [t0, t1]: the trajectory is a prediction under it (PEP 0002 §7).
    - ``pose_in_target``: where the tool sits in the target's frame (an xyz offset or a 4x4); None is on the target.
    - ``orientation``: "hold" keeps the tool's start orientation; "follow" takes the target's (composed with the offset).
    - ``time_law``: the approach, a joint-space leg ending in the tracking state (the quintic through both states); it waits at
      rest first when it would arrive early.

    Refusals are ``CannotCatch``: ``target_leaves_window`` (the sync window is not inside the target's), ``target_outruns_arm``
    (a joint needs more than its limit to keep pace), ``intercept_out_of_reach`` (the target takes the tool out of reach),
    ``time_law_error`` (no approach reaches the moving state in time). After the window the tool coasts to rest along the way
    it was going (velocity decaying minimum-jerk, tool path commanded and solved every sample)."""
    if orientation not in ("hold", "follow"):
        raise ValueError(f"orientation is 'hold' or 'follow', got {orientation!r}")
    rate, notes = space.rate_hz, []
    first, note_0 = f.sample_of(sync_start_s, rate, "sync_start_s")
    last, note_1 = f.sample_of(sync_end_s, rate, "sync_end_s")
    notes += [note for note in (note_0, note_1) if note]
    if not 0 < first < last:
        raise ValueError(f"the sync window needs 0 < sync_start_s < sync_end_s on the {rate:g} Hz grid, got {sync_start_s}, {sync_end_s}")
    t0, t1 = first / rate, last / rate
    moving = f.moving_target(f.default_target(space, start, 0.5 * (t0 + t1)) if target is None else target)
    offset = np.eye(4) if pose_in_target is None else f.as_pose(pose_in_target)
    at_t0 = np.asarray(moving.target(t0), float)
    rotation = space.forward(start)[:3, :3] if orientation == "hold" else at_t0[:3, :3] @ offset[:3, :3]
    pieces, approach_notes, path = f.meet(space, start, moving, first, last, rotation=rotation, offset=at_t0[:3, :3] @ offset[:3, 3],
                                          time_law=time_law, label=label, tolerance_m=tolerance_m)
    track = pieces[-1]
    end = State(track.q[-1], track.qd[-1], track.qdd[-1])
    velocity = path.velocities[-1]
    if np.max(np.abs(end.qd)) > 0 or np.max(np.abs(end.qdd)) > 0:
        point = path.points[-1]

        def make(steps: int, stride: int):
            return f.coast(point, velocity, steps, rate, rotation, begin_s=t1, stride=stride)

        steps = max(int(np.ceil(time_law.min_move_s * rate)), f.braking_steps(space, end.qd))
        pieces.append(f.fitted(space, make, steps, end.q, label=f"{label}: coast to rest", first=end, tolerance_m=tolerance_m))
    return Motion(pieces, [*notes, *approach_notes], {"assumes": moving.assumes, "sync": [t0, t1]})
