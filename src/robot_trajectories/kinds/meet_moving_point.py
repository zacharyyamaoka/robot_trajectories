from __future__ import annotations

import numpy as np

from ..registry import LIMITS, kind
from ..segments import Motion
from ..time_law import TimeLaw
from . import _lane_f as f


# WHY ends="moving": this is the first half of a moving-target cycle. It hands its exact moving state on, so whatever follows
# (track_and_release, the second half) starts there, and chain refuses anything that does not. Promoted 2026-11-18 from lane F's
# test stand-in (tests/fixtures/lane_f.py), the motion unchanged.
@kind(use_cases=("tracking_eval", "conveyor_pick_and_place"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model",), catalog_id="meet_moving_point", ends="moving")
def meet_moving_point(space, target=None, *, sync_start_s=1.0, sync_end_s=1.3, pose_in_target=None, orientation="hold",
                      time_law=TimeLaw.minimum_jerk(peak_speed_fraction=0.5), tolerance_m=1e-4, label="meet a moving point", start=None):
    """From rest, meet a point that moves at ``sync_start_s`` with its velocity matched and keep the tool on it to
    ``sync_end_s``, ending there MOVING with it: ``moving_target_track`` without the coast to rest. Times are on this
    trajectory's clock (t = 0 its first sample), as the target's are.

    - ``target``: as ``moving_target_track`` takes it: None (a default from the space, through the tool's start point at the
      window's middle, 5 cm/s), a ``LinearMotion`` or a mapping of its fields, a robot_msgs ``PoseTrajectory``, or a fixed
      xyz / 4x4. Its ``assumes`` goes in the segment entry with ``sync`` = [t0, t1] (PEP 0002 §7: a prediction under it).
    - ``pose_in_target``: where the tool sits in the target's frame (an xyz offset or a 4x4); None is on the target.
    - ``orientation``: "hold" keeps the tool's start orientation; "follow" takes the target's (composed with the offset).
    - ``time_law``: the approach, a joint-space leg ending in the tracking state (the quintic through both states); it waits at
      rest first when it would arrive early.
    - ``sync_end_s`` defaults to 0.3 s after the meet: long enough to prove the hand-over state is synchronised, short so the
      kind that follows does the tracking.

    Follow it with ``track_and_release(space, target, target_clock_s=float(out.t[-1]), start=out)``. Refusals are ``CannotCatch``
    as ``moving_target_track``'s: ``target_leaves_window``, ``target_outruns_arm``, ``intercept_out_of_reach``,
    ``time_law_error``."""
    if orientation not in ("hold", "follow"):
        raise ValueError(f"orientation is 'hold' or 'follow', got {orientation!r}")
    rate, notes = space.rate_hz, []
    first, note_0 = f.sample_of(sync_start_s, rate, "sync_start_s")
    last, note_1 = f.sample_of(sync_end_s, rate, "sync_end_s")
    notes += [note for note in (note_0, note_1) if note]
    if not 0 < first < last:
        raise ValueError(f"the sync window needs 0 < sync_start_s < sync_end_s on the {rate:g} Hz grid, got {sync_start_s}, {sync_end_s}")
    t0, t1 = first / rate, last / rate
    # WHY (first + last) / rate and not (t0 + t1) / 2: the stand-in's own expression, so its default target is the same to the bit
    moving = f.moving_target(f.default_target(space, start, 0.5 * (first + last) / rate) if target is None else target)
    offset = np.eye(4) if pose_in_target is None else f.as_pose(pose_in_target)
    at_t0 = np.asarray(moving.target(t0), float)
    rotation = space.forward(start)[:3, :3] if orientation == "hold" else at_t0[:3, :3] @ offset[:3, :3]
    on_target = np.zeros(3) if pose_in_target is None else at_t0[:3, :3] @ offset[:3, 3]
    pieces, approach_notes, _ = f.meet(space, start, moving, first, last, rotation=rotation, offset=on_target,
                                       time_law=time_law, label=label, tolerance_m=tolerance_m)
    return Motion(pieces, [*notes, *approach_notes], {"assumes": moving.assumes, "sync": [t0, t1]})
