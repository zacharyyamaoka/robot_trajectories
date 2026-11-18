from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..cartesian import tool_target_hold
from ..registry import LIMITS, kind
from ..segments import Motion, Piece
from ..time_law import TimeLaw
from ._lane_c import tool_line, with_point


@kind(use_cases=("tracking_eval", "stop_accuracy", "cycle_time", "contact_eval"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model",), catalog_id="approach_retreat")
def approach_retreat(space, *, distance_m=0.05, axis=None, approach=True, retreat=True,
                     time_law=TimeLaw.minimum_jerk(peak_speed_fraction=0.3), dwell_s=0.3, eef_step_m=0.001, jump_threshold=10.0,
                     max_joint_step_rad=0.1, tolerance_m=1e-4, label="approach and retreat", start=None):
    """The tool in a straight line along one axis, out and back: the short linear legs before and after a point (Pilz LIN
    legs; bam_reach's lift recipes are an approach plus a retreat). The orientation is held throughout.

    ``axis``: the direction, in the kinematics' base frame (default: the tool's own z axis at the start; for a position-only
    kinematics, a planar five-bar whose pose has no z axis to speak of, its own x axis, the tool's heading). ``approach``: the
    leg from the start ``distance_m`` along the axis (a negative distance goes the other way), then ``dwell_s``. ``retreat``:
    the leg back. Both: out and back to the start. ``approach=False``: the tool starts on the point and moves ``distance_m``
    against the axis, out, and stays there (out only). Each leg is one rest-to-rest line under ``time_law``."""
    if not approach and not retreat:
        raise ValueError("approach_retreat needs approach or retreat (or both)")
    if distance_m == 0:
        raise ValueError("approach_retreat.distance_m must not be zero")
    here = space.forward(start)
    if axis is None:
        direction = here[:3, 0] if space.position_only else here[:3, 2]
    else:
        direction = np.asarray(axis, float)
        if direction.shape != (3,):
            raise ValueError(f"axis is a direction (3,) in the base frame, got shape {direction.shape}")
    size = float(np.linalg.norm(direction))
    if size < 1e-12:
        raise ValueError("axis must be a nonzero direction")
    direction = direction / size
    line_args = {"eef_step_m": eef_step_m, "jump_threshold": jump_threshold, "max_joint_step_rad": max_joint_step_rad,
                 "tolerance_m": tolerance_m}
    pieces: list[k.Arrays] = []
    q, point = start, here[:3, 3]
    legs = []
    if approach:
        legs.append(("approach", point + distance_m * direction))
    if retreat:
        legs.append(("retreat", point if approach else point - distance_m * direction))
    for name, goal in legs:
        piece = tool_line(space, q, with_point(here[:3, :3], goal), time_law, label=f"{label}: {name}", **line_args)
        pieces.append(piece)
        q = piece.q[-1]
        if dwell_s > 0 and name == legs[0][0] and len(legs) > 1:
            rest = k.hold(q, dwell_s, space.rate_hz, label=f"{label}: dwell")
            pieces.append(Piece(rest.t, rest.q, rest.qd, rest.qdd, rest.label, channels=tool_target_hold(goal, len(rest.t))))
    return Motion(pieces, [f"{label}: {1000 * abs(distance_m):.1f} mm along ({direction[0]:+.3f}, {direction[1]:+.3f}, {direction[2]:+.3f})"])
