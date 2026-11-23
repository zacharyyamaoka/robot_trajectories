from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..cartesian import follow, poses_for, tool_target_hold
from ..registry import LIMITS, kind
from ..segments import Motion, Piece
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator
from ._lane_c import arc_geometry, arc_path, as_point


@kind(use_cases=("tracking_eval", "accuracy", "demo"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model",), catalog_id="cartesian_arc")
def cartesian_arc(space, goal=None, *, interim=None, centre=None, long_way=False,
                  waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5), orientation="hold",
                  dwell_s=0.0, eef_step_m=0.001, jump_threshold=10.0, max_joint_step_rad=0.1, tolerance_m=1e-4,
                  tolerance_rad=1e-3, label="cartesian arc", start=None):
    """The tool along a circular arc from where it is to ``goal`` (default: a 4 cm arc in the xy plane) (an xyz point in metres, or a 4x4 pose, in the kinematics' base
    frame), at rest at both ends, inverse kinematics at every ``eef_step_m`` and refused at the first sample that fails: Pilz's CIRC.

    The arc is named one of two ways. ``interim``: a point (xyz or pose) the arc passes through, between the start and the goal,
    so the circle is the one through the three points (it may be longer than a half circle). ``centre``: the centre of the circle,
    the arc taking the short way round (``long_way=True`` the other); the start and goal must lie on it. The arc is the geometry
    (a path ``s -> pose`` given to ``cartesian.follow``); the ``waypoint_generator`` times it along its length.

    ``orientation``: "hold" keeps the tool's start orientation, "follow" slerps to the goal pose's own (a position-only kinematics,
    a planar five-bar, ignores orientation). The commanded tool point is in the ``tool_target`` channel at every sample.
    WHY refused, not repaired: three collinear points, a centre off the circle and a half circle by centre name no unique arc."""
    here = space.forward(start)
    if goal is None:
        if interim is not None or centre is not None:
            raise ValueError("cartesian_arc: an interim point or centre needs a goal")
        # WHY a small fixed arc in the base xy plane (a planar five-bar's own plane): a deterministic default derived from where the
        # tool is, so the kind can be called as f(space) (catalog previews, the selfcheck)
        goal = here[:3, 3] + (0.04, 0.0, 0.0)
        interim = here[:3, 3] + (0.02, 0.012, 0.0)
    target = poses_for(space, start, [goal], orientation)[0]
    middle = None if interim is None else as_point(interim, "interim")
    spoke = None if centre is None else as_point(centre, "centre")
    circle_centre, normal, radius, sweep = arc_geometry(here[:3, 3], target[:3, 3], interim=middle, centre=spoke, long_way=long_way)
    path = arc_path(here, target, circle_centre, normal, radius, sweep, orientation)
    length = radius * abs(sweep)
    piece = follow(space, start, path, waypoint_generator, length_m=length, label=f"{label}: arc", eef_step_m=eef_step_m,
                   jump_threshold=jump_threshold, max_joint_step_rad=max_joint_step_rad, tolerance_m=tolerance_m,
                   tolerance_rad=tolerance_rad)
    pieces: list[k.Arrays] = [piece]
    if dwell_s > 0:
        rest = k.hold(piece.q[-1], dwell_s, space.rate_hz, label=f"{label}: dwell")
        pieces.append(Piece(rest.t, rest.q, rest.qd, rest.qdd, rest.label, channels=tool_target_hold(target[:3, 3], len(rest.t))))
    degrees = float(np.degrees(abs(sweep)))
    return Motion(pieces, [f"{label}: radius {1000 * radius:.1f} mm, {degrees:.1f}° of arc, {1000 * length:.1f} mm long"])
