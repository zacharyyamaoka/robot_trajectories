from __future__ import annotations

import math

import numpy as np

from ..cartesian import follow
from ..registry import LIMITS, kind
from ..segments import Motion
from ._lane_d import circle_path, peak_tool_speed, plane_axes, tool_speed_law

DIRECTIONS = ("both", "ccw", "cw")


@kind(use_cases=("tracking_eval", "backlash_id", "accuracy"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model", "reach_map"), catalog_id="circular_test")
def circular_test(space, *, diameter_m=0.1, feed_m_s=0.05, directions="both", overlap_rad=math.pi / 4,
                  plane=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), eef_step_m=0.001, tolerance_m=1e-4, label="circular test", start=None):
    """ISO 230-4's circular test (the ballbar's path): the tool traces a circle of ``diameter_m`` at a constant ``feed_m_s`` in the
    plane through its start spanned by ``plane``, one revolution plus ``overlap_rad`` run-in and run-out at each end, counter-
    clockwise and then clockwise (``directions``: "both", "ccw" or "cw"). The quadrant reversals of the axes show as spikes in
    the circular error: backlash and friction, a cheap two-joint screen.

    WHY from memory, not re-read (ISO 230-4:2022 is paywalled): the run-in and run-out overlap (45 degrees here) and the
    counter-clockwise-then-clockwise pair are as remembered. The tool starts at the circle's first point (its run-in start),
    holds its orientation, and rests where the direction changes. Each run is one ``cartesian.follow`` under a trapezoid whose
    plateau is the feed; the joint limits slow it (said in the notes), never the other way round. The last run ends where the
    first began plus the overlap: a round trip returns to the start only when both directions run."""
    if directions not in DIRECTIONS:
        raise ValueError(f"directions is one of {DIRECTIONS}, got {directions!r}")
    if diameter_m <= 0 or overlap_rad < 0:
        raise ValueError("diameter_m must be positive and overlap_rad not negative")
    u, v = plane_axes(plane)
    here_pose = space.forward(start)
    rotation, origin = here_pose[:3, :3], here_pose[:3, 3]
    radius = 0.5 * diameter_m
    # WHY the start is the circle's point at angle 0: the run-in begins where the tool is, so the circle's centre is one radius away
    centre = origin - radius * u
    sweep = 2.0 * math.pi + 2.0 * overlap_rad
    runs = [("counter-clockwise", 0.0, sweep), ("clockwise", sweep % (2.0 * math.pi), -sweep)] if directions == "both" else \
        [("counter-clockwise", 0.0, sweep)] if directions == "ccw" else [("clockwise", 0.0, -sweep)]
    pieces, notes, here_q = [], [], np.asarray(start, float)
    length = radius * sweep
    for name, phase, signed in runs:
        path = circle_path(centre, u, v, radius, rotation, phase_rad=phase, sweep_rad=signed)
        piece = follow(space, here_q, path, tool_speed_law(length, feed_m_s), length_m=length, label=f"{label}: {name}",
                       eef_step_m=eef_step_m, tolerance_m=tolerance_m)
        pieces.append(piece)
        here_q = piece.q[-1]
        peak = peak_tool_speed(piece)
        if peak < 0.98 * feed_m_s:
            notes.append(f"{name}: peak tool speed {peak:.4f} m/s, below the {feed_m_s:.4f} m/s feed (the joint limits bind)")
    return Motion(pieces, tuple(notes), {"feed_m_s": float(feed_m_s), "diameter_m": float(diameter_m), "directions": directions})
