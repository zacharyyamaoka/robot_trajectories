from __future__ import annotations

import math

import numpy as np

from ..cartesian import follow, line
from ..registry import LIMITS, kind
from ..segments import Motion
from ._lane_d import circle_path, peak_tool_speed, plane_axes, pose_of, tool_speed_law

ELEMENTS = ("line", "circle", "rectangle")

# WHY from memory, not re-read (ISO 9283 is paywalled): the path test traces a straight line, a circle and a rectangle's corners
# repeatedly (10 cycles) at set tool speeds (the standard's rated, 50 % and 10 %) and scores path accuracy, path repeatability,
# cornering deviation and speed fidelity. The 0.1 m sizes and 10 cycles are the catalog row's defaults, not the standard's.


def _runs(element, origin, u, v, rotation, *, line_length_m, circle_diameter_m, rectangle_m):
    """The runs of one cycle of ``element``, every cycle returning to ``origin``: (name, path, length_m)."""
    at = lambda point: pose_of(rotation, point)  # noqa: E731
    if element == "line":
        far = origin + line_length_m * u
        return [("out", line(at(origin), at(far)), line_length_m), ("back", line(at(far), at(origin)), line_length_m)]
    if element == "circle":
        radius = 0.5 * circle_diameter_m
        centre = origin + radius * v  # WHY the start is the circle's lowest-v point: the tool begins where it is, on the circle
        return [("revolution", circle_path(centre, u, v, radius, rotation, phase_rad=-0.5 * math.pi, sweep_rad=2.0 * math.pi),
                 math.pi * circle_diameter_m)]
    width, height = rectangle_m
    corners = [origin, origin + width * u, origin + width * u + height * v, origin + height * v, origin]
    sides = [("side 1", width), ("side 2", height), ("side 3", width), ("side 4", height)]
    return [(name, line(at(a), at(b)), length) for (name, length), a, b in zip(sides, corners[:-1], corners[1:], strict=True)]


@kind(use_cases=("tracking_eval", "accuracy", "repeatability"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model", "reach_map"), catalog_id="iso9283_path")
def iso9283_path(space, *, elements=ELEMENTS, tool_speed_m_s=0.1, circle_diameter_m=0.1, line_length_m=0.1, rectangle_m=(0.1, 0.06),
                 cycles=10, plane=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), eef_step_m=0.001, tolerance_m=1e-4, label="path test",
                 start=None):
    """ISO 9283's path test: a straight line (out and back), a circle and a rectangle's corners, each traced ``cycles`` times at
    ``tool_speed_m_s``, in the plane through the tool's start spanned by ``plane`` (two directions in the base frame; the
    robot's own plane on a planar robot). Every element starts and ends where the tool starts.

    Each run is one rest-to-rest ``cartesian.follow`` under a trapezoid whose plateau is the stated tool speed, inverse
    kinematics at every ``eef_step_m``, refused at the first sample that fails. A run the joint limits do not let reach the
    speed is slowed, never the limits: the segment's notes say the peak that came out. The tool stops at each rectangle corner
    (a corner turned at speed is an unbounded acceleration), so cornering deviation is read at the stop. The tool keeps its start
    orientation."""
    unknown = [element for element in elements if element not in ELEMENTS]
    if unknown or not elements:
        raise ValueError(f"elements are chosen from {ELEMENTS}, got {tuple(elements)!r}")
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    u, v = plane_axes(plane)
    here_pose = space.forward(start)
    rotation, origin = here_pose[:3, :3], here_pose[:3, 3]
    pieces, notes, here_q = [], [], np.asarray(start, float)
    for element in elements:
        runs = _runs(element, origin, u, v, rotation, line_length_m=line_length_m, circle_diameter_m=circle_diameter_m,
                     rectangle_m=tuple(rectangle_m))
        for cycle in range(1, cycles + 1):
            for name, path, length in runs:
                piece = follow(space, here_q, path, tool_speed_law(length, tool_speed_m_s), length_m=length,
                               label=f"{label}: {element}, cycle {cycle}, {name}", eef_step_m=eef_step_m, tolerance_m=tolerance_m)
                pieces.append(piece)
                here_q = piece.q[-1]
                if cycle == 1:
                    peak = peak_tool_speed(piece)
                    if peak < 0.98 * tool_speed_m_s:
                        notes.append(f"{element} {name}: peak tool speed {peak:.4f} m/s, below the {tool_speed_m_s:.4f} m/s asked "
                                     f"(the joint limits bind; the run is slowed, not the limits)")
    return Motion(pieces, tuple(notes), {"elements": list(elements), "tool_speed_m_s": float(tool_speed_m_s)})
