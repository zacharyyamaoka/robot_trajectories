from __future__ import annotations

import math

import numpy as np

from .. import kernels as k
from ..cartesian import Unreachable
from ..registry import LIMITS, kind
from ..segments import Motion
from ._lane_c import peak_tool_speed, tool_line, tool_speed_law, with_point

DEFAULT_SPAN_M = 0.10  # the square a raster covers when no region is given and no reach map says where the robot can go


def _admit(space, point: np.ndarray, rotation: np.ndarray, seed: np.ndarray, tolerance_m: float) -> np.ndarray | None:
    """The joint solution at ``point`` if inverse kinematics finds one inside the limits that lands on it (and, with a reach map,
    in a reachable cell), else None."""
    pose = with_point(rotation, point)
    q = space.inverse(pose, seed)
    if q is None:
        return None
    bounds = space.bounds
    if np.any(q < bounds.low - 1e-9) or np.any(q > bounds.high + 1e-9):
        return None
    try:
        reached = space.forward(q)
    except ValueError:
        return None
    if float(np.linalg.norm(reached[:3, 3] - point)) > tolerance_m:
        return None
    if space.reach is not None and not space.reach.nearest_is_reachable(q):
        return None
    return q


def _region(space, start_point: np.ndarray, cell_m: float) -> tuple[float, float, float, float]:
    if space.reach is not None:
        points = []
        for q in space.reach.reachable:
            try:
                points.append(space.forward(q)[:3, 3])
            except ValueError:
                continue
        points = np.array([p for p in points if abs(p[2] - start_point[2]) <= max(cell_m, 1e-3)])
        if len(points):
            return float(points[:, 0].min()), float(points[:, 0].max()), float(points[:, 1].min()), float(points[:, 1].max())
    half = DEFAULT_SPAN_M / 2.0
    return start_point[0] - half, start_point[0] + half, start_point[1] - half, start_point[1] + half


@kind(use_cases=("workspace_coverage", "tracking_eval", "speed_limits"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      better_with={"reach_map": "the cells admitted by inverse kinematics and the joint limits alone"},
      unchecked=("collision_model", "jerk_limits"), catalog_id="cartesian_raster")
def cartesian_raster(space, region_m=None, *, cell_m=0.02, tool_speeds_m_s=(0.05, 0.1, 0.2, 0.35, 0.6, 1.0), ramp_time_s=0.25,
                     eef_step_m=0.001, jump_threshold=10.0, max_joint_step_rad=0.1, tolerance_m=1e-4, label="cartesian raster",
                     start=None):
    """The tool across a plane in straight rows, one row per row of ``cell_m`` cells, boustrophedon, stepping over to the next row
    (every move a ``cartesian.follow`` line, inverse kinematics every ``eef_step_m``), once per tool speed in ``tool_speeds_m_s``.

    The plane is the base frame's xy plane at the height the tool starts at (a planar five-bar's own plane); the tool keeps its
    start orientation. ``region_m`` = (x_min, x_max, y_min, y_max) in metres; without one, the bounding box of the reach map's
    tool points on the plane, or a 10 cm square about the start. A cell is reachable when inverse kinematics (seeded from its
    neighbour) finds a solution inside the limits that lands on its centre, and, with a reach map, in a reachable cell; a row is a
    run of two or more reachable cells in a line, and a row or step that cannot be followed is left out (the notes name them).
    The rows are the geometry: pass i covers them at ``tool_speeds_m_s[i]`` (odd passes run them backwards, so a pass starts where
    the last ended). Each line is a rest-to-rest trapezoid on the tool's path length, ``ramp_time_s`` up and down, lengthened where
    a joint limit binds (the notes give the speed reached).

    WHY no jerk time: the built-in time laws have no jerk-limited trapezoid; that profile is the ruckig plugin's (lane E)."""
    if cell_m <= 0 or ramp_time_s <= 0 or not tool_speeds_m_s or any(v <= 0 for v in tool_speeds_m_s):
        raise ValueError("cartesian_raster needs cell_m, ramp_time_s and every tool speed positive")
    here = space.forward(start)
    rotation, height = here[:3, :3], float(here[2, 3])
    x_min, x_max, y_min, y_max = (_region(space, here[:3, 3], cell_m) if region_m is None else tuple(float(v) for v in region_m))
    columns, rows = max(1, int((x_max - x_min) / cell_m + 1e-9)), max(1, int((y_max - y_min) / cell_m + 1e-9))
    xs = x_min + (np.arange(columns) + 0.5) * cell_m
    ys = y_min + (np.arange(rows) + 0.5) * cell_m
    solved: dict[tuple[int, int], np.ndarray] = {}
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            seeds = [solved[key] for key in ((row, column - 1), (row - 1, column)) if key in solved] + [start]
            for seed in seeds:
                q = _admit(space, np.array([x, y, height]), rotation, seed, tolerance_m)
                if q is not None:
                    solved[(row, column)] = q
                    break
    runs: list[tuple[tuple[int, int], tuple[int, int]]] = []  # (first, last) cell of each row of two or more reachable cells
    singles = 0
    for row, y in enumerate(ys):
        found = [column for column in range(columns) if (row, column) in solved]
        cut = np.split(found, np.flatnonzero(np.diff(found) > 1) + 1) if found else []
        in_row = [(int(part[0]), int(part[-1])) for part in cut if len(part)]
        singles += sum(1 for a, b in in_row if a == b)
        in_row = [(a, b) for a, b in in_row if b > a]
        if row % 2:
            in_row = [(b, a) for a, b in reversed(in_row)]
        runs += [((row, a), (row, b)) for a, b in in_row]
    if not runs:
        raise ValueError(f"cartesian_raster: none of the {columns * rows} cells of the region is reachable (a row needs two in a line)")
    line_args = {"eef_step_m": eef_step_m, "jump_threshold": jump_threshold, "max_joint_step_rad": max_joint_step_rad,
                 "tolerance_m": tolerance_m}
    point_of = lambda cell: np.array([xs[cell[1]], ys[cell[0]], height])  # noqa: E731
    pieces: list[k.Arrays] = []
    notes = [f"{label}: {len(runs)} rows over {columns} x {rows} cells of {1000 * cell_m:g} mm, {len(solved)} reachable"
             + (f", {singles} isolated cells without a row" if singles else "")]
    q = start

    def line_to(goal: np.ndarray, speed: float, name: str) -> bool:
        """One straight tool line to ``goal`` at ``speed``; False (nothing added) when it cannot be followed."""
        nonlocal q
        tool = space.forward(q)[:3, 3]
        length = float(np.linalg.norm(goal - tool))
        if length < 1e-6:
            return True
        try:
            piece = tool_line(space, q, with_point(rotation, goal), tool_speed_law(length, speed, ramp_time_s), label=name, **line_args)
        except Unreachable:
            return False
        pieces.append(piece)
        used.append(piece)
        q = piece.q[-1]
        return True

    def walk(cell_from: tuple[int, int], cell_to: tuple[int, int]) -> list[tuple[int, int]] | None:
        """The shortest chain of neighbouring reachable cells (8-connected) between two cells: the cell walk that joins rows whose
        ends a straight line cannot link (the region is not convex)."""
        previous = {cell_from: None}
        queue = [cell_from]
        for cell in queue:
            if cell == cell_to:
                break
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    other = (cell[0] + dr, cell[1] + dc)
                    if other in solved and other not in previous:
                        previous[other] = cell
                        queue.append(other)
        if cell_to not in previous:
            return None
        chain_ = [cell_to]
        while previous[chain_[-1]] is not None:
            chain_.append(previous[chain_[-1]])
        return chain_[::-1]

    here_cell = min(solved, key=lambda cell: float(np.linalg.norm(point_of(cell) - space.forward(start)[:3, 3])))
    at_cell: tuple[int, int] | None = None  # the cell the tool stands on; None until it has left the start
    for index, speed in enumerate(tool_speeds_m_s):
        order = runs if index % 2 == 0 else [(b, a) for a, b in reversed(runs)]
        used: list[k.Arrays] = []
        for number, (first, last) in enumerate(order, start=1):
            name = f"{label}: {speed:g} m/s"
            if first != at_cell:
                # WHY a straight step first, then the cell walk: a straight line is the short way; the region may not be convex
                if not line_to(point_of(first), speed, f"{name} step {number}"):
                    hops = walk(here_cell if at_cell is None else at_cell, first)
                    ok = hops is not None
                    for hop in (hops or [])[0 if at_cell is None else 1:]:
                        ok = ok and line_to(point_of(hop), speed, f"{name} step {number}")
                    if not ok:
                        notes.append(f"{name} row {number} left out: no way to its start across reachable cells")
                        continue
            at_cell = here_cell = first
            if not line_to(point_of(last), speed, f"{name} row {number}"):
                notes.append(f"{name} row {number} left out: the row cannot be followed")
                continue
            at_cell = here_cell = last
        fastest = max((peak_tool_speed(piece.channels["tool_target"], piece.t) for piece in used), default=0.0)
        verdict = "reached" if fastest >= 0.97 * speed else f"limited by the joints to {fastest:.3f} m/s"
        notes.append(f"{label}: pass {index + 1} at {speed:g} m/s: {verdict}")
    if not pieces:
        raise ValueError("cartesian_raster: no row or step could be followed")
    return Motion(pieces, notes)
