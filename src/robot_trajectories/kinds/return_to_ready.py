from __future__ import annotations

import math
from collections import deque

import numpy as np

from .. import kernels as k
from ..registry import LIMITS, kind
from ..segments import Motion
from ..time_law import TimeLaw
from ._lane_d import forward_or_none

# Ported from bam_ws src/dev/bam_fivebar_recovery/bam_fivebar_recovery/home.py (straight_waypoints :87, path_home :133, walk_home_sequence
# :163): waypoints every ``waypoint_deg`` along the straight line in joint space, kept only if every one is clear of the folds (the
# floor is the smaller of the asked margin and the home's own: the walk never asks for more room than its goal has), else the reach
# map's path. The rig's JointCommandSequence (per-step speed caps, gravity feed-forward, Reached / Settled conditions) is a
# command-sequence shape this library does not emit: what is kept is the geometry, the waypoint margins and the slow walk.


def straight_waypoints(start: np.ndarray, goal: np.ndarray, step_rad: float) -> np.ndarray:
    """Waypoints from ``start`` (excluded) to ``goal`` (included) at most ``step_rad`` apart on the largest joint."""
    pieces = max(1, int(math.ceil(float(np.max(np.abs(goal - start))) / step_rad)))
    return np.array([start + (goal - start) * (index / pieces) for index in range(1, pieces + 1)])


def _grid_route(reach, start: np.ndarray, goal: np.ndarray, usable_at) -> list[np.ndarray] | None:
    """A route through the reach map's grid (breadth first over neighbouring reachable cells, each also ``usable_at``), from the cell
    nearest ``start`` to the one nearest ``goal``; the cell centres only (the caller adds the ends and shortens it)."""
    step = reach.grid_step
    cells = reach.configurations[reach.valid]
    keys = np.round(cells / step).astype(int)
    index = {tuple(key): i for i, key in enumerate(keys)}

    def nearest(point: np.ndarray) -> int:
        return int(np.argmin(np.max(np.abs(cells - point) / step, axis=1)))

    first, last = nearest(start), nearest(goal)
    parents, queue = {first: None}, deque([first])
    offsets = [np.eye(len(step), dtype=int)[j] * sign for j in range(len(step)) for sign in (1, -1)]
    while queue:
        here = queue.popleft()
        if here == last:
            break
        for offset in offsets:
            there = index.get(tuple(keys[here] + offset))
            if there is not None and there not in parents and usable_at(cells[there]):
                parents[there] = here
                queue.append(there)
    if last not in parents:
        return None
    route, cursor = [], last
    while cursor is not None:
        route.append(cells[cursor])
        cursor = parents[cursor]
    return route[::-1]


@kind(use_cases=("homing", "smoke_test"), needs=(*LIMITS, "forward_kinematics", "home_pose"), unchecked=("collision_model",),
      catalog_id="return_to_ready")
def return_to_ready(space, *, waypoint_step_rad=0.087, speed_rad_s=0.1, settle_s=1.0, limit_margin_rad=0.02, label="return to ready",
                    start=None):
    """Walk from wherever the arm rests to the space's named home (READY) slowly along joint waypoints that keep margins, then
    settle: the rig's recover-yourself move as joint waypoints (a pose-free, task-free kind).

    A waypoint is clear when it is inside the position limits by ``limit_margin_rad`` (or the home's own margin where that is
    smaller), has a tool pose (a closed linkage clear of its folds), is inside the reach map's reachable set when the space has
    one, and is free of collision when the space checks it. The walk is the straight joint-space line when every waypoint (at most
    ``waypoint_step_rad`` apart) is clear; otherwise a route through the reach map's grid, shortened; otherwise one through the
    middle of the position limits; otherwise it is refused naming the first waypoint that is not clear. The largest joint moves at
    ``speed_rad_s`` (a trapezoid plateau; the limits slow it, never the other way), and the arm settles ``settle_s`` at READY.
    The start's own margin is the start's: only the waypoints after it are checked."""
    home = space.home
    bounds = space.bounds
    # WHY the smaller of the two margins: the walk never asks for more room than its goal has (home.py path_home: the real home's
    # closure margin was 0.041, 0.011 over the 0.03 floor)
    home_margin = float(np.min(np.minimum(home - bounds.low, bounds.high - home)))
    margin = max(0.0, min(limit_margin_rad, home_margin))
    inside = k.Bounds(bounds.joint_names, bounds.low + margin, bounds.high - margin, bounds.max_velocity, bounds.max_acceleration)

    def clear(q: np.ndarray) -> bool:
        if np.any(q < inside.low - 1e-9) or np.any(q > inside.high + 1e-9):
            return False
        if forward_or_none(space, q) is None:
            return False
        if space.reach is not None and not space.reach.nearest_is_reachable(q):
            return False
        return not (space.checks_collision and space.in_collision(q))

    # WHY a quarter of the waypoint step: a fold can be a thin band; the rig's 4-degree waypoints missed one a test found at 5 degrees
    check_step = waypoint_step_rad / 4.0

    def first_blocked(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        for point in straight_waypoints(a, b, check_step):
            if not clear(point):
                return point
        return None

    start = np.asarray(start, float)
    if float(np.max(np.abs(home - start))) < 1e-9:
        route, how, checked = [start], "already at READY", 0
    else:
        blocked = first_blocked(start, home)
        if blocked is None:
            route, how = [start, home], "the straight joint-space line"
        else:
            route, how = None, ""
            if space.reach is not None:
                grid = _grid_route(space.reach, start, home, clear)
                if grid is not None:
                    vertices = [start, *grid, home]
                    route, index = [start], 0
                    while index < len(vertices) - 1:  # shorten: the furthest vertex a straight, clear line reaches
                        jump = max(j for j in range(index + 1, len(vertices))
                                   if j == index + 1 or first_blocked(vertices[index], vertices[j]) is None)
                        route.append(vertices[jump])
                        index = jump
                    how = f"a route through the reach map's grid, shortened to {len(route) - 1} straight legs"
            if route is None:
                middle = 0.5 * (inside.low + inside.high)
                if clear(middle) and first_blocked(start, middle) is None and first_blocked(middle, home) is None:
                    route, how = [start, middle, home], "via the middle of the position limits"
            if route is None:
                raise ValueError(f"{label}: no clear way to READY: the straight joint line is blocked at "
                                 f"{np.round(np.degrees(blocked), 2).tolist()}° (limit margin {math.degrees(margin):.2f}°, a pose, "
                                 f"the reach set and collision all checked), and neither a reach-map route nor one through the middle "
                                 f"of the limits is clear")
        checked = sum(len(straight_waypoints(a, b, check_step)) for a, b in zip(route[:-1], route[1:], strict=True))
    pieces: list[k.Arrays] = []
    ramp = TimeLaw.trapezoid(move_s=1.0).ramp_fraction
    for index, (a, b) in enumerate(zip(route[:-1], route[1:], strict=True), start=1):
        distance = float(np.max(np.abs(b - a)))
        law = TimeLaw.trapezoid(move_s=max(distance / (speed_rad_s * (1.0 - ramp)), 0.1))
        pieces.append(law.leg(a, b, space.limits, space.rate_hz, label=f"{label}: walk {index}"))
    pieces.append(k.hold(home, max(settle_s, 0.0), space.rate_hz, label=f"{label}: settle at READY"))
    return Motion(pieces, (f"walk to READY by {how}; {checked} waypoints at most {math.degrees(check_step):.2f}° apart checked "
                           f"clear (limit margin {math.degrees(margin):.2f}°)",),
                  {"route": [[float(v) for v in q] for q in route], "waypoints_checked": int(checked)})
