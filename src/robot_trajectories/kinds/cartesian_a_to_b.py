from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..cartesian import Unreachable, poses_for, solve_points, tool_target_hold
from ..registry import LIMITS, kind
from ..segments import TOOL_TARGET, Motion, Piece
from ..time_law import TimeLaw

DRAWS_PER_GOAL = 200  # WHY: a draw whose leg leaves the reachable set is thrown away; 200 tries find one on any map that has an interior


def _leaves_reach(space, leg: k.Arrays) -> int | None:
    """The first sample (every 5th) of a joint-space leg whose nearest reach-map cell is unreachable, or None (no map: None)."""
    if space.reach is None:
        return None
    for row in range(0, len(leg.t), 5):
        if not space.reach.nearest_is_reachable(leg.q[row]):
            return row
    return None


@kind(use_cases=("tracking_eval", "workspace_coverage", "cycle_time"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      better_with={"reach_map": "the goals drawn uniformly inside the joint room and the legs not checked against a reachable set"},
      unchecked=("collision_model", "jerk_limits"), catalog_id="cartesian_a_to_b")
def cartesian_a_to_b(space, goal=None, *, via=(), n_intermediate_points=0, seed=0, min_distance_m=0.02, orientation="hold",
                     time_law=TimeLaw.minimum_jerk(peak_speed_fraction=0.5), dwell_s=0.2, tolerance_m=1e-4, label="pose to pose",
                     start=None):
    """The tool from where it is to a goal pose, by inverse kinematics at the ends and one joint-space leg between them (rest to
    rest under ``time_law``), through ``via`` poses first when given: the "A to B" of bam_reach's trajectory dataset.

    ``goal`` (an xyz point or a 4x4 pose; ``orientation`` "hold" or "follow" as ``cartesian_waypoints``) and each of ``via`` are
    given, or, with ``goal=None``, DRAWN: ``n_intermediate_points`` + 1 poses (the last the goal) from the reach map's reachable
    set (or, with none, uniformly inside the space's room), seeded by ``seed``, each at least ``min_distance_m`` from the last.
    A drawn pose is the forward kinematics of a reachable configuration and is solved again by inverse kinematics from it, so
    it is reachable by construction; a draw whose joint-space leg leaves the reachable set is thrown away and drawn again.
    The tool does not move in a straight line between poses (that is ``cartesian_path``). A given pose with no solution inside
    the limits, or a leg that leaves the reach map, is refused, naming it.

    WHY the reach map is a cache here, not an input: the kind runs on the kinematics alone (a draw then comes from the room);
    with a map it samples where the robot can go and holds every leg to it."""
    rng = np.random.default_rng(seed)
    pieces: list[k.Arrays] = []
    notes: list[str] = []
    here_q, here_point = start, space.forward(start)[:3, 3]

    def leg_to(q: np.ndarray, point: np.ndarray, name: str) -> list[k.Arrays]:
        leg = time_law.leg(here_q, q, space.limits, space.rate_hz, label=f"{label}: {name}")
        target = np.full((len(leg.t), 3), np.nan)
        target[-1] = point  # WHY only the last row: between poses the tool's path is not commanded
        out: list[k.Arrays] = [Piece(leg.t, leg.q, leg.qd, leg.qdd, leg.label, channels={TOOL_TARGET: target})]
        if dwell_s > 0:
            rest = k.hold(q, dwell_s, space.rate_hz, label=f"{label}: dwell")
            out.append(Piece(rest.t, rest.q, rest.qd, rest.qdd, rest.label, channels=tool_target_hold(point, len(rest.t))))
        return out

    if goal is not None:
        targets = poses_for(space, start, [*via, goal], orientation)
        solutions = solve_points(space, start, targets, what=label, tolerance_m=tolerance_m)
        for index, (q, target) in enumerate(zip(solutions, targets, strict=True), start=1):
            name = "leg to the goal" if index == len(targets) else f"leg to via {index}"
            leg = time_law.leg(here_q, q, space.limits, space.rate_hz, label="check")
            row = _leaves_reach(space, leg)
            if row is not None:
                raise Unreachable(label, index=index, count=len(targets), point=target[:3, 3],
                                  why=f"the joint-space leg leaves the reach map at its sample {row}: stop at a pose between, or use cartesian_path")
            pieces += leg_to(q, target[:3, 3], name)
            here_q, here_point = q, target[:3, 3]
        return Motion(pieces, notes)
    if via:
        raise ValueError("cartesian_a_to_b: via poses need a goal (with goal=None, n_intermediate_points draws the poses between)")
    pool = space.reach.reachable if space.reach is not None else None
    room = space.room(start)
    for index in range(1, n_intermediate_points + 2):
        name = "leg to the goal" if index == n_intermediate_points + 1 else f"leg to pose {index}"
        for attempt in range(DRAWS_PER_GOAL):
            q = pool[rng.integers(len(pool))] if pool is not None else rng.uniform(room.low, room.high)
            try:
                pose = space.forward(q)
            except ValueError:
                continue
            if float(np.linalg.norm(pose[:3, 3] - here_point)) < min_distance_m:
                continue
            solved = space.inverse(pose, q)  # WHY seeded at the draw: inverse kinematics at the end, held to the configuration drawn
            if solved is None or float(np.linalg.norm(space.forward(solved)[:3, 3] - pose[:3, 3])) > tolerance_m \
                    or np.any(solved < space.bounds.low - 1e-9) or np.any(solved > space.bounds.high + 1e-9):
                continue
            leg = time_law.leg(here_q, solved, space.limits, space.rate_hz, label="check")
            if _leaves_reach(space, leg) is not None:
                continue
            pieces += leg_to(solved, pose[:3, 3], name)
            notes.append(f"{label}: {name.replace('leg to ', '')} at ({', '.join(f'{1000 * v:.1f}' for v in pose[:3, 3])}) mm, "
                         f"draw {attempt + 1} of seed {seed}")
            here_q, here_point = solved, pose[:3, 3]
            break
        else:
            raise ValueError(f"cartesian_a_to_b: no reachable pose at least {1000 * min_distance_m:.0f} mm away with a joint-space leg "
                             f"inside the reachable set in {DRAWS_PER_GOAL} draws (seed {seed})")
    return Motion(pieces, notes)
