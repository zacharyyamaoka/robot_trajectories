from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..cartesian import follow, line, tool_target_hold
from ..moving import CannotCatch
from ..registry import LIMITS, kind
from ..segments import Motion, Piece
from ..space import State
from ..time_law import TimeLaw
from . import _lane_f as f

DEFAULT_PLACE_BACK_M = 0.05


@kind(use_cases=("conveyor_pick_and_place", "cycle_time", "tracking_eval"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model",), catalog_id="moving_pick_and_place")
def moving_pick_and_place(space, target=None, place=None, *, lift_m=0.05, up=(0.0, 0.0, 1.0), sync_start_s=2.0, sync_window_s=0.5,
                          time_law=TimeLaw.minimum_jerk(peak_speed_fraction=0.3),
                          approach_time_law=TimeLaw.minimum_jerk(peak_speed_fraction=0.5), dwell_s=0.5, eef_step_m=0.001,
                          tolerance_m=1e-4, label="U on a moving point", start=None):
    """``pick_and_place``'s U where the first point moves: approach above it, down onto it, dwell on it, up off it, over to
    above the second point, down, dwell, up. ``lift_m`` along ``up`` (base frame); the tool keeps its start orientation.

    - ``target``: the first point, as ``moving_target_track`` takes it (a LinearMotion or a mapping of its fields, a
      PoseTrajectory, or a fixed point);
      ``place``: the second point (xyz, m), fixed. None: defaults from the space: the first point ``lift_m`` below the tool's
      start point at ``sync_start_s``, moving level at 5 cm/s; the second ``lift_m`` below it and 5 cm back along that line.
    - Down, dwell and up chase the first point (bam_traj_gen's ``sync_speed > 0``; ``TargetWaypoint(sync="chase")``): the tool
      is the point plus a vertical offset that ramps lift -> 0, holds 0 for ``sync_window_s`` from ``sync_start_s`` (the
      synchronised window, ``sync`` in the segment entry), and ramps 0 -> lift. Each ramp takes the time ``time_law`` gives the
      joint move it amounts to, lengthened until every joint is inside its limits.
    - The approach (``approach_time_law``) is a joint-space leg that ends above the point moving with it; it waits at rest first
      when it would arrive early. "Over" inherits the motion it starts in: a tool path from moving with the point to rest above
      the second point (velocity V (1 - s) + D s'/T), so the tool never stops on the moving point's line.
    - The second half is ``pick_and_place``'s: straight lines under ``time_law`` and a ``dwell_s`` dwell.

    One segment, its pieces the U's eight parts. WHY one: from the approach to "over" the motion never stops (the point moves
    on), so no part of it can be split off and run alone; the kind is one call. Refused with ``CannotCatch`` as
    ``moving_target_track`` is, ``time_law_error`` also when the descent cannot start after t = 0."""
    rate, notes = space.rate_hz, []
    direction = np.asarray(up, float)
    if np.linalg.norm(direction) < 1e-12:
        raise ValueError("up must be a nonzero direction")
    direction = direction / np.linalg.norm(direction)
    lift = lift_m * direction
    first, note_0 = f.sample_of(sync_start_s, rate, "sync_start_s")
    default = f.default_target(space, start, first / rate, shift=-lift, across=direction)
    moving = f.moving_target(default if target is None else target)
    if place is None:
        heading = np.asarray(moving.velocity(first / rate), float)[:3]
        heading = heading / np.linalg.norm(heading) if np.linalg.norm(heading) > 1e-12 else np.asarray(default["direction"])
        # WHY back along the line: the second point is where the first came from, so "over" never crosses ahead of it
        place = space.forward(start)[:3, 3] - lift - DEFAULT_PLACE_BACK_M * heading
    window, note_1 = f.sample_of(sync_window_s, rate, "sync_window_s")
    notes += [note for note in (note_0, note_1) if note]
    if first <= 0 or window <= 0:
        raise ValueError(f"sync_start_s and sync_window_s must be positive on the {rate:g} Hz grid, got {sync_start_s}, {sync_window_s}")
    last = first + window
    t0, t1 = first / rate, last / rate
    f.check_window(moving, t0, t1, "the synchronised dwell")
    rotation = space.forward(start)[:3, :3]
    fixed = f.fixed_offset(np.zeros(3))

    # the synchronised dwell first: it says whether the arm can keep pace at all
    dwell_path = f.chase(moving, np.arange(first, last + 1) / rate, rotation, fixed)
    seed = f.solve(space, dwell_path.pose(0), start, what=f"{label}: the tool on the first point at t = {t0:.3f} s", tolerance_m=tolerance_m)
    dwell = f.synced(space, dwell_path, seed, label="dwell at the first point", tolerance_m=tolerance_m)
    bottom_in = State(dwell.q[0], dwell.qd[0], dwell.qdd[0])
    bottom_out = State(dwell.q[-1], dwell.qd[-1], dwell.qdd[-1])

    def chased(name: str, ramp_from: np.ndarray, ramp_to: np.ndarray, at: int, seed_q: np.ndarray, **ends: State) -> Piece:
        """The tool chasing the point while its offset ramps (base frame), ending (``last=``: down) or beginning (``first=``: up)
        at sample ``at``."""
        above = f.solve(space, f.posed(rotation, np.asarray(moving.target(at / rate), float)[:3, 3] + lift), seed_q,
                        what=f"{label}: {1000 * lift_m:.1f} mm above the first point at t = {at / rate:.3f} s", tolerance_m=tolerance_m)
        steps = max(int(np.ceil(time_law.min_move_s * rate)), f.law_steps(space, time_law, seed_q, above))
        ending = "last" in ends

        def make(count: int, stride: int):
            begin = at - count if ending else at
            if begin < 1:
                raise CannotCatch("time_law_error", f"{label}: the descent onto the first point needs {count / rate:.3f} s inside the "
                                                    f"limits and an approach before it, but the point is met at t = {t0:.3f} s")
            rows = np.unique(np.append(np.arange(0, count + 1, stride), count))
            return f.chase(moving, (begin + rows) / rate, rotation, f.ramped_offset(ramp_from, ramp_to, begin / rate, count / rate))

        return f.fitted(space, make, steps, seed_q, label=name, refuse=f.out_of_reach(f"{label}: {name}"), tolerance_m=tolerance_m, **ends)

    down = chased("down to the first point", lift, np.zeros(3), first, bottom_in.q, last=bottom_in)
    up_1 = chased("up from the first point", np.zeros(3), lift, last, bottom_out.q, first=bottom_out)
    descent_s, rise_s = down.t[-1], up_1.t[-1]
    f.check_window(moving, t0 - descent_s, t1 + rise_s, "the chase (down, dwell, up)")
    top_in = State(down.q[0], down.qd[0], down.qdd[0])
    top_out = State(up_1.q[-1], up_1.qd[-1], up_1.qdd[-1])
    tool_top = up_1.channels["tool_target"][-1]
    _, velocity, _, _ = f.target_motion(moving, np.array([t1 + rise_s]))
    above_second = np.asarray(place, float) + lift
    goal = f.solve(space, f.posed(rotation, above_second), top_out.q, what=f"{label}: above the second point", tolerance_m=tolerance_m,
                   cannot_catch=False)

    def make_over(count: int, stride: int):
        return f.coast(tool_top, velocity[0], count, rate, rotation, begin_s=t1 + rise_s, end_point=above_second, stride=stride)

    over_steps = max(int(np.ceil(time_law.min_move_s * rate)), f.law_steps(space, time_law, top_out.q, goal), f.braking_steps(space, top_out.qd))
    over = f.fitted(space, make_over, over_steps, top_out.q, label="over", first=top_out, tolerance_m=tolerance_m)
    descent_start = first - (len(down.t) - 1)
    pieces, approach_notes = f.approach(space, start, top_in, descent_start, approach_time_law, label="approach",
                                        wait_label="wait at the start", point=down.channels["tool_target"][0], what=label)
    notes += approach_notes

    here = over.q[-1]
    pose_above, pose_place = f.posed(rotation, above_second), f.posed(rotation, np.asarray(place, float))
    lines = {"eef_step_m": eef_step_m, "tolerance_m": tolerance_m}
    down_2 = follow(space, here, line(pose_above, pose_place), time_law, length_m=lift_m, label="down to the second point", **lines)
    rest = k.hold(down_2.q[-1], dwell_s, rate, label="dwell at the second point")
    rest_2 = Piece(rest.t, rest.q, rest.qd, rest.qdd, rest.label, channels=tool_target_hold(place, len(rest.t)))
    up_2 = follow(space, down_2.q[-1], line(pose_place, pose_above), time_law, length_m=lift_m, label="up from the second point", **lines)
    pieces += [down, dwell, up_1, over, down_2, *([rest_2] if dwell_s > 0 else []), up_2]
    return Motion(pieces, notes, {"assumes": moving.assumes, "sync": [t0, t1]})
