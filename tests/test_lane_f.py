"""Lane F's kinds (moving targets) on the UR5e and the five-bar, the target moving at constant velocity.

    cd ~/robot_trajectories && /home/bam/robot_trajectories/.venv/bin/python tests/test_lane_f.py

Each kind is held to the kind contract (kind_contract.check_kind) and to what it is for, with rulers that can fail:
- ``synchronised(...)`` over the sync window (forward kinematics of the output within 1 mm of ``target(t)`` at every sample,
  the tool's velocity the target's at t0), and the same ruler failing on a target shifted 2 mm, or played 20 ms late;
- every commanded tool point within 1 mm by forward kinematics; an independent check, from forward kinematics and the target
  alone, that the chase keeps the tool over the moving point while it goes down and up;
- ``CannotCatch`` by cause: ``target_outruns_arm``, ``target_leaves_window``, ``intercept_out_of_reach``, ``time_law_error``;
- ``chain`` takes a moving boundary only where the states match.
The tracking error during each sync window is printed after the pass lines (mm).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

import _runner
import scenario
from fixtures import lane_f as lf
from fixtures import robots
from kind_contract import check_kind
from robot_trajectories import (CannotCatch, LinearMotion, MissingRung, State, bands, chain, hold, moving_pick_and_place,
                                moving_target_track, recipe, segment_info, synchronised, track_and_release)
from robot_trajectories.segments import TOOL_TARGET

MM = 0.001
ERRORS: dict[str, float] = {}  # sync window -> worst tracking error (m), printed at the end


def tracks(space, trajectory) -> float:
    """The worst distance (m) between forward kinematics of the output and the tool point it commanded (test_cartesian's)."""
    target = trajectory.channels[TOOL_TARGET]
    rows = np.flatnonzero(np.isfinite(target[:, 0]))
    assert len(rows) > 0, "no tool point was commanded"
    reached = space.tool_points(trajectory)[rows]
    return float(np.max(np.linalg.norm(reached - target[rows], axis=1)))


def reachable_fivebar(trajectory) -> None:
    off = [q for q in trajectory.positions[::5] if (passive := robots.fivebar_passive(*q)) is None or passive[2] < 0.2]
    assert not off, f"the five-bar leaves its reachable set (fold margin < 0.2) at {len(off)} samples"


def held_to_the_ruler(space, trajectory, target, t0, t1, name) -> None:
    """Synchronised over [t0, t1]; and the ruler fails on the same target shifted 2 mm, or 20 ms late."""
    moving = LinearMotion(**target) if isinstance(target, dict) else target
    report = synchronised(space, trajectory, moving, t0=t0, t1=t1)
    assert report, report.why
    assert report.velocity_error_m_s <= 0.01, report
    ERRORS[name] = report.worst_position_error_m
    shifted = moving.pose.copy()
    shifted[:3, 3] += (0.0, 0.0, 0.002) if not space.position_only else (0.0, 0.002, 0.0)
    off = synchronised(space, trajectory, dataclasses.replace(moving, pose=shifted), t0=t0, t1=t1)
    assert not off and "2.000 mm" in off.why, off
    late = dataclasses.replace(moving, stamp=moving.stamp + 0.002 / moving.speed)  # 2 mm behind along its way
    off = synchronised(space, trajectory, late, t0=t0, t1=t1)
    assert not off and "2.000 mm" in off.why, off


def refused(call, cause: str) -> CannotCatch:
    try:
        call()
    except CannotCatch as error:
        assert error.code == "cannot_catch" and error.cause == cause, (error.cause, str(error))
        assert error.detail, "a refusal carries its evidence"
        return error
    raise AssertionError(f"no CannotCatch({cause!r})")


# ---------------------------------------------------------------------- moving_target_track


def test_ur5e_moving_target_track():
    space = lf.ur5e()
    pose = space.forward(space.home)
    pose[:3, 3] += (0.05, -0.10, -0.05)
    target = lf.linear_target(pose, (0.0, 1.0, 0.0), 0.10)
    out = moving_target_track(space, target, sync_start_s=1.0, sync_end_s=2.0)
    check_kind(out, space)
    held_to_the_ruler(space, out, target, 1.0, 2.0, "UR5e moving_target_track, 1.0-2.0 s at 0.10 m/s")
    assert tracks(space, out) <= MM
    entry = segment_info(out)[0]
    assert entry["sync"] == [1.0, 2.0] and "constant velocity 0.1 m/s" in entry["assumes"], entry
    assert json.loads(recipe(out))["calls"][0]["target"]["speed"] == 0.1, "the target is in the recipe, as text"
    assert [band.label for band in bands(out)] == ["track a moving point: wait at the start", "track a moving point: approach",
                                                   "track a moving point: synchronised", "track a moving point: coast to rest"]


def test_ur5e_tracks_an_offset_in_the_target_frame():
    space = lf.ur5e()
    pose = space.forward(space.home)
    pose[:3, 3] += (0.0, -0.05, -0.05)
    target = lf.linear_target(pose, (1.0, 0.0, 0.0), 0.05)
    offset = np.array([0.0, 0.0, 0.03])
    out = moving_target_track(space, target, pose_in_target=tuple(offset), orientation="follow", sync_start_s=1.0, sync_end_s=1.5)
    moving = LinearMotion(**target)

    def above(t):
        point = moving.target(t)
        point[:3, 3] += point[:3, :3] @ offset
        return point

    assert synchronised(space, out, above, t0=1.0, t1=1.5), synchronised(space, out, above, t0=1.0, t1=1.5).why
    assert not synchronised(space, out, moving, t0=1.0, t1=1.5), "the offset is not on the target itself"


def test_fivebar_moving_target_track():
    space = lf.fivebar()
    pose = np.eye(4)
    pose[:3, 3] = space.forward(lf.FIVEBAR_START)[:3, 3] + (-0.03, 0.01, 0.0)
    target = lf.linear_target(pose, (1.0, 0.0, 0.0), 0.03)
    out = moving_target_track(space, target, sync_start_s=1.0, sync_end_s=2.0, start=lf.FIVEBAR_START)
    check_kind(out, space)
    reachable_fivebar(out)
    held_to_the_ruler(space, out, target, 1.0, 2.0, "five-bar moving_target_track, 1.0-2.0 s at 0.03 m/s")
    assert tracks(space, out) <= MM


def test_bam_traj_gen_sync_fields_map_on():
    """bam_traj_gen's chase (ArmParams.sync_dir / sync_speed, TrajGeneratorConfig.execution_latency) as the target."""
    space = lf.ur5e()
    pose = space.forward(space.home)
    pose[:3, 3] += (0.0, -0.08, -0.05)
    belt = LinearMotion.from_sync(pose, (0.0, 2.0, 0.0), 0.08, execution_latency=0.05)
    out = moving_target_track(space, dataclasses.asdict(belt), sync_start_s=1.0, sync_end_s=1.5)
    assert synchronised(space, out, belt, t0=1.0, t1=1.5)
    at_sync = space.forward(out.positions[int(np.searchsorted(out.t, 1.0))])[:3, 3]
    assert np.linalg.norm(at_sync - (pose[:3, 3] + (0.0, 0.08 * 1.05, 0.0))) <= 1e-6, "the latency moves where it is met"


# ---------------------------------------------------------------------- track_and_release, and the moving boundary


def _release_after_meet(space, start, target, up):
    meet = lf.meet_moving_point(space, target, sync_start_s=1.0, sync_end_s=1.3, start=start)
    release = track_and_release(space, target, dwell_s=0.3, leave_m=0.05, leave_direction=up, target_clock_s=float(meet.t[-1]),
                                start=meet)
    return meet, release


def _check_release(space, target, up, name):
    start = lf.FIVEBAR_START if space.position_only else space.home
    meet, release = _release_after_meet(space, start, target, up)
    assert abs(release.velocities[0] - meet.velocities[-1]).max() == 0.0, "the moving state is handed on exactly"
    joined = chain(meet, release)
    entries = segment_info(joined)
    assert entries[0]["ends"] == "moving" and entries[1]["starts"] == "moving", entries
    assert json.loads(recipe(release))["calls"][0]["start"]["qd"] == [float(v) for v in meet.velocities[-1]]
    check_kind(joined, space)
    if space.position_only:
        reachable_fivebar(joined)
    held_to_the_ruler(space, joined, target, 1.0, 1.6, name)
    assert tracks(space, joined) <= MM
    leave = next(band for band in bands(joined) if band.label == "track, then leave: leave")
    tool = space.tool_points(joined)
    begin, end = int(np.searchsorted(joined.t, leave.start_s)), len(joined.t) - 1
    moved = tool[end] - tool[begin]
    up = np.asarray(up, float) / np.linalg.norm(up)
    along = LinearMotion(**target)
    assert abs(float(moved @ up) - 0.05) <= MM, f"it left {1000 * float(moved @ up):.3f} mm along the direction, not 50"
    coasted = float(moved @ np.asarray(along.direction))
    assert abs(coasted - 0.5 * along.speed * (leave.end_s - leave.start_s)) <= MM, f"coasted {1000 * coasted:.3f} mm"


def test_ur5e_track_and_release_after_a_moving_end():
    space = lf.ur5e()
    pose = space.forward(space.home)
    pose[:3, 3] += (0.04, -0.08, -0.06)
    _check_release(space, lf.linear_target(pose, (0.0, 1.0, 0.0), 0.08), (0.0, 0.0, 1.0), "UR5e meet + track_and_release, 1.0-1.6 s at 0.08 m/s")


def test_fivebar_track_and_release_after_a_moving_end():
    space = lf.fivebar()
    pose = np.eye(4)
    pose[:3, 3] = space.forward(lf.FIVEBAR_START)[:3, 3] + (-0.02, 0.02, 0.0)
    _check_release(space, lf.linear_target(pose, (1.0, 0.0, 0.0), 0.03), lf.FIVEBAR_UP,
                   "five-bar meet + track_and_release, 1.0-1.6 s at 0.03 m/s")


def test_chain_takes_a_moving_boundary_only_where_the_states_match():
    space = lf.ur5e()
    pose = space.forward(space.home)
    pose[:3, 3] += (0.04, -0.08, -0.06)
    target = lf.linear_target(pose, (0.0, 1.0, 0.0), 0.08)
    meet = lf.meet_moving_point(space, target, sync_start_s=1.0, sync_end_s=1.3)
    other = lf.meet_moving_point(space, target, sync_start_s=1.0, sync_end_s=1.2)
    release_other = track_and_release(space, target, target_clock_s=float(other.t[-1]), start=other)
    chain(other, release_other)  # matched: accepted
    for following in (release_other, hold(space, 0.5, start=meet)):
        try:
            chain(meet, following)
        except ValueError as error:
            assert "ends moving" in str(error) and ("states differ" in str(error)), str(error)
        else:
            raise AssertionError("a moving end was joined to a start in another state")
    try:
        track_and_release(space, target, target_clock_s=float(meet.t[-1]), start=State(meet.positions[-1], np.zeros(6), np.zeros(6)))
    except ValueError as error:
        assert "not moving with the target" in str(error), str(error)
    else:
        raise AssertionError("a start at rest was taken as moving with a moving target")


# ---------------------------------------------------------------------- moving_pick_and_place


def test_ur5e_moving_pick_and_place():
    space = lf.ur5e()
    tool = space.forward(space.home)
    pose = tool.copy()
    pose[:3, 3] += (0.08, -0.14, -0.12)
    target = lf.linear_target(pose, (0.0, 1.0, 0.0), 0.10)
    place = tool[:3, 3] + (-0.08, 0.10, -0.12)
    cycle = moving_pick_and_place(space, target, place, lift_m=0.08, sync_start_s=2.0, sync_window_s=0.5)
    check_kind(cycle, space)
    names = [band.label for band in bands(cycle)]
    assert names[-8:] == ["approach", "down to the first point", "dwell at the first point", "up from the first point", "over",
                          "down to the second point", "dwell at the second point", "up from the second point"], names
    held_to_the_ruler(space, cycle, target, 2.0, 2.5, "UR5e moving_pick_and_place dwell, 2.0-2.5 s at 0.10 m/s")
    assert tracks(space, cycle) <= MM
    moving = LinearMotion(**target)
    reached = space.tool_points(cycle)
    for band in bands(cycle):
        rows = np.flatnonzero((cycle.t >= band.start_s - 1e-9) & (cycle.t <= band.end_s + 1e-9))
        if band.label in ("down to the first point", "up from the first point"):
            # the independent oracle: forward kinematics against the target alone (not the commanded channel)
            relative = np.array([reached[row] - moving.target(float(cycle.t[row]))[:3, 3] for row in rows])
            sideways = np.linalg.norm(relative - np.outer(relative[:, 2], (0.0, 0.0, 1.0)), axis=1).max()
            assert sideways <= MM, f"{band.label}: {1000 * sideways:.3f} mm off the moving point sideways"
            heights = relative[:, 2] if band.label.startswith("up") else relative[::-1, 2]
            assert abs(heights[0]) <= MM and abs(heights[-1] - 0.08) <= MM and np.all(np.diff(heights) >= -1e-6), band.label
        if band.label == "dwell at the second point":
            assert np.linalg.norm(reached[rows] - place, axis=1).max() <= MM, "the second dwell is not at the second point"
    assert segment_info(cycle)[0]["sync"] == [2.0, 2.5]


# ---------------------------------------------------------------------- refusals


def test_cannot_catch_by_cause():
    arm, five = lf.ur5e(), lf.fivebar()
    pose = arm.forward(arm.home)
    pose[:3, 3] += (0.05, -0.10, -0.05)
    plane = np.eye(4)
    plane[:3, 3] = five.forward(lf.FIVEBAR_START)[:3, 3]
    # WHY stamped at the sync time: a fast target is at the pose when it is met, so pace (not reach) is what fails
    fast = refused(lambda: moving_target_track(arm, lf.linear_target(pose, (0.0, 1.0, 0.0), 5.0, stamp=1.0), sync_start_s=1.0,
                                               sync_end_s=1.02), "target_outruns_arm")
    assert "keeping pace" in fast.detail and "limit" in fast.detail, fast.detail
    refused(lambda: moving_target_track(five, lf.linear_target(plane, (1.0, 0.0, 0.0), 3.0, stamp=1.0), sync_start_s=1.0, sync_end_s=1.02,
                                        start=lf.FIVEBAR_START), "target_outruns_arm")
    refused(lambda: moving_pick_and_place(arm, lf.linear_target(pose, (0.0, 1.0, 0.0), 5.0, stamp=2.0), pose[:3, 3]), "target_outruns_arm")
    windowed = lf.linear_target(pose, (0.0, 1.0, 0.0), 0.1, window=(1.5, 3.0))
    early = refused(lambda: moving_target_track(arm, windowed, sync_start_s=1.0, sync_end_s=2.0), "target_leaves_window")
    assert "1.000..2.000" in early.detail and "1.500 and 3.000" in early.detail, early.detail
    refused(lambda: moving_target_track(arm, windowed, sync_start_s=2.0, sync_end_s=3.5), "target_leaves_window")
    refused(lambda: moving_pick_and_place(arm, windowed, pose[:3, 3], sync_start_s=1.0), "target_leaves_window")
    refused(lambda: moving_target_track(arm, lf.linear_target(pose, (0.0, 0.0, -1.0), 0.4), sync_start_s=1.0, sync_end_s=2.0),
            "intercept_out_of_reach")
    refused(lambda: moving_target_track(five, lf.linear_target(plane, (1.0, 0.0, 0.0), 0.15), sync_start_s=1.0, sync_end_s=2.0,
                                        start=lf.FIVEBAR_START), "intercept_out_of_reach")
    far = pose.copy()
    far[:3, 3] += (0.15, -0.15, -0.10)
    late = refused(lambda: moving_target_track(arm, lf.linear_target(far, (0.0, 1.0, 0.0), 0.1), sync_start_s=0.06, sync_end_s=0.5),
                   "time_law_error")
    assert "t = 0.060 s" in late.detail, late.detail


# ---------------------------------------------------------------------- defaults from the space (f(space) alone)


def test_defaults_from_the_space():
    arm = lf.ur5e()
    for made in (moving_target_track(arm), track_and_release(arm), moving_pick_and_place(arm)):
        check_kind(made, arm)
        assert json.loads(recipe(made))["calls"][0]["target"] is None, "a default target is recorded as the default"
    out = moving_target_track(arm)
    assert synchronised(arm, out, LinearMotion(**_default(arm, 1.5)), t0=1.0, t1=2.0), "the default target is tracked"
    five = lf.fivebar()
    made = moving_target_track(five, start=lf.FIVEBAR_START)
    check_kind(made, five)
    reachable_fivebar(made)
    for plain in (scenario.pendulum_space(), scenario.fivebar_space(), scenario.ur5e_space()):
        for kind_function in (moving_target_track, track_and_release, moving_pick_and_place):
            try:
                kind_function(plain)
            except MissingRung as error:
                assert "with_kinematics" in str(error), str(error)
            else:
                raise AssertionError(f"{kind_function.__name__} generated without kinematics")


def _default(space, at_s):
    from robot_trajectories.kinds import _lane_f

    return _lane_f.default_target(space, space.home, at_s)


def main() -> int:
    code = _runner.run(globals(), "test_lane_f (moving targets)")
    for name, error in ERRORS.items():
        print(f"  sync  {name}: worst {1000 * error:.2e} mm")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
