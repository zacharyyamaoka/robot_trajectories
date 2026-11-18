"""Cartesian and multi-waypoint kinds with the kinematics passed in, on the UR5e (6 DOF, a full pose) and the five-bar (2 DOF, a
planar point): a straight line, the U pick-and-place and Cartesian waypoints, plus joint waypoints.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_cartesian.py

Each output must: meet the kind contract (inside limits, rebuilt from its recipe, ...: kind_contract.check_kind); track the
commanded tool points to within 1 mm by forward kinematics of the output (the ``tool_target`` channel); and, as an
independent oracle that does not read that channel, keep every sample of every straight line within 1 mm of the line between
its two commanded points. An unreachable point is refused with the failing sample and why; a line through an obstacle in an
injected scene is refused at its first colliding sample.
"""

from __future__ import annotations

import json
import math

import numpy as np

import _runner
import scenario
from fixtures import robots
from kind_contract import check_kind
from robot_trajectories import (JointSpace, Unreachable, UnsafeTrajectory, cartesian_path, cartesian_waypoints, joint_waypoints,
                                labels, pick_and_place, segment_info, split)
from robot_trajectories.cartesian import rotation_angle
from robot_trajectories.segments import TOOL_TARGET

MM = 0.001
FIVEBAR_START = np.radians([128.0, -126.0])  # a configuration well inside the reach map: the serial elbow at -60°, fold margin 1.0


def ur5e() -> JointSpace:
    return scenario.ur5e_space().with_kinematics(robots.UR5eKinematics())


def fivebar() -> JointSpace:
    return JointSpace.from_limits(robots.FIVEBAR_LIMITS, home=np.radians([100.0, -76.0])).with_kinematics(robots.FiveBarKinematics())


def tracks(space: JointSpace, trajectory) -> float:
    """The worst distance (m) between forward kinematics of the output and the tool point it commanded, where it commanded one."""
    target = trajectory.channels[TOOL_TARGET]
    rows = np.flatnonzero(np.isfinite(target[:, 0]))
    assert len(rows) > 0, "no tool point was commanded"
    reached = space.tool_points(trajectory)[rows]
    return float(np.max(np.linalg.norm(reached - target[rows], axis=1)))


def off_line(space: JointSpace, trajectory) -> float:
    """The independent oracle: every cartesian_path segment's samples against the straight line between its commanded points."""
    worst = 0.0
    for piece, entry in zip(split(trajectory), segment_info(trajectory), strict=True):
        call = entry.get("call") or {}
        if call.get("kind") != "cartesian_path":
            continue
        points = [np.asarray(item, float) for item in call["points_or_poses"]]
        vertices = [space.forward(piece.positions[0])[:3, 3]] + [p[:3, 3] if p.shape == (4, 4) else p for p in points]
        reached = space.tool_points(piece)
        for point in reached:
            distances = []
            for a, b in zip(vertices[:-1], vertices[1:]):
                direction = b - a
                along = 0.0 if not np.any(direction) else float(np.clip(np.dot(point - a, direction) / np.dot(direction, direction), 0, 1))
                distances.append(float(np.linalg.norm(point - (a + along * direction))))
            worst = max(worst, min(distances))
    return worst


def reachable_fivebar(trajectory) -> None:
    off = [q for q in trajectory.positions[::5] if (passive := robots.fivebar_passive(*q)) is None or passive[2] < 0.2]
    assert not off, f"the five-bar leaves its reachable set (fold margin < 0.2) at {len(off)} samples"


# ---------------------------------------------------------------------- the UR5e


def test_ur5e_forward_kinematics_is_the_published_zero_pose():
    tool = robots.UR5eKinematics().forward(np.zeros(6))[:3, 3]
    assert np.allclose(tool, (-0.8172, -0.2329, 0.0628), atol=1e-4), tool  # -(a2 + a3), -(d4 + d6), d1 - d5


def test_ur5e_straight_line():
    space = ur5e()
    start_pose = space.forward(space.home)
    line = cartesian_path(space, [start_pose[:3, 3] + (0.10, 0.05, -0.05)])
    check_kind(line, space)
    assert tracks(space, line) <= MM and off_line(space, line) <= MM
    turn = max(rotation_angle(space.forward(q)[:3, :3], start_pose[:3, :3]) for q in line.positions[::10])
    assert turn <= math.radians(0.1), f"the held orientation turned {math.degrees(turn):.3f}°"


def test_ur5e_u_pick_and_place():
    space = ur5e()
    tool = space.forward(space.home)[:3, 3]
    first, second = tool + (0.08, -0.08, -0.12), tool + (-0.08, 0.10, -0.12)
    cycle = pick_and_place(space, first, second, lift_m=0.08)
    check_kind(cycle, space)
    assert labels(cycle) == ["approach", "down to the first point", "dwell at the first point", "up from the first point", "over",
                             "down to the second point", "dwell at the second point", "up from the second point"], labels(cycle)
    assert tracks(space, cycle) <= MM and off_line(space, cycle) <= MM
    pieces = split(cycle)
    for index, point in ((2, first), (6, second)):
        assert np.linalg.norm(space.tool_points(pieces[index]) - point, axis=1).max() <= MM, f"segment {index} is not at its point"
    assert all(entry.get("wait_after") for entry in (segment_info(cycle)[2], segment_info(cycle)[6]))
    assert json.loads(cycle.header.meta["recipe"])["calls"][0]["kind"] == "pick_and_place", "one call in the recipe"


def test_ur5e_cartesian_waypoints():
    space = ur5e()
    tool = space.forward(space.home)[:3, 3]
    points = [tool + (0.10, 0.0, 0.0), tool + (0.0, 0.10, -0.05), tool + (-0.05, -0.05, 0.05)]
    tour = cartesian_waypoints(space, points, dwell_s=0.3)
    check_kind(tour, space)
    assert tracks(space, tour) <= MM
    dwells = [band for band in __import__("robot_trajectories").bands(tour) if band.label.endswith("dwell")]
    assert len(dwells) == 3


def test_ur5e_refuses_an_unreachable_point():
    space = ur5e()
    tool = space.forward(space.home)[:3, 3]
    for call in (lambda: cartesian_path(space, [tool + (2.0, 0.0, 0.0)]), lambda: cartesian_waypoints(space, [tool + (2.0, 0.0, 0.0)]),
                 lambda: pick_and_place(space, tool + (2.0, 0.0, -0.1), tool)):
        try:
            call()
        except Unreachable as error:
            assert "sample" in str(error) and "unreachable" in str(error) and error.why, str(error)
        else:
            raise AssertionError("a point 2 m away was not refused")


def test_ur5e_line_through_an_obstacle_is_refused_at_its_first_colliding_sample():
    space = ur5e()
    tool = space.forward(space.home)[:3, 3]
    end = tool + (0.10, 0.05, -0.05)
    middle = 0.5 * (tool + end)
    blocked = space.with_collision(robots.UR5eScene(box_min=tuple(middle - 0.01), box_max=tuple(middle + 0.01)))
    try:
        cartesian_path(blocked, [end])
    except UnsafeTrajectory as error:
        text = str(error)
        assert "collides with the scene" in text and "sample" in text, text
        first = int(text.split("sample ")[1].split(" ")[0])
        assert 0 < first, first
    else:
        raise AssertionError("a line through the box was not refused")
    clear = space.with_collision(robots.UR5eScene())
    check_kind(cartesian_path(clear, [end]), clear)


# ---------------------------------------------------------------------- the five-bar


def test_fivebar_closed_form_round_trip():
    kinematics = robots.FiveBarKinematics()
    for q in (FIVEBAR_START, np.radians([120.0, -144.0]), np.radians([132.0, -114.0])):
        back = kinematics.inverse(kinematics.forward(q), q)
        assert back is not None and np.allclose(back, q, atol=1e-9), (np.degrees(q), back)


def test_fivebar_straight_line():
    space = fivebar()
    tool = space.forward(FIVEBAR_START)[:3, 3]
    line = cartesian_path(space, [tool + (0.03, 0.0, 0.0), tool + (0.03, 0.03, 0.0)], start=FIVEBAR_START)
    check_kind(line, space)
    reachable_fivebar(line)
    assert tracks(space, line) <= MM and off_line(space, line) <= MM


def test_fivebar_u_pick_and_place():
    space = fivebar()
    tool = space.forward(FIVEBAR_START)[:3, 3]
    # WHY up is -y: the plane frame's y-hat hangs downward (fb_bench/mapping.py)
    cycle = pick_and_place(space, tool + (-0.03, 0.02, 0.0), tool + (0.03, 0.02, 0.0), lift_m=0.03, up=(0.0, -1.0, 0.0),
                           start=FIVEBAR_START)
    check_kind(cycle, space)
    reachable_fivebar(cycle)
    assert tracks(space, cycle) <= MM and off_line(space, cycle) <= MM


def test_fivebar_cartesian_waypoints():
    space = fivebar()
    tool = space.forward(FIVEBAR_START)[:3, 3]
    tour = cartesian_waypoints(space, [tool + (0.03, 0.0, 0.0), tool + (0.0, 0.03, 0.0), tool + (-0.03, 0.02, 0.0)],
                               start=FIVEBAR_START)
    check_kind(tour, space)
    reachable_fivebar(tour)
    assert tracks(space, tour) <= MM


def test_fivebar_refuses_an_unreachable_point():
    space = fivebar()
    for call in (lambda: cartesian_path(space, [(1.0, 0.0, 0.0)], start=FIVEBAR_START),
                 lambda: cartesian_waypoints(space, [(1.0, 0.0, 0.0)], start=FIVEBAR_START)):
        try:
            call()
        except Unreachable as error:
            assert "no inverse-kinematics solution" in str(error) or "unreachable" in str(error), str(error)
        else:
            raise AssertionError("a point 1 m away was not refused")


# ---------------------------------------------------------------------- joint waypoints


def test_joint_waypoints_on_every_robot():
    for space in (scenario.pendulum_space(), fivebar(), ur5e()):
        half = 0.4 * (space.room(space.home).high - space.home)
        waypoints = [space.home + half, space.home - half, space.home + 0.5 * half]
        tour = joint_waypoints(space, waypoints, dwell_s=0.25)
        check_kind(tour, space)
        dwell_rows = [band for band in __import__("robot_trajectories").bands(tour) if band.label.endswith("dwell")]
        assert len(dwell_rows) == 3
        index = np.searchsorted(tour.t, dwell_rows[0].end_s)
        assert np.allclose(tour.positions[index], waypoints[0], atol=1e-12), "the first dwell is not at the first waypoint"
    try:
        joint_waypoints(ur5e(), [np.zeros(6)])
    except ValueError as error:
        assert "outside its limits" in str(error) and "waypoint 0" in str(error), str(error)
    else:
        raise AssertionError("a waypoint outside the limits was not refused")


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_cartesian (FK/IK passed in, joint trajectories out)"))
