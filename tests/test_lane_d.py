"""Lane D's kinds (standard tests, calibration pose sets, task cycles), each held to the kind contract on the robots it applies to,
and to what it is for.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_lane_d.py

Every Cartesian kind is held to its commanded tool points by forward kinematics within 1 mm (the ``tool_target`` channel), and to
an independent oracle that re-derives the geometry from the definition (``fixtures/lane_d.py``): ISO 9283's P1-P5, a circle's
radius, a polyline's distance, a view's aim. An unreachable request is refused with a named reason, never clipped.
"""

from __future__ import annotations

import json
import math

import numpy as np

import _runner
from fixtures import lane_d as fixtures, robots
from kind_contract import TASK_WORDS, check_kind
from robot_trajectories import (JointSpace, MissingRung, Unreachable, adept_cycle, bands, camera_board_views, catalog, circular_test,
                                hand_eye_station_set, iso9283_path, iso9283_pose_cube, kinds, kinematic_calibration_pose_set, labels,
                                rebuild, return_to_ready, segment_info, split, touch_down_contact)
from robot_trajectories.cartesian import rotation_angle
from robot_trajectories.segments import TOOL_TARGET

MM = 0.001
LANE_D = ("iso9283_pose_cube", "iso9283_path", "circular_test", "kinematic_calibration_pose_set", "hand_eye_station_set",
          "camera_board_views", "return_to_ready", "adept_cycle", "touch_down_contact")
UR5E_SPACES = (("ur5e", fixtures.ur5e, None, {}),)
FIVEBAR = (fixtures.fivebar, fixtures.FIVEBAR_START)


def tracks(space: JointSpace, trajectory) -> float:
    """The worst distance (m) between forward kinematics of the output and the tool point it commanded, where it commanded one."""
    target = trajectory.channels[TOOL_TARGET]
    rows = np.flatnonzero(np.isfinite(target[:, 0]))
    assert len(rows) > 0, "no tool point was commanded"
    return float(np.max(np.linalg.norm(space.tool_points(trajectory)[rows] - target[rows], axis=1)))


def entry(trajectory) -> dict:
    return segment_info(trajectory)[0]


def stations_of(trajectory) -> list[np.ndarray]:
    return [np.array(station) for station in entry(trajectory)["stations"]]


def band_rows(trajectory, band) -> np.ndarray:
    return (trajectory.t >= band.start_s - 1e-12) & (trajectory.t <= band.end_s + 1e-12)


def station_rows(trajectory) -> list[np.ndarray]:
    """The samples of each station's dwell, in order."""
    return [band_rows(trajectory, band) for band in bands(trajectory) if ": station " in band.label]


def refused(call, *errors, contains: str = "") -> str:
    try:
        call()
    except errors as error:
        assert contains in str(error), f"{contains!r} not in {str(error)!r}"
        return str(error)
    raise AssertionError(f"not refused (wanted {[e.__name__ for e in errors]})")


# ====================================================================== the catalog and the vocabulary


def test_the_nine_kinds_claim_their_rows_and_use_no_task_words():
    registry = kinds()
    todo = catalog().todo.names
    for name in LANE_D:
        assert name in registry and registry[name].catalog_id == name, name
        assert name not in todo, f"{name} is still listed as not built"
        words = " ".join([name, *registry[name].parameters]).lower() + " "
        for word in TASK_WORDS:
            assert word not in words, f"task word {word.strip()!r} in {name}'s name or parameters"
    # the pose sets declare their output shape in the entry (KINDS_TODO rule 1): stations, and a dwell piece labelled per station
    assert registry["adept_cycle"].higher_order


# ====================================================================== standard tests


def test_iso9283_pose_cube_ur5e():
    space = fixtures.ur5e()
    cycles, dwell = 2, 0.5
    trajectory = iso9283_pose_cube(space, cycles=cycles, dwell_s=dwell)
    check_kind(trajectory, space)
    assert tracks(space, trajectory) <= MM
    info = entry(trajectory)
    assert info["poses"] == ["P1", "P2", "P3", "P4", "P5"] * cycles and len(info["stations"]) == 5 * cycles
    # the stations' tool points are the standard's P1-P5 (re-derived apart from the kind), every cycle
    cube = info["cube"]
    expected = fixtures.pose_cube_points(np.array(cube["centre_m"]), cube["edge_m"])
    rows = station_rows(trajectory)
    assert len(rows) == 5 * cycles
    tool = space.tool_points(trajectory)
    for index, mask in enumerate(rows):
        reached = tool[mask]
        assert np.linalg.norm(reached - expected[index % 5], axis=1).max() <= MM, f"station {index + 1} is not P{index % 5 + 1}"
        assert trajectory.t[mask][-1] - trajectory.t[mask][0] >= dwell - 1e-9, "a station is not dwelt at"
    # all three axes move between P1..P5 (the diagonal plane's point), and the orientation is held at every station
    assert all(np.ptp(expected[:, axis]) > 0.1 for axis in range(3))
    start_rotation = space.forward(space.home)[:3, :3]
    assert max(rotation_angle(space.forward(q)[:3, :3], start_rotation) for q in stations_of(trajectory)) <= 2e-3
    # repeatability starts from commanding the same configuration every cycle
    first, second = stations_of(trajectory)[:5], stations_of(trajectory)[5:]
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    # the whole cube is inside the reach: its eight corners solve (the kind's own claim, checked here by another inverse call)
    half = 0.5 * cube["edge_m"]
    for sx in (-half, half):
        for sy in (-half, half):
            for sz in (-half, half):
                pose = space.forward(space.home).copy()
                pose[:3, 3] = np.array(cube["centre_m"]) + (sx, sy, sz)
                solution = space.inverse(pose, stations_of(trajectory)[0])
                assert solution is not None and np.linalg.norm(space.forward(solution)[:3, 3] - pose[:3, 3]) <= MM, "a corner is unreachable"


def test_iso9283_pose_cube_speed_fraction_and_scene():
    space = fixtures.ur5e()
    fast = iso9283_pose_cube(space, cycles=1, dwell_s=0.3, speed_fraction=1.0)
    slow = iso9283_pose_cube(space, cycles=1, dwell_s=0.3, speed_fraction=0.1)
    check_kind(slow, space)
    assert slow.t[-1] > 1.5 * fast.t[-1], "a tenth of the speed is not slower"
    scene = space.with_collision(robots.UR5eScene())
    clear = iso9283_pose_cube(scene, cycles=1, dwell_s=0.3)
    check_kind(clear, scene)


def test_iso9283_pose_cube_refusals():
    space = fixtures.ur5e()
    message = refused(lambda: iso9283_pose_cube(space, cycles=1, cube_edge_m=1.5), ValueError, contains="does not fit")
    assert "largest edge" in message, message
    refused(lambda: iso9283_pose_cube(space, cycles=1, centre_m=(2.0, 0.0, 0.0)), ValueError, contains="does not fit")
    # a planar robot has no cube: its corners are not on its plane
    flat = fixtures.fivebar()
    refused(lambda: iso9283_pose_cube(flat, cycles=1, cube_edge_m=0.05, start=fixtures.FIVEBAR_START), ValueError, contains="planar")
    refused(lambda: iso9283_pose_cube(space, cycles=0), ValueError, contains="cycles")
    bare = JointSpace.from_limits(robots.FIVEBAR_LIMITS)
    refused(lambda: iso9283_pose_cube(bare), MissingRung)


def test_iso9283_path_on_both_robots():
    for space, start, kwargs in ((fixtures.ur5e(), None, {}),
                                 (fixtures.fivebar(), fixtures.FIVEBAR_START,
                                  dict(circle_diameter_m=0.06, line_length_m=0.05, rectangle_m=(0.05, 0.04)))):
        trajectory = iso9283_path(space, cycles=2, start=start, **kwargs)
        check_kind(trajectory, space)
        assert tracks(space, trajectory) <= MM
        origin = space.forward(space.start_at(start))[:3, 3]
        u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        length = kwargs.get("line_length_m", 0.1)
        diameter = kwargs.get("circle_diameter_m", 0.1)
        width, height = kwargs.get("rectangle_m", (0.1, 0.06))
        tool = space.tool_points(trajectory)
        seen = {"line": 0, "circle": 0, "rectangle": 0}
        for band in bands(trajectory):
            element = band.label.split(": ")[1].split(",")[0]
            if element not in seen:
                continue
            seen[element] += 1
            points = tool[band_rows(trajectory, band)]
            if element == "line":
                assert fixtures.polyline_error(points, [origin, origin + length * u]) <= MM
            elif element == "circle":
                assert fixtures.circle_error(points, origin + 0.5 * diameter * v, 0.5 * diameter, np.array([0.0, 0.0, 1.0])) <= MM
            else:
                corners = [origin, origin + width * u, origin + width * u + height * v, origin + height * v, origin]
                assert fixtures.polyline_error(points, corners) <= MM
        assert seen == {"line": 4, "circle": 2, "rectangle": 8}, seen  # (out, back) x 2 cycles; 1 revolution x 2; 4 sides x 2


def test_iso9283_path_holds_its_tool_speed():
    space = fixtures.ur5e()
    trajectory = iso9283_path(space, elements=("line",), cycles=1, tool_speed_m_s=0.1, line_length_m=0.15)
    out = next(band for band in bands(trajectory) if band.label.endswith("out"))
    speeds = fixtures.tool_speeds(space, trajectory, band_rows(trajectory, out))
    assert abs(speeds.max() - 0.1) <= 0.003, f"peak tool speed {speeds.max():.4f} m/s, asked 0.1"
    assert np.mean(speeds >= 0.097) >= 0.4, "no plateau at the stated speed"
    # a speed the limits cannot give is slowed and said, never the limits
    slow = iso9283_path(space, elements=("line",), cycles=1, tool_speed_m_s=50.0, line_length_m=0.15)
    check_kind(slow, space)
    assert any("joint limits bind" in note for note in segment_info(slow)[0].get("notes", ())), segment_info(slow)[0]


def test_iso9283_path_refuses_what_it_cannot_reach():
    space = fixtures.ur5e()
    refused(lambda: iso9283_path(space, cycles=1, circle_diameter_m=3.0, elements=("circle",)), Unreachable, contains="unreachable")
    refused(lambda: iso9283_path(space, cycles=1, line_length_m=2.0, elements=("line",)), Unreachable, contains="sample")
    refused(lambda: iso9283_path(space, elements=("spiral",)), ValueError, contains="elements")
    flat = fixtures.fivebar()
    refused(lambda: iso9283_path(flat, cycles=1, line_length_m=1.0, elements=("line",), start=fixtures.FIVEBAR_START), Unreachable)


def test_circular_test_on_both_robots():
    for space, start, diameter in ((fixtures.ur5e(), None, 0.1), (fixtures.fivebar(), fixtures.FIVEBAR_START, 0.06)):
        trajectory = circular_test(space, diameter_m=diameter, start=start)
        check_kind(trajectory, space)
        assert tracks(space, trajectory) <= MM
        origin = space.forward(space.start_at(start))[:3, 3]
        centre, normal = origin - np.array([0.5 * diameter, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        tool = space.tool_points(trajectory)
        assert fixtures.circle_error(tool, centre, 0.5 * diameter, normal) <= MM, "the tool leaves the circle"
        runs = [band for band in bands(trajectory)]
        assert [band.label.split(": ")[1] for band in runs] == ["counter-clockwise", "clockwise"]
        for band, sign in zip(runs, (1.0, -1.0), strict=True):
            points = tool[band_rows(trajectory, band)] - centre
            angle = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
            assert sign * (angle[-1] - angle[0]) >= 2.0 * math.pi + 0.99 * 2.0 * math.pi / 8.0, "less than a revolution plus its overlap"
            assert np.all(sign * np.diff(angle) >= -1e-9), "the circle reverses within a direction"
        # the second run starts where the first ended: the pair is one continuous test
        assert np.linalg.norm(tool[np.searchsorted(trajectory.t, runs[0].end_s)] - tool[np.searchsorted(trajectory.t, runs[1].start_s)]) <= MM


def test_circular_test_holds_its_feed_and_one_direction():
    space = fixtures.ur5e()
    trajectory = circular_test(space, feed_m_s=0.05, directions="cw")
    check_kind(trajectory, space)
    assert labels(trajectory) == ["circular test"] and [b.label for b in bands(trajectory)] == ["circular test: clockwise"]
    speeds = fixtures.tool_speeds(space, trajectory, np.ones(len(trajectory.t), bool))
    assert abs(speeds.max() - 0.05) <= 0.0015 and np.mean(speeds >= 0.0485) >= 0.4, f"feed {speeds.max():.4f} m/s"
    refused(lambda: circular_test(space, diameter_m=3.0), Unreachable, contains="unreachable")
    refused(lambda: circular_test(space, directions="sideways"), ValueError, contains="directions")


# ====================================================================== calibration


def test_kinematic_calibration_pose_set_on_both_robots():
    for build, start in ((fixtures.ur5e, None), (fixtures.fivebar, fixtures.FIVEBAR_START)):
        space = build()
        trajectory = kinematic_calibration_pose_set(space, poses=10, candidates=300, dwell_s=0.4, start=start)
        check_kind(trajectory, space)
        info = entry(trajectory)
        stations = stations_of(trajectory)
        assert len(stations) == 10 and len({tuple(np.round(s, 9)) for s in stations}) == 10
        room = space.room(space.start_at(start))
        for q in stations:
            assert np.all(q >= room.low - 1e-9) and np.all(q <= room.high + 1e-9) and space.forward(q) is not None
        assert tracks(space, trajectory) <= MM
        # the choice beats chance by its own index, and the entry says so
        assert info["index"] > info["index_of_random_sets_median"], (info["index"], info["index_of_random_sets_median"])
        assert [band.label for band in bands(trajectory) if ": station " in band.label] == \
            [f"calibration poses: station {k}" for k in range(1, 11)]


def test_kinematic_calibration_pose_set_recovers_hidden_offsets_better_than_random_poses():
    """The purpose, against a hidden truth: joints with zero offsets and a tool off the flange, a tracker with noise. A fit from the
    chosen poses predicts held-out tool points better, on average, than a fit from random sets of the same size."""
    for build, start, spread in ((fixtures.ur5e, None, 0.01), (fixtures.fivebar, fixtures.FIVEBAR_START, 0.002)):
        space = build()
        stations = stations_of(kinematic_calibration_pose_set(space, poses=12, candidates=400, start=start, seed=0))
        rng = np.random.default_rng(5)
        room = space.room(space.start_at(start))

        def draw(count: int) -> list[np.ndarray]:
            out = []
            while len(out) < count:
                q = rng.uniform(room.low, room.high)
                if space.kinematics.forward(q) is not None:
                    out.append(q)
            return out

        validation = draw(150)
        chosen, random = [], []
        for trial in range(20):
            offsets, tool = rng.normal(0.0, spread, space.dof), rng.normal(0.0, 0.004, 3)
            try:
                noise = lambda: np.random.default_rng(100 + trial)  # noqa: E731
                chosen_error = fixtures.validation_error(space, fixtures.fit_offsets(space, stations, offsets, tool, 2e-4, noise()),
                                                         offsets, tool, validation)
                rivals = [fixtures.validation_error(space, fixtures.fit_offsets(space, draw(12), offsets, tool, 2e-4, noise()), offsets,
                                                    tool, validation) for _ in range(8)]
            except ValueError:  # an offset configuration with no tool pose on the five-bar: a trial without a measurement
                continue
            chosen.append(chosen_error)
            random.append(float(np.median(rivals)))
        assert len(chosen) >= 8, f"too few trials survived: {len(chosen)} on {space.source}"
        assert np.mean(chosen) < np.mean(random), f"chosen {1e3 * np.mean(chosen):.3f} mm vs random {1e3 * np.mean(random):.3f} mm"


def test_kinematic_calibration_pose_set_seed_and_refusals():
    space = fixtures.ur5e()
    a = stations_of(kinematic_calibration_pose_set(space, poses=6, candidates=120, dwell_s=0.2, seed=1))
    b = stations_of(kinematic_calibration_pose_set(space, poses=6, candidates=120, dwell_s=0.2, seed=1))
    c = stations_of(kinematic_calibration_pose_set(space, poses=6, candidates=120, dwell_s=0.2, seed=2))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True)) and not np.array_equal(a[0], c[0])
    refused(lambda: kinematic_calibration_pose_set(space, seed=1.5), ValueError, contains="seed must be an int")
    refused(lambda: kinematic_calibration_pose_set(space, criterion="vibes"), ValueError, contains="criterion")
    refused(lambda: kinematic_calibration_pose_set(space, poses=50, candidates=10), ValueError, contains="poses")
    refused(lambda: kinematic_calibration_pose_set(JointSpace.from_limits(robots.FIVEBAR_LIMITS)), MissingRung)
    # a collision scene is honoured: no station collides
    scene = space.with_collision(robots.UR5eScene())
    kept = kinematic_calibration_pose_set(scene, poses=6, candidates=120, dwell_s=0.2)
    check_kind(kept, scene)
    assert not any(scene.in_collision(q) for q in stations_of(kept))
    # other criteria run too
    for criterion in ("noise_amplification", "min_singular_value", "condition_number"):
        check_kind(kinematic_calibration_pose_set(space, poses=6, candidates=100, dwell_s=0.2, criterion=criterion), space)


def test_hand_eye_station_set_ur5e():
    space = fixtures.ur5e()
    trajectory = hand_eye_station_set(space, stations=10, check_set_stations=5, dwell_s=0.4)
    check_kind(trajectory, space)
    assert tracks(space, trajectory) <= MM
    info = entry(trajectory)
    stations = stations_of(trajectory)
    assert len(stations) == 15 and info["check_set"] == list(range(11, 16))
    poses = [space.forward(q) for q in stations]
    focus = np.array(info["focus_m"])
    for pose in poses:  # every station aims the tool's +z at the focus point (independent of the kind's construction)
        to_focus = focus - pose[:3, 3]
        aim = math.acos(float(np.clip(np.dot(pose[:3, 2], to_focus / np.linalg.norm(to_focus)), -1.0, 1.0)))
        assert aim <= math.radians(0.2), f"a station misses the focus by {math.degrees(aim):.2f}°"
    turns = [rotation_angle(a[:3, :3], b[:3, :3]) for a, b in zip(poses[:-1], poses[1:], strict=True)]
    assert min(turns) >= 0.52 - 1e-3, f"consecutive stations differ by only {math.degrees(min(turns)):.1f}°"
    assert info["rotation_axis_spread"] > 0.2, "the rotation axes are nearly parallel: AX = XB would be ill-posed"
    assert len({tuple(np.round(p[:3, 3], 4)) for p in poses}) == 15, "two stations are the same pose"
    # the held-out check set is distinct from the fit set in rotation too
    assert np.linalg.norm(np.array([p[:3, 3] for p in poses[:10]]).mean(axis=0) - focus) < 0.5


def test_hand_eye_station_set_refusals():
    space = fixtures.ur5e()
    refused(lambda: hand_eye_station_set(space, stations=6, check_set_stations=2, min_rotation_between_rad=3.0, candidates=200),
            ValueError, contains="found")
    refused(lambda: hand_eye_station_set(space, stations=2), ValueError, contains="at least 3")
    refused(lambda: hand_eye_station_set(space, stations=4, check_set_stations=2, standoff_m=5.0, candidates=50), ValueError, contains="found 0")
    scene = space.with_collision(robots.UR5eScene())
    kept = hand_eye_station_set(scene, stations=5, check_set_stations=2, dwell_s=0.3)
    check_kind(kept, scene)
    assert hand_eye_station_set(space, stations=4, check_set_stations=0, dwell_s=0.2, seed=3).t[-1] > 0


def test_camera_board_views_ur5e():
    space = fixtures.ur5e()
    views, cells, tilt_limit, half_field = 20, 9, 0.785, 0.35
    trajectory = camera_board_views(space, views=views, dwell_s=0.3)
    check_kind(trajectory, space)
    assert tracks(space, trajectory) <= MM
    info = entry(trajectory)
    stations = stations_of(trajectory)
    assert len(stations) == views
    focus = np.array(info["focus_m"])
    start_pose = space.forward(space.home)
    normal = -start_pose[:3, 2]  # the board faces the tool at its start
    seen_cells = set()
    worst_tilt = 0.0
    for q in stations:
        pose = space.forward(q)
        direction = pose[:3, :3].T @ (focus - pose[:3, 3])  # the board's centre in the camera frame
        direction = direction / np.linalg.norm(direction)
        offset = np.array([direction[0] / direction[2], direction[1] / direction[2]])  # tan of the angle across the image
        assert np.all(np.abs(np.arctan(offset)) <= half_field + 0.01), "the board's centre leaves the field of view"
        seen_cells.add((int(np.round(offset[0] / math.tan(half_field))), int(np.round(offset[1] / math.tan(half_field)))))
        to_tool = (pose[:3, 3] - focus) / np.linalg.norm(pose[:3, 3] - focus)
        worst_tilt = max(worst_tilt, math.acos(float(np.clip(np.dot(to_tool, normal), -1.0, 1.0))))
    assert worst_tilt <= tilt_limit + 1e-6, f"a view tilts {math.degrees(worst_tilt):.1f}° past the limit"
    assert worst_tilt >= 0.5 * tilt_limit, "the views barely tilt: Zhang needs the board seen at an angle"
    assert len(seen_cells) >= 7, f"the board's centre visits only {len(seen_cells)} of {cells} image cells: {sorted(seen_cells)}"
    depths = {round(float(np.linalg.norm(space.forward(q)[:3, 3] - focus)), 2) for q in stations}
    assert len(depths) >= 3, f"the board is seen at only {len(depths)} distances"


def test_camera_board_views_refusals():
    space = fixtures.ur5e()
    refused(lambda: camera_board_views(space, views=5, focus_m=(3.0, 3.0, 3.0), standoff_m=0.3), Unreachable, contains="usable")
    refused(lambda: camera_board_views(space, image_cells=8), ValueError, contains="square")
    scene = space.with_collision(robots.UR5eScene())
    kept = camera_board_views(scene, views=6, dwell_s=0.3)
    check_kind(kept, scene)


# ====================================================================== recover, and the cycles


def test_return_to_ready_on_the_five_bar():
    straight = fixtures.fivebar(named_home=True)
    start = np.radians([128.0, -126.0])
    trajectory = return_to_ready(straight, start=start)
    check_kind(trajectory, straight)
    assert np.allclose(trajectory.positions[-1], straight.home, atol=1e-9)
    assert [band.label for band in bands(trajectory)][-1] == "return to ready: settle at READY"
    assert "the straight joint-space line" in entry(trajectory)["notes"][0]
    # the walk is slow: the largest joint's speed peaks at the asked rad/s (a trapezoid plateau), and every waypoint is clear
    assert abs(np.abs(trajectory.velocities).max() - 0.1) <= 0.003, np.abs(trajectory.velocities).max()
    assert all(straight.kinematics.forward(q) is not None for q in trajectory.positions[::10]), "a sample has no tool pose"
    # a start that is a fold rest (no tool pose) is walked home through the reach map's grid, never straight over the fold
    mapped = fixtures.fivebar(reach_map=True, named_home=True)
    rest = np.radians([60.0, -100.0])
    assert mapped.kinematics.forward(rest) is None
    routed = return_to_ready(mapped, start=rest)
    check_kind(routed, mapped)
    assert "reach map's grid" in entry(routed)["notes"][0] and np.allclose(routed.positions[-1], mapped.home, atol=1e-9)
    # WHY after the first walk: leaving a rest that has no tool pose crosses its fold by necessity (the kind says so in its notes); the rest is clear
    first_walk = next(band for band in bands(routed) if band.label.endswith("walk 1"))
    away = routed.positions[routed.t > first_walk.end_s]
    assert len(away) > 100 and all(mapped.kinematics.forward(q) is not None for q in away[::25]), "walks over a fold"
    # without a reach map the same rest has no clear way, and says where the line is blocked
    refused(lambda: return_to_ready(straight, start=rest), ValueError, contains="no clear way to READY")
    # already home: just the settle
    still = return_to_ready(straight, start=straight.home)
    check_kind(still, straight)
    assert np.ptp(still.positions, axis=0).max() == 0.0


def test_return_to_ready_on_the_ur5e_and_with_a_scene():
    space = fixtures.ur5e(named_home=True)
    start = space.home + np.radians([25.0, -20.0, 15.0, 10.0, -12.0, 30.0])
    trajectory = return_to_ready(space, start=start, speed_rad_s=0.3)
    check_kind(trajectory, space)
    assert np.allclose(trajectory.positions[-1], space.home, atol=1e-9)
    # the straight line is blocked by a box placed on its middle waypoint: routed round it, or refused, never driven through
    middle_pose = space.forward(0.5 * (start + space.home))[:3, 3]
    blocked = space.with_collision(robots.UR5eScene(box_min=tuple(middle_pose - 0.03), box_max=tuple(middle_pose + 0.03)))
    try:
        routed = return_to_ready(blocked, start=start, speed_rad_s=0.3)
    except ValueError as error:
        assert "no clear way to READY" in str(error), str(error)
    else:
        check_kind(routed, blocked)
        assert "straight joint-space line" not in entry(routed)["notes"][0]
    refused(lambda: return_to_ready(fixtures.ur5e()), MissingRung, contains="home")  # no named home: refused naming the fix


def test_adept_cycle_on_both_robots():
    cases = ((fixtures.ur5e(), None, dict(across=(0.0, 1.0, 0.0), across_m=0.2), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
             (fixtures.fivebar(), fixtures.FIVEBAR_START, dict(up=fixtures.FIVEBAR_UP, across=(1.0, 0.0, 0.0), across_m=0.08), np.array([1.0, 0.0, 0.0]),
              np.array([0.0, -1.0, 0.0])))
    for space, start, kwargs, across, up in cases:
        trajectory = adept_cycle(space, start=start, **kwargs)
        check_kind(trajectory, space)
        assert labels(trajectory) == [f"cycle 1: {name}" for name in ("up", "across", "down", "up", "back across", "down")]
        origin = space.forward(space.start_at(start))[:3, 3]
        up_m, across_m = 0.025, kwargs["across_m"]
        corners = [origin, origin + up_m * up, origin + up_m * up + across_m * across, origin + across_m * across,
                   origin + up_m * up + across_m * across, origin + up_m * up, origin]
        pieces = split(trajectory)
        for index, piece in enumerate(pieces):  # every line is straight (independent oracle) and arrives within the benchmark's 0.25 mm
            tool = space.tool_points(piece)
            assert fixtures.polyline_error(tool, [corners[index], corners[index + 1]]) <= MM
            assert np.linalg.norm(tool[-1] - corners[index + 1]) <= 0.00025, f"line {index + 1} arrives off its point"
        assert tracks(space, trajectory) <= MM
        # as fast as the limits allow: some joint reaches its velocity or acceleration limit on the longest line
        longest = pieces[1]
        limits = space.limits
        assert max(np.abs(longest.velocities).max(axis=0).__truediv__(limits.max_velocity).max(),
                   np.abs(longest.accelerations).max(axis=0).__truediv__(limits.max_acceleration).max()) >= 0.9
        slower = adept_cycle(space, start=start, time_law=__import__("robot_trajectories").TimeLaw.minimum_jerk(peak_speed_fraction=0.3),
                             **kwargs)
        assert slower.t[-1] > trajectory.t[-1], "a slower law is not slower"
    twice = adept_cycle(fixtures.ur5e(), cycles=2, across=(0.0, 1.0, 0.0), across_m=0.2)
    assert len(labels(twice)) == 12


def test_adept_cycle_refusals():
    space = fixtures.ur5e()
    refused(lambda: adept_cycle(space, across_m=3.0), Unreachable, contains="unreachable")
    refused(lambda: adept_cycle(space, tolerance_m=0.001), ValueError, contains="arrival_tolerance_m")
    refused(lambda: adept_cycle(space, up=(0.0, 0.0, 0.0)), ValueError, contains="up")
    flat = fixtures.fivebar()
    refused(lambda: adept_cycle(flat, across_m=1.0, start=fixtures.FIVEBAR_START, up=fixtures.FIVEBAR_UP), Unreachable)


def test_touch_down_contact_on_both_robots():
    for space, start, up, down in ((fixtures.ur5e(), None, (0.0, 0.0, 1.0), np.array([0.0, 0.0, -1.0])),
                                   (fixtures.fivebar(), fixtures.FIVEBAR_START, fixtures.FIVEBAR_UP, np.array([0.0, 1.0, 0.0]))):
        trajectory = touch_down_contact(space, up=up, start=start)
        check_kind(trajectory, space)
        assert tracks(space, trajectory) <= MM
        assert [band.label for band in bands(trajectory)] == [f"touch down: {name}" for name in
                                                                ("speed up", "touch stretch", "press below the surface", "dwell in contact", "lift")]
        origin = space.forward(space.start_at(start))[:3, 3]
        depth = (space.tool_points(trajectory) - origin) @ down  # how far below the start the tool is, along "down"
        assert abs(depth.max() - 0.05) <= MM, f"the lowest point is {1000 * depth.max():.2f} mm down, wanted 40 + 10"
        assert abs(depth[-1] - (0.05 - 0.04)) <= MM, "the lift does not end 40 mm above the lowest point"
        # the touch stretch is at the touch speed (by forward kinematics, apart from the commanded channel) and the surface (40 mm) is crossed at it
        stretch = next(band for band in bands(trajectory) if band.label.endswith("touch stretch"))
        speeds = fixtures.tool_speeds(space, trajectory, band_rows(trajectory, stretch))
        assert np.all(np.abs(speeds - 0.10) <= 0.002), f"touch speed {speeds.min():.4f}..{speeds.max():.4f} m/s"
        crossing = int(np.argmax(depth >= 0.04))
        crossing_speed = float(np.linalg.norm(space.tool_points(trajectory)[crossing + 1] - space.tool_points(trajectory)[crossing]) /
                               (trajectory.t[crossing + 1] - trajectory.t[crossing]))
        assert abs(crossing_speed - 0.10) <= 0.003, crossing_speed
        dwell = next(band for band in bands(trajectory) if band.label.endswith("dwell in contact"))
        held = trajectory.velocities[band_rows(trajectory, dwell)]
        assert np.abs(held).max() <= 1e-9 and abs(dwell.end_s - dwell.start_s - 0.8) <= 0.01, "not at rest for the 0.8 s dwell"
        # touch speeds of the ladder (0.05 / 0.10 / 0.20 m/s) all generate; a slower touch takes longer
        times = [touch_down_contact(space, up=up, touch_speed_m_s=speed, start=start).t[-1] for speed in (0.05, 0.10)]
        assert times[0] > times[1], times


def test_touch_down_contact_refusals():
    space = fixtures.ur5e()
    refused(lambda: touch_down_contact(space, approach_height_m=3.0, touch_stretch_m=1.0), Unreachable, contains="unreachable")
    flat = fixtures.fivebar()
    message = refused(lambda: touch_down_contact(flat, up=fixtures.FIVEBAR_UP, touch_speed_m_s=40.0, start=fixtures.FIVEBAR_START), ValueError)
    assert "limit" in message and "touch_speed_m_s" in message, message
    refused(lambda: touch_down_contact(space, touch_speed_m_s=0.2), ValueError, contains="limit")  # 0.20 m/s needs 1.28x wrist acceleration
    refused(lambda: touch_down_contact(space, touch_stretch_m=0.05), ValueError, contains="touch_stretch_m")
    refused(lambda: touch_down_contact(space, lift_m=0.5), ValueError, contains="lift_m")


def test_every_lane_d_output_rebuilds_from_its_recipe_in_its_space_only():
    space = fixtures.ur5e()
    other = fixtures.fivebar()
    for trajectory in (touch_down_contact(space), circular_test(space, directions="ccw"),
                       hand_eye_station_set(space, stations=4, check_set_stations=0, dwell_s=0.2)):
        check_kind(trajectory, space, other_space=other)
    assert json.loads(adept_cycle(space, across=(0.0, 1.0, 0.0), across_m=0.2).header.meta["recipe"])["calls"][0]["kind"] == "adept_cycle"


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_lane_d (standard tests, calibration, task cycles)"))
