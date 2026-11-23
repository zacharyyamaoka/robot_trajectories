"""The ``toppra`` waypoint trajectory generator (TOPP-RA), held to robot_trajectories' WaypointTrajectoryGeneratorBackend contract on the
pendulum, the five-bar and the UR5e, through its real entry point. A plain script (the arm may be powered: never pytest):

    /home/bam/robot_trajectories/plugins/robot_trajectories_toppra/.venv/bin/python /home/bam/robot_trajectories/plugins/robot_trajectories_toppra/tests/test_toppra.py

Independent oracles, not the backend's words: a straight leg's time-optimal duration in closed form (bang-coast-bang under the
tightest joint's scaled limits); the dense path re-read at the backend's ``path_progress`` by linear interpolation; on a
Cartesian line, forward kinematics of the output against the line itself; and in the torque-limited mode, the dynamics
model's own torque along the output (``space.torques``) against ``max_effort``.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys

CORE = pathlib.Path(__file__).resolve().parents[3]  # /home/bam/robot_trajectories
sys.path.insert(0, str(CORE / "tests"))

import numpy as np  # noqa: E402

import _runner  # noqa: E402
import robot_trajectories as rt  # noqa: E402
import scenario  # noqa: E402
from fixtures import robots  # noqa: E402
from fixtures import lane_b  # noqa: E402
from kind_contract import check_kind  # noqa: E402
from robot_trajectories import BoundaryState, JointSpace, MissingRung, WaypointTrajectoryGenerator  # noqa: E402
from robot_trajectories.segments import TOOL_TARGET  # noqa: E402

PROVIDER = "robot-trajectories-toppra"
FIVEBAR_START = np.radians([128.0, -126.0])
MM = 0.001


def pendulum() -> JointSpace:
    return scenario.pendulum_space()


def fivebar() -> JointSpace:
    return scenario.fivebar_space()


def ur5e() -> JointSpace:
    return scenario.ur5e_space()


def dense_path(space: JointSpace, count: int = 400) -> np.ndarray:
    """A smooth curve of ``count`` points around home, inside the room (every joint on its own sine, turning back)."""
    room, home = space.room(space.home), space.home
    half = np.minimum(home - room.low, room.high - home)
    s = np.linspace(0.0, 1.0, count)[:, None]
    phase = np.arange(space.dof)[None, :]
    return home + 0.6 * half * np.sin(2 * np.pi * s * (1 + 0.5 * phase)) * np.where(phase % 2, -1.0, 1.0)


def within(arrays, limits) -> None:
    qd, qdd = np.asarray(arrays.qd), np.asarray(arrays.qdd)
    assert np.max(np.abs(qd) / limits.max_velocity) <= 1 + 1e-6, f"velocity {np.max(np.abs(qd) / limits.max_velocity):.6f}x"
    assert np.max(np.abs(qdd) / limits.max_acceleration) <= 1 + 1e-6, f"acceleration {np.max(np.abs(qdd) / limits.max_acceleration):.6f}x"


def line_optimum(q0: np.ndarray, q1: np.ndarray, limits) -> float:
    """The time-optimal rest-to-rest duration along the straight joint-space line q0 -> q1, in closed form: with q = q0 + d u,
    u in [0, 1], the line's own limits are V = min |v_k / d_k| and A = min |a_k / d_k|; then bang-coast-bang (or bang-bang)."""
    distance = np.abs(q1 - q0)
    moving = distance > 0
    top = float(np.min(limits.max_velocity[moving] / distance[moving]))
    push = float(np.min(limits.max_acceleration[moving] / distance[moving]))
    return 2.0 * math.sqrt(1.0 / push) if top * top / push >= 1.0 else 1.0 / top + top / push


def fresh(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, cwd=CORE,
                          env={**os.environ, "PYTHONPATH": str(CORE / "tests")})


# ---------------------------------------------------------------------- found by its entry point


def test_found_through_its_entry_point_in_a_fresh_interpreter():
    done = fresh(
        "from importlib import metadata\n"
        "import robot_trajectories as rt\n"
        "from robot_trajectories import plugins\n"
        "points = [p for p in metadata.entry_points(group='robot_trajectories.waypoint_trajectory_generators') if p.name == 'toppra']\n"
        "assert [p.value for p in points] == ['robot_trajectories_toppra:ToppraBackend'], points\n"
        "rt.WaypointTrajectoryGenerator('toppra', peak_speed_fraction=1.0)\n"
        f"assert [e.provider for e in rt.waypoint_trajectory_generators()['toppra']] == [{PROVIDER!r}], "
        "rt.waypoint_trajectory_generators()['toppra']\n"
        "assert plugins.errors() == [], plugins.errors()\n"
        "print('found', type(rt.waypoint_trajectory_generator.backend('toppra')).__name__)\n")
    assert done.returncode == 0 and "found ToppraBackend" in done.stdout, done.stderr[-3000:]


# ---------------------------------------------------------------------- the contract on three robots


def test_straight_legs_match_the_closed_form_optimum_and_beat_minimum_jerk():
    law = WaypointTrajectoryGenerator("toppra", peak_speed_fraction=1.0)
    for space in (pendulum(), fivebar(), ur5e()):
        room, home = space.room(space.home), space.home
        half = np.minimum(home - room.low, room.high - home)
        for fraction in (0.9, -0.4, 0.05):
            q0, q1 = home, home + fraction * np.linspace(1.0, 0.3, space.dof) * half
            arrays = law.leg(q0, q1, space.limits, space.rate_hz)
            assert np.allclose(arrays.t, np.arange(len(arrays.t)) / space.rate_hz, atol=1e-12), "not on the rate's grid from t = 0"
            assert np.array_equal(arrays.q[0], q0) and np.array_equal(arrays.q[-1], q1) and not np.any(arrays.qd[[0, -1]])
            within(arrays, space.limits)
            # WHY the floor: a leg lasts at least the law's min_move_s (0.1 s, ten 100 Hz plan ticks), optimal or not
            best = max(line_optimum(q0, q1, space.limits), law.min_move_s)
            assert best - 1e-9 <= arrays.t[-1] <= best * 1.01 + 1.0 / space.rate_hz, \
                f"{arrays.t[-1]:.4f} s against the closed-form optimum {best:.4f} s"
            reference = WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=1.0).leg(q0, q1, space.limits, space.rate_hz)
            assert len(arrays.t) <= len(reference.t), f"{arrays.t[-1]:.3f} s is longer than minimum-jerk {reference.t[-1]:.3f} s"


def test_dense_paths_are_followed_point_by_point_inside_the_limits():
    law = WaypointTrajectoryGenerator("toppra", peak_speed_fraction=1.0)
    for space in (pendulum(), fivebar(), ur5e()):
        path = dense_path(space)
        timed = law.along(path, space.limits, space.rate_hz)
        arrays = timed.arrays
        assert np.allclose(arrays.t, np.arange(len(arrays.t)) / space.rate_hz, atol=1e-12), "not on the rate's grid from t = 0"
        assert np.array_equal(arrays.q[0], path[0]) and np.array_equal(arrays.q[-1], path[-1]) and not np.any(arrays.qd[[0, -1]])
        within(arrays, space.limits)
        progress = timed.progress
        assert progress[0] == 0.0 and progress[-1] == 1.0 and np.all(np.diff(progress) >= -1e-12), "path_progress not 0 -> 1, monotone"
        # The path re-read at path_progress by linear interpolation (independent of the spline) lands on the samples.
        index = progress * (len(path) - 1)
        expected = np.stack([np.interp(index, np.arange(len(path)), path[:, joint]) for joint in range(space.dof)], axis=1)
        miss = float(np.max(np.abs(expected - arrays.q)))
        # WHY this bound: a smooth curve sits within |q''| h^2 / 8 of its chords, i.e. a quarter of the largest second
        # difference of the points (twice the textbook bound, for the spline's own wiggle)
        bound = float(np.max(np.abs(np.diff(path, 2, axis=0)))) / 4.0 + 1e-9
        assert miss <= bound, f"the samples leave the path read at path_progress by {miss:.2e} rad (bound {bound:.2e})"
        reference = WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=1.0).along(path, space.limits, space.rate_hz).arrays
        assert arrays.t[-1] <= reference.t[-1], f"{arrays.t[-1]:.3f} s is longer than minimum-jerk along it {reference.t[-1]:.3f} s"


def test_a_cartesian_line_through_toppra_tracks_its_commanded_points_within_a_millimetre():
    law = WaypointTrajectoryGenerator("toppra", peak_speed_fraction=1.0)
    five = JointSpace.from_limits(robots.FIVEBAR_LIMITS, home=np.radians([100.0, -76.0])).with_kinematics(robots.FiveBarKinematics())
    arm = ur5e().with_kinematics(robots.UR5eKinematics())
    five_tool = five.forward(FIVEBAR_START)[:3, 3]
    arm_tool = arm.forward(arm.home)[:3, 3]
    for space, start, tool, end in ((five, FIVEBAR_START, five_tool, five_tool + (0.03, 0.03, 0.0)),
                                    (arm, arm.home, arm_tool, arm_tool + (0.10, 0.05, -0.05))):
        line = rt.cartesian_path(space, [end], waypoint_generator=law, start=start)
        check_kind(line, space)
        target = line.channels[TOOL_TARGET]
        rows = np.flatnonzero(np.isfinite(target[:, 0]))
        reached = space.tool_points(line)
        tracking = float(np.max(np.linalg.norm(reached[rows] - target[rows], axis=1)))
        assert tracking <= MM, f"forward kinematics is {tracking * 1000:.3f} mm from the commanded tool point (path_progress)"
        direction = (end - tool) / np.linalg.norm(end - tool)
        offsets = reached - tool
        off = float(np.max(np.linalg.norm(offsets - np.outer(offsets @ direction, direction), axis=1)))
        assert off <= MM, f"the tool leaves the straight line by {off * 1000:.3f} mm"
        slow = rt.cartesian_path(space, [end], start=start,
                                 waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=1.0))
        assert line.t[-1] <= slow.t[-1], f"toppra {line.t[-1]:.3f} s is slower than minimum-jerk {slow.t[-1]:.3f} s on the line"


def test_time_optimal_path_kind_on_three_robots():
    for space in (pendulum(), fivebar(), ur5e()):
        path = dense_path(space, 60)
        tape = rt.time_optimal_path(space, path)
        check_kind(tape, space)
        reference = WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=1.0).along(path, space.limits, space.rate_hz).arrays
        assert tape.t[-1] <= reference.t[-1], f"{tape.t[-1]:.3f} s is longer than minimum-jerk along the same path"
        alone = rt.time_optimal_path(space)  # the contract: every kind is callable as f(space) alone (a default loop path)
        check_kind(alone, space)
        assert np.array_equal(alone.positions[0], space.home) and np.array_equal(alone.positions[-1], space.home)
    space = pendulum()
    two = rt.time_optimal_path(space, [[0.0], [0.3]])
    best = line_optimum(np.zeros(1), np.full(1, 0.3), space.limits)
    assert best - 1e-9 <= two.t[-1] <= best * 1.01 + 1.0 / space.rate_hz, f"{two.t[-1]:.4f} s against the optimum {best:.4f} s"
    try:
        rt.time_optimal_path(space, [[0.0], [0.5]])
    except ValueError as error:
        assert "outside its limits" in str(error), error
    else:
        raise AssertionError("a path point outside the limits was not refused")


# ---------------------------------------------------------------------- refusals, by name


def test_refuses_moving_ends_a_torque_mode_without_its_inputs_and_unknown_options_by_name():
    space = pendulum()
    law = WaypointTrajectoryGenerator("toppra", peak_speed_fraction=1.0)
    try:
        law.leg(np.zeros(1), np.full(1, 0.2), space.limits, space.rate_hz, end_state=BoundaryState(np.full(1, 0.5), np.zeros(1)))
    except ValueError as error:
        assert "rest to rest" in str(error) and "minimum_jerk" in str(error), error
    else:
        raise AssertionError("a moving end was not refused")
    for options, words in (({"torque_limits": True}, "no option ['torque_limits']"), ({"grid": 50}, "no option ['grid']"),
                           ({"acceleration_fraction": 1.5}, "acceleration_fraction"), ({"grid_points": 1}, "grid_points")):
        try:
            WaypointTrajectoryGenerator("toppra", peak_speed_fraction=1.0, options=options).leg(np.zeros(1), np.full(1, 0.2),
                                                                                         space.limits, space.rate_hz)
        except ValueError as error:
            assert words in str(error), (options, error)
        else:
            raise AssertionError(f"{options} was not refused")
    # The torque-limited mode needs the space's dynamics and a max_effort: refused before TOPP-RA runs, naming the rung.
    for bare, need in ((pendulum(), "dynamics_model"),
                       (JointSpace.from_limits(robots.PENDULUM_LIMITS).with_dynamics(lane_b.Can11Dynamics()), "torque_limits")):
        try:
            rt.time_optimal_path(bare, effort_fraction=1.0)
        except MissingRung as error:
            assert error.need == need and "Add it" in str(error), error
        else:
            raise AssertionError(f"the torque-limited mode ran without {need}")
    try:
        rt.time_optimal_path(lane_b.pendulum_space(), effort_fraction=1.5)
    except ValueError as error:
        assert "effort_fraction must lie in (0, 1]" in str(error), error
    else:
        raise AssertionError("effort_fraction 1.5 was not refused")
    try:  # holding still at the path's far end already needs gravity 4.661 sin(0.33) = 1.51 N·m
        rt.time_optimal_path(lane_b.pendulum_space(max_effort_nm=1.0), [[0.0], [0.33]], effort_fraction=1.0)
    except ValueError as error:
        assert "holding still at path point 1" in str(error), error
    else:
        raise AssertionError("a path whose gravity alone breaks the bound was timed")


# ---------------------------------------------------------------------- the torque-limited mode, on the CAN 11 pendulum


def test_torque_limited_on_the_can11_pendulum_is_never_faster_and_stays_within_max_effort():
    """Lane B's CAN 11 stand-in (tests/fixtures/lane_b.py: inertia 0.122 kg m^2, gravity 4.661 N·m, Coulomb 0.833 N·m, viscous
    0.02): its friction is outside TOPP-RA's rigid-body form, so the bound has to hold by the model's own torque, not TOPP-RA's."""
    probe = pendulum()
    paths = {"the default loop": None, "a dense sine": dense_path(probe, 60)}
    slower = 0
    for effort_nm in (lane_b.CAN11_DRIVE_LIMIT_NM, 2.0, 1.6, 1.4):
        space = lane_b.pendulum_space(max_effort_nm=effort_nm)
        for name, path in paths.items():
            kinematic = rt.time_optimal_path(space, path)
            limited = rt.time_optimal_path(space, path, effort_fraction=1.0)
            check_kind(limited, space)
            assert limited.t[-1] >= kinematic.t[-1], f"{name} at {effort_nm} N·m: {limited.t[-1]:.3f} s beats the kinematic {kinematic.t[-1]:.3f} s"
            effort = float(np.max(np.abs(space.torques(limited))))
            assert effort <= effort_nm * (1 + 1e-9), f"{name}: the model needs {effort:.4f} N·m, over max_effort {effort_nm} N·m"
            needed = float(np.max(np.abs(space.torques(kinematic))))
            if needed <= effort_nm:  # the bound does not bind: the kinematic optimum, sample for sample
                assert np.array_equal(limited.positions, kinematic.positions) and np.array_equal(limited.t, kinematic.t), name
            elif limited.t[-1] > kinematic.t[-1]:
                slower += 1
    assert slower >= 3, f"the torque bound slowed only {slower} timings: the test is not exercising the mode"
    # the fraction: within 0.8 x 2.0 N·m
    space = lane_b.pendulum_space(max_effort_nm=2.0)
    tape = rt.time_optimal_path(space, effort_fraction=0.8)
    assert float(np.max(np.abs(space.torques(tape)))) <= 1.6 * (1 + 1e-9)
    assert json.loads(rt.recipe(tape))["calls"][0]["effort_fraction"] == 0.8, "the mode is in the recipe"


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_toppra (plugin robot-trajectories-toppra: TOPP-RA)"))
