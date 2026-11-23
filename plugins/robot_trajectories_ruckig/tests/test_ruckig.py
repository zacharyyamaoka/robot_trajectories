"""The ``ruckig`` waypoint trajectory generator from Ruckig Community, held to robot_trajectories' WaypointTrajectoryGeneratorBackend
contract on the pendulum, the five-bar and the UR5e, through its real entry point. A plain script (the arm may be powered: never pytest):

    /home/bam/robot_trajectories/plugins/robot_trajectories_ruckig/.venv/bin/python /home/bam/robot_trajectories/plugins/robot_trajectories_ruckig/tests/test_ruckig.py

The jerk limits below are stand-ins (the fixtures state none): 200 rad/s^3 on the pendulum, 300 on the five-bar, 100 on the
UR5e. Every check re-derives its claim from the samples (velocity, acceleration, and jerk as the change of acceleration
between samples), never from the backend's own words.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

CORE = pathlib.Path(__file__).resolve().parents[3]  # /home/bam/robot_trajectories
sys.path.insert(0, str(CORE / "tests"))

import numpy as np  # noqa: E402

import _runner  # noqa: E402
import robot_trajectories as rt  # noqa: E402
import scenario  # noqa: E402
from fixtures import robots  # noqa: E402
from kind_contract import check_kind  # noqa: E402
from robot_trajectories import BoundaryState, JointSpace, MissingRung, WaypointTrajectoryGenerator  # noqa: E402

PROVIDER = "robot-trajectories-ruckig"
PRO = "robot-trajectories-ruckig-pro"
FIVEBAR_START = np.radians([128.0, -126.0])


def pendulum() -> JointSpace:
    return JointSpace.from_limits(robots.PENDULUM_LIMITS.replace(max_jerk=np.array([200.0])))


def fivebar() -> JointSpace:
    return (JointSpace.from_limits(robots.FIVEBAR_LIMITS.replace(max_jerk=np.array([300.0, 300.0])), home=np.radians([100.0, -76.0]))
            .with_reach_map(robots.fivebar_reach_map()))


def ur5e() -> JointSpace:
    return JointSpace.from_limits(scenario.ur5e_space().limits.replace(max_jerk=np.full(6, 100.0)))


def legs(space: JointSpace) -> list[tuple[np.ndarray, np.ndarray]]:
    """Three legs from home to fractions of the room (every joint moving, by different amounts)."""
    room, home = space.room(space.home), space.home
    half = np.minimum(home - room.low, room.high - home)
    weights = np.linspace(1.0, 0.4, space.dof)
    return [(home, home + fraction * weights * half) for fraction in (0.8, -0.5, 0.15)]


def within(arrays, limits, rate_hz: float) -> None:
    """Inside the velocity, acceleration and jerk limits it was handed, on the samples; jerk as the change of acceleration."""
    qd, qdd = np.asarray(arrays.qd), np.asarray(arrays.qdd)
    assert np.max(np.abs(qd) / limits.max_velocity) <= 1 + 1e-9, f"velocity {np.max(np.abs(qd) / limits.max_velocity):.6f}x"
    assert np.max(np.abs(qdd) / limits.max_acceleration) <= 1 + 1e-9, f"acceleration {np.max(np.abs(qdd) / limits.max_acceleration):.6f}x"
    jerk = np.abs(np.diff(qdd, axis=0)) * rate_hz
    assert np.max(jerk / limits.max_jerk) <= 1 + 1e-6, f"jerk {np.max(jerk / limits.max_jerk):.6f}x the limit"


def fresh(script: str, *extra_path: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """``script`` in a fresh interpreter of this venv (the plugin found only through its installed entry point)."""
    paths = [*extra_path, str(CORE / "tests")]
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, cwd=CORE,
                          env={**os.environ, **(env or {}), "PYTHONPATH": os.pathsep.join(paths)})
    return done


# ---------------------------------------------------------------------- found by its entry point


def test_found_through_its_entry_point_in_a_fresh_interpreter():
    done = fresh(
        "from importlib import metadata\n"
        "import robot_trajectories as rt\n"
        "from robot_trajectories import plugins\n"
        "points = [p for p in metadata.entry_points(group='robot_trajectories.waypoint_trajectory_generators') if p.name == 'ruckig']\n"
        "assert [p.value for p in points] == ['robot_trajectories_ruckig:RuckigBackend'], points\n"
        "law = rt.WaypointTrajectoryGenerator('ruckig', peak_speed_fraction=1.0)\n"
        f"assert [e.provider for e in rt.waypoint_trajectory_generators()['ruckig']] == [{PROVIDER!r}], "
        "rt.waypoint_trajectory_generators()['ruckig']\n"
        "assert plugins.errors() == [], plugins.errors()\n"
        "serving = rt.waypoint_trajectory_generator.backend('ruckig')\n"
        "assert type(serving).__module__ == 'robot_trajectories_ruckig' and serving.edition == 'community', serving\n"
        "print('found', type(serving).__name__)\n")
    assert done.returncode == 0 and "found RuckigBackend" in done.stdout, done.stderr[-3000:]


# ---------------------------------------------------------------------- the contract on three robots


def test_rest_to_rest_legs_start_and_end_at_the_path_ends_inside_their_limits():
    law = WaypointTrajectoryGenerator("ruckig", peak_speed_fraction=1.0)
    for space in (pendulum(), fivebar(), ur5e()):
        for q0, q1 in legs(space):
            arrays = law.leg(q0, q1, space.limits, space.rate_hz)
            assert np.allclose(arrays.t, np.arange(len(arrays.t)) / space.rate_hz, atol=1e-12), "not on the rate's grid from t = 0"
            assert np.array_equal(arrays.q[0], q0) and np.array_equal(arrays.q[-1], q1), "does not start and end at the path's ends"
            assert not np.any(arrays.qd[0]) and not np.any(arrays.qd[-1]), "not at rest at both ends"
            within(arrays, space.limits, space.rate_hz)


def test_agrees_with_minimum_jerk_on_rest_to_rest_legs():
    # WHY this jerk limit: a min-jerk leg of duration T peaks at a jerk of 60 |d| / T^3 per joint. With exactly that limit the
    # min-jerk leg is one of the motions Ruckig's time-optimal search ranges over, so Ruckig's leg can be no longer. The ruler
    # can fail: at a tenth of that jerk, Ruckig's leg must be longer than min-jerk's.
    for space in (pendulum(), fivebar(), ur5e()):
        for q0, q1 in legs(space):
            reference = WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=1.0).leg(q0, q1, space.limits, space.rate_hz)
            duration = reference.t[-1]
            distance = np.abs(q1 - q0)
            jerk = np.where(distance > 0, 60.0 * distance / duration**3, np.max(60.0 * distance / duration**3))
            limits = space.limits.replace(max_jerk=jerk)
            fast = WaypointTrajectoryGenerator("ruckig", peak_speed_fraction=1.0).leg(q0, q1, limits, space.rate_hz)
            assert len(fast.t) <= len(reference.t), f"ruckig {fast.t[-1]:.3f} s is longer than minimum-jerk {duration:.3f} s"
            within(fast, limits, space.rate_hz)
            slow = WaypointTrajectoryGenerator("ruckig", peak_speed_fraction=1.0).leg(q0, q1, limits.replace(max_jerk=jerk / 10),
                                                                               space.rate_hz)
            assert slow.t[-1] > duration, "a tenth of the jerk did not lengthen the leg: the jerk limit is not reaching Ruckig"


def test_a_leg_ends_moving_in_exactly_the_state_it_was_given():
    law = WaypointTrajectoryGenerator("ruckig", peak_speed_fraction=1.0)
    for space, v0, v1, a1 in ((pendulum(), [0.3], [0.8], [-1.0]), (fivebar(), [0.0, 0.2], [0.6, -0.4], [0.0, 0.0])):
        q0, q1 = legs(space)[1]
        start, end = BoundaryState(np.array(v0), np.zeros(space.dof)), BoundaryState(np.array(v1), np.array(a1))
        arrays = law.leg(q0, q1, space.limits, space.rate_hz, start_state=start, end_state=end)
        assert np.array_equal(arrays.q[-1], q1) and np.array_equal(arrays.qd[-1], end.velocity), "does not end in the end state"
        assert np.array_equal(arrays.qdd[-1], end.acceleration), "does not end at the end acceleration"
        assert np.array_equal(arrays.qd[0], start.velocity), "does not start in the start state"
        assert np.allclose(arrays.t, np.arange(len(arrays.t)) / space.rate_hz, atol=1e-12)
        within(arrays, space.limits, space.rate_hz)
    try:
        law.leg(np.zeros(1), np.full(1, 0.2), pendulum().limits, 500.0, end_state=BoundaryState(np.full(1, 5.0), np.zeros(1)))
    except ValueError as error:
        assert "exceeds its maximum velocity" in str(error), error
    else:
        raise AssertionError("an end speed over the limit was not refused")


def test_jerk_limited_scurve_kind_on_three_robots_keeps_phase_synchronised_legs_straight():
    for space, joint in ((pendulum(), "can11"), (fivebar(), None), (ur5e(), None)):
        tape = rt.jerk_limited_scurve(space, joint)
        check_kind(tape, space)
        within(rt.kernels.Arrays(tape.t, tape.positions, tape.velocities, tape.accelerations), space.limits, space.rate_hz)
        call = json.loads(rt.recipe(tape))["calls"][0]
        assert call["kind"] == "jerk_limited_scurve" and call["synchronization"] == "phase", call
    # The catalog row's claim ("phase synchronisation keeps a multi-joint move straight"), and the ruler failing on "time".
    for space in (fivebar(), ur5e()):
        for sync, straight in (("phase", True), ("time", False)):
            tape = rt.jerk_limited_scurve(space, targets_fraction=(0.6,), back_to_start=False, synchronization=sync, dwell_s=0.0)
            q0, q1 = tape.positions[0], tape.positions[-1]
            direction = (q1 - q0) / np.linalg.norm(q1 - q0)
            offsets = tape.positions - q0
            off = float(np.max(np.linalg.norm(offsets - np.outer(offsets @ direction, direction), axis=1)))
            assert (off <= 1e-6) == straight, f"{sync}: {off:.2e} rad off the straight joint-space line"


# ---------------------------------------------------------------------- refusals, by name


def test_refuses_intermediate_waypoints_before_ruckig_is_ever_called():
    import robot_trajectories_ruckig as plugin

    space = fivebar()
    path = np.stack([space.home, space.home + 0.1, space.home + (0.2, 0.0)])
    law = WaypointTrajectoryGenerator("ruckig", peak_speed_fraction=1.0)
    original = plugin.solve_leg

    def tripwire(*args, **kwargs):
        raise AssertionError("Ruckig was called with a waypoint path")

    plugin.solve_leg = tripwire
    try:
        for call in (lambda: law.along(path, space.limits, space.rate_hz),
                     lambda: rt.waypoint_trajectory_generator.backend("ruckig").time_parameterize(path, space.limits, space.rate_hz,
                                                                                          law)):
            try:
                call()
            except ValueError as error:
                assert PRO in str(error) and "cloud" in str(error), error
            else:
                raise AssertionError("a three-point path was timed by Ruckig Community")
        refusal = rt.waypoint_trajectory_generator.backend("ruckig").can_service(path, space.limits, law, None, None)
        assert PRO in refusal and "cloud" in refusal, refusal
        # A Cartesian line hands the waypoint trajectory generator its dense IK path: refused the same way, never sent.
        kinematic = JointSpace.from_limits(space.limits, home=space.home).with_kinematics(robots.FiveBarKinematics())
        tool = kinematic.forward(FIVEBAR_START)[:3, 3]
        try:
            rt.cartesian_path(kinematic, [tool + (0.03, 0.0, 0.0)], waypoint_generator=law, start=FIVEBAR_START)
        except ValueError as error:
            assert PRO in str(error), error
        else:
            raise AssertionError("a dense Cartesian path was timed by Ruckig Community")
    finally:
        plugin.solve_leg = original


def test_refuses_without_a_jerk_limit_and_names_how_to_add_one():
    bare = JointSpace.from_limits(robots.PENDULUM_LIMITS)
    law = WaypointTrajectoryGenerator("ruckig", peak_speed_fraction=1.0)
    try:
        law.leg(np.zeros(1), np.full(1, 0.2), bare.limits, bare.rate_hz)
    except ValueError as error:
        assert "max_jerk" in str(error), error
    else:
        raise AssertionError("a leg without a jerk limit was timed")
    try:
        rt.jerk_limited_scurve(bare, "can11")
    except MissingRung as error:
        assert "jerk" in str(error), error
    else:
        raise AssertionError("the kind ran on a space without a jerk limit")
    stated = WaypointTrajectoryGenerator("ruckig", peak_speed_fraction=1.0, options={"max_jerk": 200.0})
    within(stated.leg(np.zeros(1), np.full(1, 0.2), bare.limits, bare.rate_hz), bare.limits.replace(max_jerk=np.array([200.0])), 500.0)
    try:
        WaypointTrajectoryGenerator("ruckig", peak_speed_fraction=1.0, options={"jerk_fractoin": 0.5}).leg(np.zeros(1), np.full(1, 0.2),
                                                                                                    pendulum().limits, 500.0)
    except ValueError as error:
        assert "jerk_fractoin" in str(error), error
    else:
        raise AssertionError("an unknown option was ignored")


def test_checks_the_position_limits_community_ignores():
    space = pendulum()
    # From rest at 0.25 rad, arrive at 0.30 rad moving back at 1.5 rad/s: it must swing past 0.30 first, beyond the 0.33 limit.
    try:
        WaypointTrajectoryGenerator("ruckig", peak_speed_fraction=1.0).leg(np.full(1, 0.25), np.full(1, 0.30), space.limits,
                                                                    space.rate_hz,
                                                                    end_state=BoundaryState(np.full(1, -1.5), np.zeros(1)))
    except ValueError as error:
        assert "position limits" in str(error) and PRO in str(error), error
    else:
        raise AssertionError("a leg past the position limit was not refused")


def test_refuses_to_load_beside_the_ruckig_pro_distribution():
    with tempfile.TemporaryDirectory() as site:
        info = pathlib.Path(site) / "ruckig_pro-0.19.4.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: ruckig-pro\nVersion: 0.19.4\n")
        done = fresh("import warnings\nwarnings.simplefilter('ignore')\n"
                     "import robot_trajectories as rt\nfrom robot_trajectories import plugins\nplugins.load()\n"
                     "assert 'ruckig' not in rt.waypoint_trajectory_generators(), rt.waypoint_trajectory_generators()\n"
                     "assert any('exactly one' in e and 'ruckig-pro' in e for e in plugins.errors()), plugins.errors()\n"
                     "print('refused')\n", site)
    assert done.returncode == 0 and "refused" in done.stdout, done.stderr[-3000:]


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_ruckig (plugin robot-trajectories-ruckig: Ruckig Community)"))
