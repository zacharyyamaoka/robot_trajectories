"""The signal library's contract (docs/peps/0002-signal-library.md), each promise held by a test that can fail.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_contract.py

The decorator (refusals, guard, recipe), segments (chain / split / repeat / annotate), moving boundaries (starts / ends
"moving"), the time-law backend seam (built-ins, a plugin's contract, ambiguity, the Ruckig Community rule), the injected
slots (dynamics, collision, effort-commanded kinds) and the moving-target contract. The test kinds registered here have names
starting ``contract_`` and live only in this module.
"""

from __future__ import annotations

import json
import math

import numpy as np

import _runner
import scenario
from fixtures import robots
from kind_contract import check_kind
from robot_trajectories import (BoundaryState, CannotCatch, JointSpace, LinearMotion, MissingRung, Piece, TargetWaypoint, TimeLaw,
                                UnsafeTrajectory, annotate, as_moving_target, bands, cartesian_path, catalog, chain, hold, kind,
                                labels, rebuild, recipe, register_time_law, repeat, segment_info, split, step_train, synchronised,
                                time_laws)
from robot_trajectories import kernels as k

PENDULUM = scenario.pendulum_space()


# ---------------------------------------------------------------------- test kinds (contract_*), registered at import


@kind(use_cases=("smoke_test",), family="point_to_point", starts_as="path", dimensions=("1D",))
def contract_too_fast(space, joint=None, *, start=None):
    """A min-jerk move at three times what the limits allow: the guard must refuse it."""
    target = start.copy()
    target[space.joint_index(joint)] += 0.5 * (space.bounds.high - start)[space.joint_index(joint)]
    return k.min_jerk(start, target, space.bounds, space.rate_hz, speed=3.0, label="too fast")


@kind(use_cases=("smoke_test",), needs=("joint_limits", "dynamics_model"), family="hold", starts_as="time_signal", dimensions=("1D",))
def contract_needs_dynamics(space, *, start=None):
    """A hold that declares it needs a dynamics model."""
    return k.hold(start, 0.1, space.rate_hz, label="hold")


@kind(use_cases=("smoke_test",), needs=("joint_limits", "collision_model"), family="hold", starts_as="time_signal", dimensions=("1D",))
def contract_needs_collision(space, *, start=None):
    """A hold that declares it needs a collision checker."""
    return k.hold(start, 0.1, space.rate_hz, label="hold")


@kind(use_cases=("smoke_test",), family="point_to_point", starts_as="path", dimensions=("1D",), ends="moving")
def contract_ramp_out(space, joint=None, *, fraction=0.3, speed_fraction=0.3, label="ramp out", start=None):
    """From rest to a point, arriving moving at ``speed_fraction`` of top speed (the quintic through both states)."""
    index = space.joint_index(joint)
    target = start.copy()
    target[index] += fraction * (space.bounds.high - start)[index]
    velocity = np.zeros(space.dof)
    velocity[index] = speed_fraction * space.bounds.max_velocity[index]
    return TimeLaw.minimum_jerk(peak_speed_fraction=0.6).leg(start, target, space.limits, space.rate_hz, label=label,
                                                             end_state=BoundaryState(velocity, np.zeros(space.dof)))


@kind(use_cases=("smoke_test",), family="point_to_point", starts_as="path", dimensions=("1D",), starts="moving")
def contract_ramp_in(space, joint=None, *, fraction=0.2, label="ramp in", start=None):
    """From a moving state (``start`` is a State), on to a point and to rest."""
    index = space.joint_index(joint)
    target = start.q.copy()
    target[index] += fraction * (space.bounds.high - start.q)[index]
    return TimeLaw.minimum_jerk(peak_speed_fraction=0.6).leg(start.q, target, space.limits, space.rate_hz, label=label,
                                                             start_state=BoundaryState(start.qd, start.qdd))


@kind(use_cases=("smoke_test",), needs=("joint_limits", "torque_limits", "dynamics_model"), family="excitation",
      starts_as="time_signal", dimensions=("1D",), command="effort", ends="anywhere", until="the pulse ends; the joint swings free")
def contract_torque_pulse(space, joint=None, *, effort_n_m=0.1, duration_s=0.4, label="torque pulse", start=None):
    """An effort-commanded kind: a raised-cosine torque pulse, its positions predicted by the space's dynamics (lane B's shape).
    It ends wherever the pulse leaves the joint (``ends="anywhere"``)."""
    t = np.arange(int(round(duration_s * space.rate_hz)) + 1) / space.rate_hz
    effort = np.zeros((len(t), space.dof))
    effort[:, space.joint_index(joint)] = effort_n_m * 0.5 * (1 - np.cos(2 * np.pi * t / t[-1]))
    effort += space.torque(start, np.zeros(space.dof), np.zeros(space.dof))  # hold gravity at the start
    q, qd, qdd = space.predict(start, effort)
    return Piece(t, q, qd, qdd, label, effort=effort)


# ---------------------------------------------------------------------- the decorator


def test_a_kind_meets_the_contract():
    check_kind(step_train(PENDULUM, "can11"), PENDULUM, other_space=JointSpace.from_limits(robots.PENDULUM_LIMITS, max_acceleration=5.0))


def test_the_guard_refuses_an_unsafe_kind():
    try:
        contract_too_fast(PENDULUM, "can11")
    except UnsafeTrajectory as error:
        assert "can11 acceleration reaches" in str(error) and "contract_too_fast" in str(error), str(error)
    else:
        raise AssertionError("a move at 3x the limits was not refused")


def test_a_missing_rung_names_the_rung_and_the_call_that_adds_it():
    cases = ((lambda: cartesian_path(PENDULUM, [(0.1, 0.0, 0.0)]), "rung 2", "with_kinematics"),
             (lambda: contract_needs_dynamics(PENDULUM), "rung 3", "with_dynamics"),
             (lambda: contract_needs_collision(PENDULUM), "beside the ladder", "with_collision"))
    for call, rung, fix in cases:
        try:
            call()
        except MissingRung as error:
            assert rung in str(error) and fix in str(error), str(error)
        else:
            raise AssertionError(f"not refused: {fix}")
    status = catalog(space=PENDULUM)["cartesian_path"].status
    assert status.state == "locked" and "with_kinematics" in status.fix, status


def test_the_recipe_writes_every_argument_and_rebuilds_from_text_alone():
    trajectory = step_train(PENDULUM, "can11", dwell_s=0.5)
    document = json.loads(recipe(trajectory))
    assert document["calls"] == [{"kind": "step_train", "joint": "can11", "steps_fraction": [0.15, -0.065, 0.065, -0.065, 0.065, -0.15],
                                  "dwell_s": 0.5, "speed": 1.0, "label": "step train", "start": None}], document["calls"]
    assert document["space"]["fingerprint"] == PENDULUM.fingerprint
    assert np.array_equal(rebuild(recipe(trajectory)).positions, trajectory.positions)  # no space passed: limits are in the text
    try:
        rebuild(recipe(trajectory), scenario.fivebar_space())
    except ValueError:
        pass
    else:
        raise AssertionError("a recipe rebuilt on another robot")
    try:
        step_train(PENDULUM, "can11", steps_fraction=(1.5,))
    except ValueError as error:
        assert "[-1, 1]" in str(error)
    else:
        raise AssertionError("an impossible argument was not refused")


def test_seeds_must_be_ints():
    from robot_trajectories import random_waypoints

    try:
        random_waypoints(PENDULUM, 3, seed=1.5)
    except ValueError as error:
        assert "seed must be an int" in str(error)
    else:
        raise AssertionError("a float seed was accepted")


# ---------------------------------------------------------------------- segments


def test_segments_are_a_view_chain_and_split_are_inverses():
    session = scenario.pendulum()
    pieces = split(session)
    assert len(pieces) == 7 and labels(session)[0] == "step train"
    joined = chain(*pieces)
    for name in ("t", "positions", "velocities", "accelerations"):
        assert np.array_equal(getattr(joined, name), getattr(session, name)), name
    assert pieces[1].header.stamp == float(pieces[0].t[-1]), "a piece's offset on the joined clock is its header stamp"
    assert all(band.end_s >= band.start_s for band in bands(session)) and bands(session)[-1].end_s == float(session.t[-1])
    marked = annotate(pieces[0], "overshoot", wait_after=True)
    assert segment_info(marked)[0]["purpose"] == "overshoot" and segment_info(marked)[0]["wait_after"] is True
    assert np.array_equal(marked.positions, pieces[0].positions)


def test_chain_puts_a_transition_where_a_kind_starts_elsewhere():
    first = step_train(PENDULUM, "can11", steps_fraction=(0.3,))
    away = hold(PENDULUM, 0.5, start=np.array([0.1]))
    joined = chain(first, away)
    assert labels(joined) == ["step train", "transition", "hold"], labels(joined)
    assert segment_info(joined)[1]["call"] is None


def test_repeat_is_one_segment_of_plays():
    tape = step_train(PENDULUM, "can11", steps_fraction=(0.2,))
    played = repeat(tape, 3, label="floor")
    assert labels(played) == ["floor"] and len(played.t) == 3 * (len(tape.t) - 1) + 1
    assert [band.label for band in bands(played)][0].startswith("floor: 1/3: ")
    check_kind(played, PENDULUM)


# ---------------------------------------------------------------------- moving boundaries


def test_a_moving_end_hands_its_exact_state_to_a_moving_start():
    out = contract_ramp_out(PENDULUM, "can11")
    assert abs(out.velocities[-1, 0] - 0.3 * 2.0) < 1e-9, out.velocities[-1]
    on = contract_ramp_in(PENDULUM, "can11", start=out)
    assert abs(on.velocities[0, 0] - out.velocities[-1, 0]) < 1e-12 and abs(on.velocities[-1, 0]) < 1e-12
    joined = chain(out, on)
    check_kind(joined, PENDULUM)
    assert segment_info(joined)[0]["ends"] == "moving" and segment_info(joined)[1]["starts"] == "moving"
    assert json.loads(recipe(on))["calls"][0]["start"]["qd"] == [float(out.velocities[-1, 0])], "a moving start records its state"


def test_chain_refuses_a_moving_end_followed_by_a_rest_start():
    out = contract_ramp_out(PENDULUM, "can11")
    for following in (hold(PENDULUM, 0.5, start=out), step_train(PENDULUM, "can11")):
        try:
            chain(out, following)
        except ValueError as error:
            assert "ends moving" in str(error), str(error)
        else:
            raise AssertionError("a moving end was joined to a rest start")


def test_a_trapezoid_refuses_a_moving_end_by_name():
    try:
        TimeLaw.trapezoid(peak_speed_fraction=0.5).leg(np.zeros(1), np.ones(1) * 0.1, PENDULUM.limits, 500.0,
                                                       end_state=BoundaryState(np.ones(1) * 0.2, np.zeros(1)))
    except ValueError as error:
        assert "rest-to-rest" in str(error)
    else:
        raise AssertionError("a trapezoid ended moving")


# ---------------------------------------------------------------------- the time-law seam


class _Straight:
    """A plugin-shaped time law: a two-point leg via the built-in minimum-jerk, refusing more points (Ruckig Community's rule)."""

    name = "contract_community_like"

    def can_service(self, path_points, limits, law, start_state, end_state):
        return "more than two points would go to a cloud service: refused" if len(path_points) > 2 else ""

    def time_parameterize(self, path_points, limits, rate_hz, law, *, start_state=None, end_state=None):
        from robot_msgs.trajectory import JointTrajectory

        arrays = TimeLaw.minimum_jerk(peak_speed_fraction=1.0).leg(path_points[0], path_points[-1], limits, rate_hz)
        return JointTrajectory(t=arrays.t, joint_names=tuple(limits.joint_names), positions=arrays.q, velocities=arrays.qd,
                               accelerations=arrays.qdd)


class _Broken(_Straight):
    name = "contract_broken"

    def time_parameterize(self, path_points, limits, rate_hz, law, *, start_state=None, end_state=None):
        out = super().time_parameterize(path_points, limits, rate_hz, law)
        return out.replace(positions=out.positions + 0.01)  # does not start where it should


def test_time_law_backends_are_one_seam():
    laws = time_laws()
    assert [entry.provider for entry in laws["minimum_jerk"]] == ["robot_trajectories"] and "trapezoid" in laws
    register_time_law(_Straight(), provider="contract-test")
    law = TimeLaw("contract_community_like", peak_speed_fraction=0.5)
    leg = law.leg(np.zeros(1), np.full(1, 0.2), PENDULUM.limits, 500.0)
    assert abs(leg.q[-1, 0] - 0.2) < 1e-12 and np.max(np.abs(leg.qd)) <= 0.5 * 2.0 * (1 + 1e-9), "the speed fraction scaled the limits"
    try:
        law.along(np.linspace(0, 0.2, 9)[:, None], PENDULUM.limits, 500.0)
    except ValueError as error:
        assert "cloud" in str(error), str(error)
    else:
        raise AssertionError("a backend's own refusal (can_service) was not honoured")
    register_time_law(_Broken(), provider="contract-test")
    try:
        TimeLaw("contract_broken", peak_speed_fraction=0.5).leg(np.zeros(1), np.full(1, 0.2), PENDULUM.limits, 500.0)
    except ValueError as error:
        assert "broke the contract" in str(error)
    else:
        raise AssertionError("a backend that broke the contract was used")
    register_time_law(_Straight(), provider="another-distribution")
    try:
        TimeLaw("contract_community_like", peak_speed_fraction=0.5)
    except ValueError as error:
        assert "ambiguous" in str(error)
    else:
        raise AssertionError("two providers of one time law were not refused")


class _NeedsDynamics:
    """A plugin-shaped time law that needs the space's dynamics in one mode (TOPP-RA's torque mode has this shape): with the
    option ``torque`` its ``needs(law)`` names dynamics_model and torque_limits. Minimum-jerk underneath; it records what it was
    handed, and with the option ``too_fast`` hands back a leg the model cannot drive within max_effort."""

    name = "contract_needs_dynamics"

    def __init__(self):
        self.handed: list[dict] = []

    def needs(self, law):
        return ("dynamics_model", "torque_limits") if dict(law.options).get("torque") else ()

    def time_parameterize(self, path_points, limits, rate_hz, law, **handed):
        from robot_trajectories.time_law import BuiltInBackend

        self.handed.append(dict(handed))
        plain = TimeLaw.minimum_jerk(move_s=0.15) if dict(law.options).get("too_fast") else law
        return BuiltInBackend("minimum_jerk").time_parameterize(path_points, limits, rate_hz, plain, start_state=handed.get("start_state"),
                                                                end_state=handed.get("end_state"))


def test_a_time_law_that_needs_the_dynamics_is_handed_the_space_s():
    from fixtures import lane_b

    probe = _NeedsDynamics()
    register_time_law(probe, provider="contract-test")
    torque_law = TimeLaw("contract_needs_dynamics", peak_speed_fraction=0.5, options={"torque": True})
    space = lane_b.pendulum_space()  # CAN 11's limits, a 5 N·m stand-in max_effort, the CAN 11 dynamics stand-in
    waypoints = [[0.1], [-0.1]]
    from robot_trajectories import joint_waypoints, move

    check_kind(joint_waypoints(space, waypoints, time_law=torque_law), space)
    assert probe.handed and all(entry["dynamics"] is space.dynamics for entry in probe.handed), "legs_through did not hand it on"
    probe.handed.clear()
    move(space, time_law=torque_law)
    assert probe.handed and all(entry["dynamics"] is space.dynamics for entry in probe.handed), "move did not hand it on"
    arm = lane_b.ur5e_space()  # kinematics, dynamics and a max_effort: cartesian.follow hands the dynamics to along()
    probe.handed.clear()
    check_kind(cartesian_path(arm, [arm.forward(arm.home)[:3, 3] + (0.05, 0.0, 0.0)], time_law=torque_law), arm)
    assert probe.handed and probe.handed[-1]["dynamics"] is arm.dynamics, "cartesian.follow did not hand it on"
    probe.handed.clear()
    joint_waypoints(space, waypoints, time_law=TimeLaw("contract_needs_dynamics", peak_speed_fraction=0.5))
    assert probe.handed and all("dynamics" not in entry for entry in probe.handed), "a backend that does not need it was handed it"
    for bare, need in ((scenario.pendulum_space(), "dynamics_model"),
                       (JointSpace.from_limits(robots.PENDULUM_LIMITS).with_dynamics(lane_b.Can11Dynamics()), "torque_limits")):
        try:
            joint_waypoints(bare, waypoints, time_law=torque_law)
        except MissingRung as error:
            assert error.need == need and "contract_needs_dynamics" in str(error) and "Add it" in str(error), str(error)
        else:
            raise AssertionError(f"a time law that needs {need} ran without it")
    try:  # the caller holds the model's torque along the backend's result to max_effort (an independent ruler)
        joint_waypoints(lane_b.pendulum_space(max_effort_nm=1.0), [[0.3]],
                        time_law=TimeLaw("contract_needs_dynamics", peak_speed_fraction=1.0, options={"torque": True, "too_fast": True}))
    except ValueError as error:
        assert "broke the contract" in str(error) and "max_effort" in str(error), str(error)
    else:
        raise AssertionError("a backend handed the dynamics returned motion over max_effort, and it was used")


class _AlwaysNeedsDynamics(_NeedsDynamics):
    """The same probe with a static ``needs`` tuple naming the dynamics: every leg it times must be handed the space's model."""

    name = "contract_always_dynamics"
    needs = ("dynamics_model",)


# WHY listed: input_shaped_move shapes a built-in law's closed-form profile and refuses a plugin law by name, so no backend runs
NO_PLUGIN_LAW = {"input_shaped_move"}


def test_every_kind_that_takes_a_time_law_hands_the_backend_the_space_s_dynamics():
    from fixtures import lane_b

    import robot_trajectories as rt

    probe = _AlwaysNeedsDynamics()
    register_time_law(probe, provider="contract-test")
    law = TimeLaw("contract_always_dynamics", peak_speed_fraction=0.5)
    pendulum, five, arm = lane_b.pendulum_space(), lane_b.fivebar_space(), lane_b.ur5e_space()  # each with dynamics and max_effort
    tool = arm.forward(arm.home)[:3, 3]
    calls = {
        "move": lambda: rt.move(pendulum, time_law=law),
        "joint_waypoints": lambda: rt.joint_waypoints(pendulum, [[0.1], [-0.1]], time_law=law),
        "repeat_floor_tape": lambda: rt.repeat_floor_tape(pendulum, time_law=law),
        "via_point_spline": lambda: rt.via_point_spline(pendulum, time_law=law),
        "speed_ladder": lambda: rt.speed_ladder(pendulum, shape="contract_always_dynamics"),
        "space_filling_pose_set": lambda: rt.space_filling_pose_set(arm, time_law=law),
        "cartesian_path": lambda: rt.cartesian_path(arm, [tool + (0.05, 0.0, 0.0)], time_law=law),
        "cartesian_waypoints": lambda: rt.cartesian_waypoints(arm, [tool + (0.05, 0.0, 0.0)], time_law=law),
        "cartesian_a_to_b": lambda: rt.cartesian_a_to_b(arm, tool + (0.05, 0.05, -0.05), via=(tuple(tool + (0.05, 0.0, 0.0)),), time_law=law),
        "cartesian_arc": lambda: rt.cartesian_arc(arm, time_law=law),
        "approach_retreat": lambda: rt.approach_retreat(arm, time_law=law),
        "adept_cycle": lambda: rt.adept_cycle(arm, time_law=law),
        "pick_and_place": lambda: rt.pick_and_place(arm, tool + (0.08, -0.08, -0.12), tool + (-0.08, 0.10, -0.12), lift_m=0.08,
                                                     time_law=law, approach_time_law=law),
        "kinematic_calibration_pose_set": lambda: rt.kinematic_calibration_pose_set(arm, time_law=law),
        "camera_board_views": lambda: rt.camera_board_views(arm, time_law=law),
        "hand_eye_station_set": lambda: rt.hand_eye_station_set(arm, time_law=law),
        "power_up_random_rests": lambda: rt.power_up_random_rests(five, time_law=law),
        "quasi_static_pose_holds": lambda: rt.quasi_static_pose_holds(five, time_law=law),
        "weightless_float": lambda: rt.weightless_float(five, time_law=law),
        "moving_target_track": lambda: rt.moving_target_track(arm, time_law=law),
        "meet_moving_point": lambda: rt.meet_moving_point(arm, time_law=law),
        "track_and_release": lambda: rt.track_and_release(arm, time_law=law, start=rt.meet_moving_point(arm)),
        "moving_pick_and_place": lambda: rt.moving_pick_and_place(arm, time_law=law, approach_time_law=law),
    }
    takes_a_law = {name for name, info in rt.kinds().items() if "time_law" in info.parameters or name == "speed_ladder"}
    missing = sorted(takes_a_law - set(calls) - NO_PLUGIN_LAW - {"envelope_challenge"})
    assert not missing, f"kinds that take a time law but are not held to hand it the dynamics here: {missing}"
    unreached = []
    for name, call in calls.items():
        probe.handed.clear()
        try:
            call()
        except MissingRung:  # the seam refused: the kind timed a leg without handing the dynamics on
            unreached.append(name)
            continue
        if not probe.handed or any(entry.get("dynamics") is None for entry in probe.handed):
            unreached.append(name)
    assert not unreached, f"the time-law backend was not handed the space's dynamics by: {unreached}"


# ---------------------------------------------------------------------- injected slots: dynamics, effort, collision


def test_an_effort_commanded_kind_predicts_its_positions_and_checks_its_torque():
    space = JointSpace.from_limits(robots.PENDULUM_LIMITS.replace(max_effort=np.array([1.0]))).with_dynamics(robots.PendulumDynamics())
    pulse = contract_torque_pulse(space, "can11", start=np.array([0.05]))
    assert segment_info(pulse)[0]["command"] == "effort" and pulse.effort is not None
    assert np.allclose(space.torques(pulse)[1:-1], pulse.effort[1:-1], atol=0.05), "the predicted motion is the command's"
    try:
        contract_torque_pulse(space, "can11", effort_n_m=3.0)
    except UnsafeTrajectory as error:
        assert "torque limit" in str(error), str(error)
    else:
        raise AssertionError("an effort above max_effort was not refused")
    joined = chain(hold(space, 0.2, start=np.array([0.0])), contract_torque_pulse(space, "can11", effort_n_m=0.05, start=np.array([0.0])))
    assert np.all(np.isnan(joined.effort[1:50])) and not np.any(np.isnan(joined.effort[-10:])), "effort is NaN where none is commanded"


def test_a_collision_scene_refuses_a_colliding_output():
    arm = scenario.ur5e_space().with_kinematics(robots.UR5eKinematics())
    tool = arm.forward(arm.home)[:3, 3]
    blocked = arm.with_collision(robots.UR5eScene(box_min=tuple(tool - 0.02), box_max=tuple(tool + 0.02)))
    try:
        hold(blocked, 0.2)
    except UnsafeTrajectory as error:
        assert "sample 0" in str(error) and "the scene" in str(error), str(error)
    else:
        raise AssertionError("a hold inside the box was not refused")
    free = arm.with_collision(robots.UR5eScene())
    check_kind(hold(free, 0.2), free)
    assert catalog(space=free)["random_waypoints"].status.unchecked == (), "with a checker, collisions are checked, not skipped"
    assert catalog(space=arm)["random_waypoints"].status.unchecked == ("collision_model",)


# ---------------------------------------------------------------------- moving targets


def test_the_moving_target_contract():
    pose = np.eye(4)
    pose[:3, 3] = (0.4, 0.0, 0.2)
    belt = LinearMotion.from_sync(pose, (0.0, 2.0, 0.0), 0.25, stamp=1.0, execution_latency=0.1, window=(1.0, 3.0))
    assert np.allclose(belt.target(2.0)[:3, 3], (0.4, 0.25 * 1.1, 0.2)) and np.allclose(belt.velocity(2.0), (0, 0.25, 0, 0, 0, 0))
    assert "constant velocity 0.25" in belt.assumes
    assert TargetWaypoint.from_arm_params((0, 0, 0.05), sync_speed=0.25, sync_at_target_time=False).sync == "chase"
    assert TargetWaypoint.from_arm_params((0, 0, 0.05), sync_speed=0.0, sync_at_target_time=True).sync == "inherit"
    error = CannotCatch("target_outruns_arm", "0.8 m/s against 0.5 m/s")
    assert error.code == "cannot_catch" and "outruns" in str(error)
    windowed = as_moving_target(lambda t: pose)
    assert windowed.window == (-math.inf, math.inf)
    arm = scenario.ur5e_space().with_kinematics(robots.UR5eKinematics())
    line = cartesian_path(arm, [arm.forward(arm.home)[:3, 3] + (0.1, 0.0, 0.0)])
    t = line.t

    def exact(time: float) -> np.ndarray:  # the trajectory's own tool path, interpolated: a target it tracks by construction
        return arm.forward([np.interp(time, t, line.positions[:, joint]) for joint in range(arm.dof)])

    middle = float(t[len(t) // 2])
    assert synchronised(arm, line, exact, t0=middle, t1=middle + 0.05), synchronised(arm, line, exact, t0=middle, t1=middle + 0.05).why

    def shifted(time: float) -> np.ndarray:
        pose = exact(time)
        pose[:3, 3] += (0.0, 0.002, 0.0)
        return pose

    report = synchronised(arm, line, shifted, t0=middle, t1=middle + 0.05)
    assert not report and "2.000 mm" in report.why, report


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_contract (the kind contract)"))
