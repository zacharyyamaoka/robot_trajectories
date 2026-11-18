"""Lane B's kinds (quasi-static, hold, stress, and every torque / event-ended kind), each held to the kind contract on the robots
it applies to, and to what it is FOR (a claim that can fail).

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_lane_b.py

Torque kinds run on the CAN 11 pendulum stand-in (gravity 4.66 N.m, inertia 0.1219 kg.m^2, Coulomb 0.833 N.m: tests/fixtures/lane_b.py
says which numbers are the registry's and which are stand-ins); the arms' gravity is a stand-in too.
"""

from __future__ import annotations

import math

import numpy as np

import _runner
from fixtures import lane_b as L
from fixtures import robots
from kind_contract import check_kind
from robot_trajectories import (JointSpace, MissingRung, UnsafeTrajectory, bands, breakaway_torque_ramp, burn_in_wiggle, catalog,
                                disturbance_push, endurance_duty_cycle, free_swing_release, hard_stop_seek, heating_run,
                                locked_output_torque_ramp, power_up_random_rests, quasi_static_pose_holds, relay_feedback, segment_info,
                                small_reversals, torque_chirp, torque_steps, weightless_float)

LANE_B_ROWS = ("torque_chirp", "torque_steps", "free_swing_release", "relay_feedback", "breakaway_torque_ramp", "small_reversals",
               "locked_output_torque_ramp", "quasi_static_pose_holds", "hard_stop_seek", "burn_in_wiggle", "heating_run",
               "endurance_duty_cycle", "power_up_random_rests", "disturbance_push", "weightless_float")


# ---------------------------------------------------------------------- helpers


def crossings(values: np.ndarray) -> int:
    signs = np.sign(values[np.abs(values) > 1e-9])
    return int(np.count_nonzero(np.diff(signs) != 0))


def refused(call, error=Exception, *words: str) -> str:
    try:
        call()
    except error as caught:
        for word in words:
            assert word in str(caught), (word, str(caught))
        return str(caught)
    raise AssertionError(f"not refused: expected {error.__name__} naming {words}")


def effort_checks(out, space, joint: int = 0) -> np.ndarray:
    """What every torque profile owes: declared effort-commanded, effort inside the limit, the motion the model's prediction of it."""
    entry = segment_info(out)[0]
    assert entry["command"] == "effort" and out.effort is not None, entry
    cap = np.asarray(space.limits.max_effort, float)
    assert np.all(np.nanmax(np.abs(out.effort), axis=0) <= cap + 1e-9), (np.nanmax(np.abs(out.effort), axis=0), cap)
    return np.asarray(out.effort)


def reproduces_prediction(out, space) -> None:
    """The positions of an effort profile are ``space.predict(start, effort)``: the guard's and the contract's reading, one reading."""
    effort = np.asarray(out.effort)
    first = np.flatnonzero(~np.isnan(effort[:, 0]))
    assert first[0] == 0, "an effort profile starts commanding at its first sample"
    q, qd, qdd = space.predict(out.positions[0], effort)
    assert np.allclose(q, out.positions) and np.allclose(qd, out.velocities) and np.allclose(qdd, out.accelerations)


PEND = L.pendulum_space()
PEND_LOW_FRICTION = L.pendulum_space(L.Can11Dynamics(coulomb=0.05))
PEND_FREE = L.pendulum_space(L.Can11Dynamics(coulomb=0.005, damping=0.005))  # swings for many periods


def test_the_rows_are_implemented():
    todo = {entry.catalog_id for entry in catalog().todo}
    assert not todo & set(LANE_B_ROWS), todo & set(LANE_B_ROWS)
    names = set(catalog().implemented.names)
    assert set(LANE_B_ROWS) <= names


def test_the_fixture_stands_in_for_can11():
    dynamics = L.Can11Dynamics()
    assert (dynamics.gravity_torque, dynamics.inertia, dynamics.coulomb) == (4.661, 0.12193, 0.833)
    assert dynamics.rate_hz == PEND.rate_hz, "the stick must be stepped at the space's rate"
    # sticks below breakaway, slides above it, and never sticks at speed
    assert float(dynamics.forward([0.0], [0.0], [0.8])[0]) == 0.0
    assert float(dynamics.forward([0.0], [0.0], [1.0])[0]) > 0.0
    assert float(dynamics.forward([0.0], [1.0], [0.0])[0]) < -5.0


# ---------------------------------------------------------------------- torque kinds


def test_torque_chirp():
    out = torque_chirp(PEND)
    check_kind(out, PEND, other_space=L.pendulum_space(max_effort_nm=4.0))
    effort = effort_checks(out, PEND)
    reproduces_prediction(out, PEND)
    entry = segment_info(out)[0]
    assert entry["ends"] == "anywhere" and "rung down" in entry["until"], entry
    amplitude = entry["torque_amplitude_n_m"]
    deviation = effort[:, 0] - entry["hold_torque_n_m"][0]
    assert abs(float(np.max(np.abs(deviation))) - amplitude) < 1e-6 and amplitude <= 0.25 * 4.661 + 1e-9, amplitude
    assert amplitude > 0.5, "the solved amplitude kept a useful excitation, not a token one"
    t = np.asarray(out.t)
    early, late = (crossings(deviation[(t > a) & (t < a + 3)]) for a in (2.0, 11.0))
    assert late > 2 * early, f"it is a chirp: {early} zero crossings at 2-5 s, {late} at 11-14 s"
    assert not np.any(np.isnan(effort)), "an effort-commanded kind commands every sample"
    assert float(np.max(np.abs(out.accelerations))) <= 7.3896 + 1e-9
    # the drive limit and the model are respected, or the kind refuses by name
    refused(lambda: torque_chirp(L.pendulum_space(max_effort_nm=1.0), start=[0.3]), ValueError, "cannot hold")
    tight = L.pendulum_space(max_effort_nm=0.5)
    assert float(np.nanmax(np.abs(torque_chirp(tight).effort))) <= 0.5 + 1e-9
    refused(lambda: torque_chirp(JointSpace.from_limits(robots.PENDULUM_LIMITS).with_dynamics(L.Can11Dynamics())), MissingRung, "torque")
    no_forward = JointSpace.from_limits(L.pendulum_limits()).with_dynamics(_NoForward(L.SerialGravity((0,), (0.5,), (0.25,), (1.0,), (0.1,))))
    refused(lambda: torque_chirp(no_forward), MissingRung, "forward")
    refused(lambda: torque_chirp(PEND, f0_hz=6.0, f1_hz=1.0), ValueError, "f0_hz")


def test_torque_steps():
    out = torque_steps(PEND, steps=4, hold_s=1.0)
    check_kind(out, PEND)
    effort = effort_checks(out, PEND)
    reproduces_prediction(out, PEND)
    amplitude = segment_info(out)[0]["step_torque_n_m"]
    deviation = effort[:, 0]
    levels = np.round(deviation / amplitude, 6)
    assert set(np.unique(levels)) == {-1.0, 0.0, 1.0}, np.unique(levels)
    order = [levels[int(i)] for i in (0.5 * 500, 1.5 * 500, 2.5 * 500, 3.5 * 500, 4.5 * 500, 5.5 * 500, 6.5 * 500, 7.5 * 500, 8.5 * 500)]
    assert order == [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0], order
    # the joint follows the torque: it is displaced up under the + step and down under the - step (the sign of the step response)
    q = out.positions[:, 0]
    assert q[int(1.9 * 500)] > 0.02 and q[int(3.9 * 500)] < q[int(2.9 * 500)], "a + step pushes the joint up, a - step down"
    refused(lambda: torque_steps(PEND, steps=0), ValueError, "steps")


def test_free_swing_release():
    out = free_swing_release(PEND)
    check_kind(out, PEND)
    entry = segment_info(out)[0]
    assert entry["ends"] == "anywhere" and entry["until"].startswith("released: kp = kd = release_kd"), entry
    assert entry["release"]["kp"] == 0.0 and entry["release"]["kd"] == 0.0
    labels = [band.label for band in bands(out)]
    assert labels[0].endswith("lift") and any("released" in text for text in labels), labels
    released = [band for band in bands(out) if "released" in band.label][0]
    t = np.asarray(out.t)
    coast = t >= released.start_s
    assert np.nanmax(np.abs(out.effort[coast])) == 0.0, "released means no torque (kd = 0)"
    assert np.all(np.isnan(out.effort[t < released.start_s - 1e-9])), "the lift is position-commanded: no effort there"
    assert float(np.max(np.abs(out.positions[coast, 0] - 0.0))) <= entry["release"]["at_rad"] + 1e-9, "it never swings past where it was released"
    # the claim it is FOR: a nearly frictionless pendulum swings at its own frequency, T = 2 pi sqrt(J / G), whatever the lift
    free = free_swing_release(PEND_FREE, coast_s=8.0)
    check_kind(free, PEND_FREE)
    start_of_coast = [band for band in bands(free) if "released" in band.label][0].start_s
    t, q = np.asarray(free.t), free.positions[:, 0]
    mask = t >= start_of_coast
    up = np.flatnonzero((q[mask][:-1] < 0) & (q[mask][1:] >= 0))
    assert len(up) >= 4, f"{len(up)} upward crossings: it should keep swinging with 0.05 N.m of friction"
    period = float(np.mean(np.diff(t[mask][up])))
    expected = 2 * math.pi * math.sqrt(0.12193 / 4.661)
    assert abs(period - expected) < 0.05 * expected, (period, expected)
    # damping only (kd) decays faster than a free swing
    damped = free_swing_release(PEND_FREE, release_kd=0.5)
    start_d = [band for band in bands(damped) if "released" in band.label][0].start_s
    t = np.asarray(damped.t)
    assert np.max(np.abs(damped.positions[t > start_d + 4.0, 0])) < 0.5 * np.max(np.abs(free.positions[np.asarray(free.t) > start_of_coast + 4.0, 0]))
    assert segment_info(damped)[0]["release"]["kd"] == 0.5
    # without a model it plans only the lift and says so
    bare = JointSpace.from_limits(robots.PENDULUM_LIMITS)
    plan = free_swing_release(bare)
    check_kind(plan, bare)
    assert out.t[-1] > plan.t[-1] and "not predicted" in " ".join(segment_info(plan)[0]["notes"])
    refused(lambda: free_swing_release(PEND, start=[0.33]), ValueError, "room")


def test_relay_feedback():
    # CAN 11's own 0.833 N.m Coulomb sits above the default 0.2 N.m relay: the model says it never switches, and the kind says so
    stalls = relay_feedback(PEND)
    check_kind(stalls, PEND)
    assert "never completes" in " ".join(segment_info(stalls)[0]["notes"]) and float(np.ptp(stalls.positions)) == 0.0
    out = relay_feedback(PEND_LOW_FRICTION, relay_torque_nm=0.4, cycles=6)
    check_kind(out, PEND_LOW_FRICTION)
    effort = effort_checks(out, PEND_LOW_FRICTION)
    reproduces_prediction(out, PEND_LOW_FRICTION)
    entry = segment_info(out)[0]
    assert entry["ends"] == "anywhere" and entry["until"].startswith("limit cycle read"), entry
    relay = effort[:, 0]  # the hold at the start (q = 0) is 0, so the effort is the relay
    assert set(np.round(np.unique(relay), 9)) == {-0.4, 0.4}, np.unique(relay)
    switches = int(np.count_nonzero(np.diff(np.sign(relay)) != 0))
    assert switches >= 12 and out.t[-1] < 10.0, (switches, out.t[-1])
    # it follows the measurement: the relay flips only after the error passes the hysteresis band
    q = out.positions[:, 0]
    flips = np.flatnonzero(np.diff(np.sign(relay)) != 0) + 1
    assert all(abs(q[i]) >= 0.001 - 1e-6 for i in flips), "a flip needs |error| >= hysteresis_rad"
    # the measured error is what the relay answers: a bigger hysteresis band swings wider
    wide = relay_feedback(PEND_LOW_FRICTION, relay_torque_nm=0.4, hysteresis_rad=0.004)
    assert 0.5 * np.ptp(wide.positions) > 0.5 * np.ptp(out.positions)
    # too hard a relay breaks the position limits of the prediction: the guard refuses it by name
    refused(lambda: relay_feedback(PEND_LOW_FRICTION, relay_torque_nm=3.0), UnsafeTrajectory, "acceleration")
    bare = JointSpace.from_limits(L.pendulum_limits())
    plan = relay_feedback(bare)
    check_kind(plan, bare)
    assert segment_info(plan)[0]["command"] == "effort"


def test_breakaway_torque_ramp():
    out = breakaway_torque_ramp(PEND)
    check_kind(out, PEND)
    effort = effort_checks(out, PEND)
    reproduces_prediction(out, PEND)
    entry = segment_info(out)[0]
    assert entry["ends"] == "anywhere" and entry["until"].startswith("breakaway: |q - q_start|"), entry
    breakaway = entry["ramp"]["predicted_breakaway_n_m"]
    # the claim it is FOR: the ramp finds the static friction, within what the leave threshold lets through
    assert 0.833 <= breakaway <= 0.833 + 0.15, breakaway
    left = int(np.flatnonzero(np.abs(out.positions[:, 0]) > 0.002)[0])
    assert abs(effort[left, 0] - breakaway) < 0.5 / 500 + 1e-9
    # the other flank, and a different friction, are found too
    down = breakaway_torque_ramp(PEND, direction=-1.0)
    check_kind(down, PEND)
    assert out.positions[-1, 0] > 0 > down.positions[-1, 0]
    soft = breakaway_torque_ramp(PEND_LOW_FRICTION)
    assert segment_info(soft)[0]["ramp"]["predicted_breakaway_n_m"] < 0.2
    # the ramp is linear at the asked rate
    assert abs((effort[300, 0] - effort[100, 0]) / (200 / 500) - 0.5) < 1e-6
    bare = JointSpace.from_limits(L.pendulum_limits())
    plan = breakaway_torque_ramp(bare)
    check_kind(plan, bare)
    assert np.ptp(plan.positions) == 0.0 and segment_info(plan)[0]["command"] == "effort"
    refused(lambda: breakaway_torque_ramp(PEND, start_torque_nm=9.0), ValueError, "start_torque_nm")


def test_locked_output_torque_ramp():
    out = locked_output_torque_ramp(PEND, peak_torque_nm=2.0, ramp_s=4.0, cycles=2)
    check_kind(out, PEND)
    effort = effort_checks(out, PEND)[:, 0]
    assert segment_info(out)[0]["output"] == "locked" and "locked" in " ".join(segment_info(out)[0]["notes"])
    assert float(np.ptp(out.positions)) == 0.0 and float(np.max(np.abs(out.velocities))) == 0.0, "the output is locked: positions are held"
    assert abs(effort.max() - 2.0) < 1e-3 and abs(effort.min() + 2.0) < 1e-3
    assert out.t[-1] == 2 * 4 * 4.0, out.t[-1]
    slope = np.diff(effort) * 500
    assert abs(slope.max() - 2.0 / 4.0) < 1e-6 and abs(slope.min() + 2.0 / 4.0) < 1e-6, "ramps at peak / ramp_s, up and down, both signs"
    assert crossings(effort) == 3, "up, down, then through zero to the other sign and back, twice"
    refused(lambda: locked_output_torque_ramp(PEND, peak_torque_nm=6.0), ValueError, "torque limit")
    refused(lambda: locked_output_torque_ramp(JointSpace.from_limits(robots.PENDULUM_LIMITS)), MissingRung, "torque")


def test_heating_run():
    out = heating_run(PEND, heat_current_fraction=0.5, probe_every_s=5.0, heat_s=22.0, cool_s=10.0)
    check_kind(out, PEND)
    effort = effort_checks(out, PEND)[:, 0]
    entry = segment_info(out)[0]
    assert entry["until"].startswith("the winding temperature reaches its limit") and entry["heat_effort_n_m"] == 2.5
    assert entry["probe_times_s"] == [5.0, 10.0, 15.0, 20.0, 22.0], entry["probe_times_s"]
    t = np.asarray(out.t)
    assert np.all(effort[t <= 22.0] == 2.5) and np.all(effort[t > 22.0] == 0.0), "constant current, then none"
    assert float(np.ptp(out.positions)) == 0.0, "the rotor holds: a current, not a motion"
    assert [b.label for b in bands(out)][-1].endswith("cool-down") and abs(out.t[-1] - 32.0) < 1e-9
    big = heating_run(PEND)  # the catalog's 600 s + 900 s, one joint: the plan is as long as asked
    assert abs(big.t[-1] - 1500.0) < 1e-6 and len(bands(big)) == 61
    refused(lambda: heating_run(PEND, heat_current_fraction=1.5), ValueError, "heat_current_fraction")


# ---------------------------------------------------------------------- position kinds on one joint


def test_small_reversals():
    cases = ((PEND, None), (L.fivebar_space(), "j2"), (L.ur5e_space(), "elbow_joint"))
    for space, joint in cases:
        out = small_reversals(space, joint, amplitude_rad=0.01, speed_rad_s=0.02, reversals=6, dwell_s=1.0)
        check_kind(out, space)
        j = space.joint_index(joint)
        q = out.positions[:, j] - space.home[j]
        assert abs(q.max() - 0.01) < 1e-9 and abs(q.min() + 0.01) < 1e-9, "it turns at +-amplitude about the held pose"
        assert abs(np.max(np.abs(out.velocities[:, j])) - 0.02) / 0.02 < 0.02, "peak speed is the asked slow speed"
        others = [i for i in range(space.dof) if i != j]
        assert all(float(np.ptp(out.positions[:, i])) == 0.0 for i in others), "the other joints stay still"
        moving = np.abs(out.velocities[:, j]) > 1e-9
        direction = np.sign(out.velocities[moving, j])
        assert int(np.count_nonzero(np.diff(direction) != 0)) == 6 + 1, "six reversals about the pose, and the turn back at the end"
        assert abs(out.positions[-1, j] - space.home[j]) < 1e-12, "it ends back where it was held"
    refused(lambda: small_reversals(PEND, amplitude_rad=0.5), ValueError, "outside its limits")


def test_hard_stop_seek():
    cases = ((PEND, None), (L.fivebar_space(), "c1"), (L.ur5e_space(), "elbow_joint"))
    for space, joint in cases:
        out = hard_stop_seek(space, joint, speed_rad_s=0.1, max_travel_rad=0.5)
        check_kind(out, space)
        entry = segment_info(out)[0]
        j = space.joint_index(joint)
        assert entry["ends"] == "anywhere" and entry["until"].startswith("stall: |qd| < 0.01 rad/s") and entry["seek"]["cap_steps_nm"] == [0.5, 1.0, 2.0]
        assert abs(np.max(np.abs(out.velocities[:, j])) - 0.1) < 1e-3, "the setpoint slews at the seek speed"
        assert np.all(out.positions[:, j] >= space.bounds.low[j] - 1e-12) and np.all(out.positions[:, j] <= space.bounds.high[j] + 1e-12)
        travel = out.positions[-1, j] - out.positions[0, j]
        assert abs(travel - min(0.5, space.bounds.high[j] - space.home[j])) < 1e-9, "it seeks as far as the expected stop allows"
        assert all(float(np.ptp(out.positions[:, i])) == 0.0 for i in range(space.dof) if i != j), "the others are held"
    two = hard_stop_seek(L.ur5e_space(), "elbow_joint", touches=2)
    check_kind(two, L.ur5e_space())
    j = 2
    presses = int(np.count_nonzero((np.abs(two.positions[:, j] - two.positions[:, j].max()) < 1e-12) & (np.abs(two.velocities[:, j]) < 1e-9)))
    assert presses >= 2 and any("press again" in b.label for b in bands(two)), "two touches: back off and press again"
    down = hard_stop_seek(PEND, direction=-1.0)
    assert down.positions[-1, 0] < 0, "direction -1 seeks the low side"
    # it refuses without the drive's torque limit, and a cap above it
    refused(lambda: hard_stop_seek(JointSpace.from_limits(robots.PENDULUM_LIMITS)), MissingRung, "torque")
    refused(lambda: hard_stop_seek(PEND, cap_steps_nm=(1.0, 9.0)), ValueError, "torque limit")
    refused(lambda: hard_stop_seek(PEND, cap_steps_nm=(2.0, 1.0)), ValueError, "cap_steps_nm")


def test_burn_in_wiggle():
    home_pendulum = L.pendulum_home_space()
    cases = ((home_pendulum, None), (L.fivebar_space(), None), (L.ur5e_space(), None))
    for space, joint in cases:
        out = burn_in_wiggle(space, joint, amplitude_rad=0.3, speeds_rad_s=(0.35, 0.7), sweeps=1)
        check_kind(out, space)
        q = out.positions
        centred = q - space.home
        assert np.allclose(q[0], space.home) and np.allclose(q[-1], space.home)
        # one joint at a time: at every sample at most one joint is away from home
        away = (np.abs(centred) > 1e-12).sum(axis=1)
        assert int(away.max()) == 1, f"others held: {int(away.max())} joints away at once"
        swung = [int(np.flatnonzero(np.abs(centred[:, j]) > 1e-12).size > 0) for j in range(space.dof)]
        assert sum(swung) == space.dof, "every joint is swept in turn"
        speeds = sorted({round(float(np.max(np.abs(out.velocities[:, j]))), 1) for j in range(space.dof)})
        assert max(speeds) == 0.7, speeds
    # a start elsewhere is walked to home first
    pend = L.pendulum_home_space()
    walked = burn_in_wiggle(pend, speeds_rad_s=(0.35,), sweeps=1, amplitude_rad=0.2, start=[0.2])
    assert bands(walked)[0].label.endswith("walk to home") and abs(walked.positions[-1, 0]) < 1e-12
    # the amplitude is cut to the room (and says so); a speed above the limit is cut to it
    cut = burn_in_wiggle(L.fivebar_space(), "j2", amplitude_rad=1.0, speeds_rad_s=(5.0,), sweeps=1)
    notes = " ".join(segment_info(cut)[0]["notes"])
    assert "amplitude cut" in notes and "cut to the joint's" in notes, notes
    # collision: a scene across the sweep refuses it, a distant one passes
    arm = L.ur5e_space().with_collision(robots.UR5eScene())
    check_kind(burn_in_wiggle(arm, "elbow_joint", amplitude_rad=0.2, speeds_rad_s=(0.7,), sweeps=1), arm)
    tool = L.ur5e_space().forward(L.ur5e_space().home)[:3, 3]
    blocked = L.ur5e_space().with_collision(robots.UR5eScene(box_min=tuple(tool - 0.02), box_max=tuple(tool + 0.02)))
    refused(lambda: burn_in_wiggle(blocked, "elbow_joint", amplitude_rad=0.2, speeds_rad_s=(0.7,), sweeps=1), UnsafeTrajectory, "collides")
    refused(lambda: burn_in_wiggle(PEND), MissingRung, "home")


def test_disturbance_push():
    for space in (L.fivebar_space(), L.ur5e_space()):
        out = disturbance_push(space)
        check_kind(out, space)
        entry = segment_info(out)[0]
        assert entry["push"] == {"force_n": 15.0, "direction": [1.0, 0.0, 0.0], "frame": "base", "start_s": 0.5, "duration_s": 0.4}
        assert float(np.max(np.ptp(out.positions, axis=0))) == 0.0 and float(np.max(np.abs(out.velocities))) == 0.0, "the trajectory is a plain hold"
        assert abs(out.t[-1] - 2.0) < 1e-9
        push = [band for band in bands(out) if "during the push" in band.label][0]
        assert (push.start_s, push.end_s) == (0.5, 0.9), push
    refused(lambda: disturbance_push(L.fivebar_space(), push_at_s=1.8, push_s=0.4), ValueError, "recover")


# ---------------------------------------------------------------------- pose sets and higher-order kinds


def _station_checks(out, space, count: int) -> list[np.ndarray]:
    entry = segment_info(out)[0]
    stations = [np.array(row) for row in entry["stations"]]
    assert len(stations) == count
    dwells = [band for band in bands(out) if ": station " in band.label and "move" not in band.label]
    assert len(dwells) == count and [band.label.split("station ")[-1].split()[0] for band in dwells] == [str(i) for i in range(1, count + 1)]
    t = np.asarray(out.t)
    for band, station in zip(dwells, stations, strict=True):
        window = (t >= band.start_s) & (t <= band.end_s)
        assert np.allclose(out.positions[window][0], station) and np.allclose(out.positions[window][-1], station, atol=1e-9)
    return stations


def test_quasi_static_pose_holds():
    for space in (L.fivebar_space(), L.ur5e_space()):
        out = quasi_static_pose_holds(space, poses=6, hold_s=2.0, dither_amplitude_rad=0.01, seed=3)
        check_kind(out, space)
        stations = _station_checks(out, space, 6)
        for station in stations:
            space.forward(station)  # a tool pose exists at every station
            assert np.all(np.abs(L.FIVEBAR_DYNAMICS.gravity(station) if space.dof == 2 else L.UR5E_DYNAMICS.gravity(station)) < 1e9)
            if space.reach is not None:
                assert space.reach.nearest_is_reachable(station), "inside the reach map"
        entry = segment_info(out)[0]
        assert len(entry["hold_torque_n_m"]) == 6 and np.allclose(entry["hold_torque_n_m"][0], space.torque(stations[0], np.zeros(space.dof), np.zeros(space.dof)))
        # the claim it is FOR: friction-cancelling dither. at each hold every joint reaches +-amplitude and its velocity takes both signs
        t = np.asarray(out.t)
        for band, station in zip([b for b in bands(out) if "move" not in b.label], stations, strict=True):
            window = (t >= band.start_s) & (t <= band.end_s)
            swing = out.positions[window] - station
            assert np.allclose(np.max(np.abs(swing), axis=0), 0.01, atol=1e-9), np.max(np.abs(swing), axis=0)
            assert np.all(np.max(swing, axis=0) > 0.009) and np.all(np.min(swing, axis=0) < -0.009), "full amplitude on both sides of the station"
            assert np.all(np.max(out.velocities[window], axis=0) > 0) and np.all(np.min(out.velocities[window], axis=0) < 0)
        # the stations spread over the room: no two within 5 % of the joint ranges of each other
        span = space.bounds.high - space.bounds.low
        gaps = [float(np.linalg.norm((a - b) / span)) for i, a in enumerate(stations) for b in stations[i + 1:]]
        assert min(gaps) > 0.05, min(gaps)
        # seeded: the same seed is the same set, another seed another
        same = quasi_static_pose_holds(space, poses=6, seed=3)
        other = quasi_static_pose_holds(space, poses=6, seed=4)
        assert segment_info(same)[0]["stations"] == segment_info(out)[0]["stations"] != segment_info(other)[0]["stations"]
    # a fold margin the kinematics reports is honoured (and a space with no dynamics is locked, naming the rung)
    refused(lambda: quasi_static_pose_holds(L.fivebar_space(dynamics=None)), MissingRung, "dynamics")
    scene = L.ur5e_space().with_collision(robots.UR5eScene())
    check_kind(quasi_static_pose_holds(scene, poses=3, seed=1), scene)


def test_endurance_duty_cycle():
    for space in (L.fivebar_space(), L.ur5e_space()):
        cycle = endurance_duty_cycle(space, cycles=1)
        out = endurance_duty_cycle(space, cycles=4)
        check_kind(out, space)
        assert segment_info(out)[0]["made_by"] == "endurance_duty_cycle" and len(segment_info(out)) == 1, "one segment of plays"
        assert abs(out.t[-1] - 4 * cycle.t[-1]) < 1e-6 and np.allclose(out.positions[0], out.positions[-1]), "four plays of one cycle, back at the start"
        assert float(np.max(np.abs(out.positions - space.home))) > 0.1, "the cycle moves"
        # a gap check every N seconds of play: its own segment, marked wait_after
        spaced = endurance_duty_cycle(space, cycles=5, gap_check_every_s=float(cycle.t[-1]) * 2)
        check_kind(spaced, space)
        entries = segment_info(spaced)
        checks = [e for e in entries if "gap check" in e["label"]]
        assert len(checks) == 2 and all(e["wait_after"] for e in checks) and len(entries) == 5, [e["label"] for e in entries]
        # S3: the motion is duty_fraction of the period
        s3 = endurance_duty_cycle(space, cycles=2, duty_type="S3", duty_fraction=0.5)
        check_kind(s3, space)
        moving = np.abs(s3.velocities).max(axis=1) > 1e-9
        assert abs(moving.mean() - 0.5 * (1 - 0.2 / (s3.t[-1] / 2)) ) < 0.1 or 0.3 < moving.mean() < 0.55, moving.mean()
        assert s3.t[-1] > 1.8 * out.t[-1] / 2, "the rest lengthens the period"
    refused(lambda: endurance_duty_cycle(L.fivebar_space(), duty_type="S2"), ValueError, "S2")
    refused(lambda: endurance_duty_cycle(JointSpace.from_limits(robots.PENDULUM_LIMITS)), MissingRung, "home")


def _conditioning(space, q: np.ndarray) -> float:
    """An independent ruler: sigma_min / sigma_max of the tool position's Jacobian, central differences (written here, not imported)."""
    columns = []
    for joint in range(space.dof):
        step = np.zeros(space.dof)
        step[joint] = 1e-5
        try:
            columns.append((space.forward(q + step)[:3, 3] - space.forward(q - step)[:3, 3]) / 2e-5)
        except ValueError:
            return 0.0
    values = np.linalg.svd(np.array(columns).T, compute_uv=False)
    return float(values[-1] / values[0])


def test_power_up_random_rests():
    for space in (L.fivebar_space(), L.ur5e_space()):
        out = power_up_random_rests(space, power_ups=24, seed=5)
        check_kind(out, space)
        stations = _station_checks(out, space, 24)
        strata = segment_info(out)[0]["strata"]
        assert sorted({s: strata.count(s) for s in set(strata)}.items()) == [("flipped", 6), ("free", 10), ("singular", 8)], strata
        for station in stations:
            space.forward(station)
            if space.reach is not None:
                assert space.reach.nearest_is_reachable(station)
        # the claim it is FOR: the oversampled strata really are the awkward ones
        by = {name: [i for i, s in enumerate(strata) if s == name] for name in ("singular", "flipped", "free")}
        cond = {name: float(np.median([_conditioning(space, stations[i]) for i in idx])) for name, idx in by.items()}
        assert cond["singular"] < cond["free"], cond
        far = {name: float(np.median([np.max(np.abs(stations[i] - space.home)) for i in idx])) for name, idx in by.items()}
        assert far["flipped"] > far["free"], far
        assert segment_info(power_up_random_rests(space, power_ups=24, seed=5))[0]["stations"] == segment_info(out)[0]["stations"]
    refused(lambda: power_up_random_rests(L.fivebar_space(), strata=("sideways",)), ValueError, "strata")


def test_weightless_float():
    for space in (L.fivebar_space(), L.ur5e_space()):
        out = weightless_float(space, poses=3, float_s=2.0, hold_s=1.0, seed=7)
        check_kind(out, space)
        entry = segment_info(out)[0]
        assert entry["command"] == "effort" and entry["ends"] == "anywhere" and "float_s" in entry["until"]
        floats = [band for band in bands(out) if "floating" in band.label]
        assert len(floats) == 3
        t = np.asarray(out.t)
        cap = np.asarray(space.limits.max_effort, float)
        assert np.all(np.nanmax(np.abs(out.effort), axis=0) <= cap + 1e-9)
        for band, station in zip(floats, [np.array(row) for row in entry["stations"]], strict=True):
            window = (t > band.start_s) & (t <= band.end_s)
            assert abs((band.end_s - band.start_s) - 2.0) < 1e-9, "the float lasts float_s"
            gravity_only = space.torque(station, np.zeros(space.dof), np.zeros(space.dof))
            assert np.allclose(out.effort[window], gravity_only, atol=1e-9), "only the gravity torque is sent"
            drift = float(np.max(np.abs(out.positions[window] - station)))
            assert drift < 1e-9, f"under its own model the float does not drift ({drift})"
            before = (t < band.start_s) & (t >= band.start_s - 1.0 + 1e-9)
            assert np.all(np.isnan(out.effort[before])), "the hold before it is position-commanded"
        # the claim it is FOR can fail: against a plant 10 % heavier than the model, the same effort drifts
        heavy = L.SerialGravity(**{**{key: getattr(L.FIVEBAR_DYNAMICS if space.dof == 2 else L.UR5E_DYNAMICS, key) for key in
                                      ("plane_joints", "lengths", "centres", "inertia", "damping")},
                                   "masses": tuple(1.1 * m for m in (L.FIVEBAR_DYNAMICS if space.dof == 2 else L.UR5E_DYNAMICS).masses)})
        truth = JointSpace.from_limits(space.limits, home=space.home).with_dynamics(heavy)
        band = floats[0]
        window = (t > band.start_s) & (t <= band.end_s)
        station = np.array(entry["stations"][0])
        q, _, _ = truth.predict(station, out.effort[window])
        assert float(np.max(np.abs(q - station))) > np.radians(0.5), "a 10 % gravity-model error shows as drift"
    refused(lambda: weightless_float(L.fivebar_space(dynamics=None)), MissingRung, "dynamics")
    no_forward = L.fivebar_space(dynamics=_NoForward(L.FIVEBAR_DYNAMICS))
    refused(lambda: weightless_float(no_forward, poses=1), MissingRung, "forward")
    refused(lambda: weightless_float(L.fivebar_space(), float_s=0.0), ValueError, "float_s")


class _NoForward:
    """A dynamics model that predicts torque but cannot simulate: the effort-commanded kinds must lock, never guess."""

    name, cache_key = "torque-only stand-in", "torque-only-1"

    def __init__(self, inner):
        self.inner = inner

    def torque(self, q, qd, qdd):
        return self.inner.torque(q, qd, qdd)


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_lane_b"))
