"""Lane A's kinds (excitation and controller tests), each held to the kind contract on the robots it applies to, and to what it is for.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_lane_a.py

Robots (tests/scenario.py, tests/fixtures/robots.py): the 1D pendulum (CAN 11's limits), the 2D five-bar (with its reach map) and the 6D
UR5e (in its demo box). A single-joint kind runs on the pendulum and on one joint of the other two; a multi-joint kind on the
five-bar and the UR5e. Every kind then gets one claim about what it is FOR that can fail, read from the samples, never from the
kind's own labels or notes.
"""

from __future__ import annotations

import dataclasses
import importlib

import numpy as np

import _runner
import scenario
from fixtures import lane_a, robots
from kind_contract import check_kind
from robot_trajectories import (MissingRung, UnsafeTrajectory, biased_dither, band_limited_noise, doublet_3211, fourier_series_excitation,
                                bands, labels, notes, orthogonal_multisine, prbs, rapid_replan, rebuild, recipe, segment_info, slow_plus_fast_sines,
                                split, staircase, stepped_sine_dwell)

SINGLE = (("pendulum", scenario.pendulum_space, "can11"), ("five-bar", scenario.fivebar_space, "j2"),
          ("UR5e", scenario.ur5e_space, "elbow_joint"))
MULTI = (("five-bar", scenario.fivebar_space), ("UR5e", scenario.ur5e_space))


def deg(values):
    return np.degrees(values)


def fit_lines(t: np.ndarray, signal: np.ndarray, frequencies) -> tuple[np.ndarray, float]:
    """Least-squares amplitudes of sines at ``frequencies`` (Hz) in ``signal``, and the residual rms: the independent reading of what a
    kind's signal contains."""
    columns = [f(2 * np.pi * frequency * t) for frequency in frequencies for f in (np.sin, np.cos)]
    design = np.stack(columns, axis=1)
    solution, *_ = np.linalg.lstsq(design, signal, rcond=None)
    amplitudes = np.hypot(solution[0::2], solution[1::2])
    return amplitudes, float(np.sqrt(np.mean((design @ solution - signal) ** 2)))


def assert_derivatives_consistent(trajectory, space) -> None:
    """The velocity the kind reports is the slope of the positions it reports (an independent oracle: finite differences)."""
    t, q, qd = trajectory.t, trajectory.positions, trajectory.velocities
    slope = np.gradient(q, t, axis=0)
    error = np.max(np.abs(slope - qd) / space.limits.max_velocity)
    assert error < 0.02, f"velocity differs from the position's slope by {error:.4f} of the limit"


def room_of(space, centre) -> np.ndarray:
    bounds = space.room(centre)
    return np.minimum(centre - bounds.low, bounds.high - centre)


# ---------------------------------------------------------------------- slow_plus_fast_sines


def test_slow_plus_fast_sines():
    for name, make, joint in SINGLE:
        space = make()
        trajectory = slow_plus_fast_sines(space, joint)
        check_kind(trajectory, space)
        assert_derivatives_consistent(trajectory, space)
        index = space.joint_index(joint)
        start = trajectory.positions[0]
        others = np.delete(trajectory.positions, index, axis=1)
        assert others.size == 0 or np.ptp(others, axis=0).max() == 0.0, f"{name}: a joint that was not asked to move moved"
        # what it is FOR: a slow large sine with a small fast one on top, in the catalog's ratio, the limits solved
        t = trajectory.t
        inside = (t > 1.0) & (t < t[-1] - 1.0)
        amplitudes, residual = fit_lines(t[inside], trajectory.positions[inside, index] - start[index], (0.15, 2.5))
        assert abs(amplitudes[1] / amplitudes[0] - 0.16) < 0.01, (name, amplitudes)
        assert residual < 0.01 * amplitudes[0], f"{name}: more than two sines in it (residual {residual:.2e})"
        bounds = space.room(start)
        ratios = (np.max(np.abs(trajectory.positions[:, index] - start[index])) / room_of(space, start)[index],
                  np.max(np.abs(trajectory.velocities[:, index])) / bounds.max_velocity[index],
                  np.max(np.abs(trajectory.accelerations[:, index])) / bounds.max_acceleration[index])
        assert abs(max(ratios) - 0.8) < 1e-3, f"{name}: the tightest limit is at {max(ratios):.4f}, not the 0.8 asked ({ratios})"


# ---------------------------------------------------------------------- stepped_sine_dwell


def test_stepped_sine_dwell():
    for name, make, joint in SINGLE:
        space = make()
        for rule in ("constant_speed", "constant_angle"):
            trajectory = stepped_sine_dwell(space, joint, f_low_hz=0.5, f_high_hz=4.0, frequencies=5, amplitude_rule=rule)
            check_kind(trajectory, space)
            index = space.joint_index(joint)
            pieces, names = split(trajectory), labels(trajectory)
            dwells = [(piece, label) for piece, label in zip(pieces, names, strict=True) if "rest" not in label]
            assert len(dwells) == 5, (name, names)
            expected = np.geomspace(0.5, 4.0, 5)
            swings = []
            for (piece, _), frequency in zip(dwells, expected, strict=True):
                t, q = piece.t, piece.positions[:, index] - trajectory.positions[0, index]
                flat = (t >= 1.0 / frequency) & (t <= t[-1] - 1.0 / frequency)  # between the fades (one ramp cycle each)
                amplitudes, residual = fit_lines(t[flat], q[flat], (frequency,))
                assert residual < 0.02 * amplitudes[0], f"{name}/{rule}: the dwell at {frequency:.3g} Hz is not a sine of that frequency (residual {residual:.2e})"
                swings.append(float(np.max(np.abs(q))))
            swings = np.array(swings)
            if rule == "constant_angle":
                assert np.ptp(swings) < 0.03 * swings.max(), (name, swings)
            else:
                assert np.all(np.diff(swings) <= 1e-6) and swings[-1] < 0.95 * swings[0], f"{name}: amplitude did not give way as the frequency rose: {swings}"
            rests = [piece for piece, label in zip(pieces, names, strict=True) if "rest" in label]
            assert all(np.ptp(piece.positions, axis=0).max() == 0.0 for piece in rests), "a rest between dwells moves"


# ---------------------------------------------------------------------- orthogonal_multisine


def test_orthogonal_multisine():
    module = importlib.import_module("robot_trajectories.kinds.orthogonal_multisine")
    for name, make in MULTI:
        space = make()
        trajectory = orthogonal_multisine(space, periods=3, velocity_rms_rad_s=0.05)
        check_kind(trajectory, space)
        assert_derivatives_consistent(trajectory, space)
        t, q, qd = trajectory.t, trajectory.positions, trajectory.velocities
        rate = space.rate_hz
        period = int(round(10.0 * rate))
        steady = slice(period, 2 * period)  # one whole period, clear of the fade
        spectra = np.abs(np.fft.rfft(q[steady] - q[steady].mean(axis=0), axis=0)) ** 2  # bin k is the k-th harmonic of 0.1 Hz
        owned = []
        for joint in range(space.dof):
            power = spectra[:, joint]
            own = np.flatnonzero(power > 1e-6 * power.max())
            owned.append(set(own.tolist()))
            assert power[sorted(own)].sum() > 0.999 * power.sum()
            rms = float(np.sqrt(np.mean(qd[steady, joint] ** 2)))
            assert abs(rms - 0.05) < 0.003, f"{name}: joint {joint} speed rms {rms:.4f}, asked 0.05"
        for a in range(space.dof):
            for b in range(a + 1, space.dof):
                assert not owned[a] & owned[b], f"{name}: joints {a} and {b} share harmonics {sorted(owned[a] & owned[b])}"
        assert all(3 <= min(own) and max(own) <= 40 for own in owned), "a line outside the 0.3-4 Hz band"
        # Morelli's phases are never worse than Schroeder's, on the same lines
        lines = np.arange(3, 41)
        rng = np.random.default_rng(0)
        schroeder = module._phases(lines[::2], 10.0, 0, rng)
        searched = module._phases(lines[::2], 10.0, 200, np.random.default_rng(0))
        grid = np.arange(1280) / 128.0

        def peak(phi):
            return module._relative_peak_factor(sum(np.sin(2 * np.pi * h / 10.0 * grid + p) / h for h, p in zip(lines[::2], phi, strict=True)))

        assert peak(searched) <= peak(schroeder) + 1e-12
    # the collision checker, when the space has one: a free scene passes (check_kind holds every sample to it), a scene it cannot avoid refuses
    ur5e = scenario.ur5e_space().with_collision(robots.UR5eScene())
    check_kind(orthogonal_multisine(ur5e, periods=1, velocity_rms_rad_s=0.05), ur5e)
    blocked = scenario.ur5e_space().with_collision(robots.UR5eScene(box_min=(-3.0, -3.0, -3.0), box_max=(3.0, 3.0, 3.0)))
    try:
        orthogonal_multisine(blocked, periods=1)
    except UnsafeTrajectory as refusal:
        assert "collides" in str(refusal), refusal
    else:
        raise AssertionError("a multisine inside an obstacle was not refused")


# ---------------------------------------------------------------------- prbs


def test_prbs_sequence_is_maximal_for_every_register_length():
    helper = importlib.import_module("robot_trajectories.kinds._lane_a")
    for stages in sorted(helper.MAXIMAL_TAPS):
        bits = helper.m_sequence(stages)
        period = 2**stages - 1
        assert len(bits) == period and bits.sum() == 2 ** (stages - 1), (stages, bits.sum())
        # maximal length: the periodic autocorrelation of the +-1 sequence is -1 at every non-zero lag
        signs = 2.0 * bits - 1.0
        spectrum = np.abs(np.fft.fft(signs)) ** 2
        assert np.allclose(spectrum[1:], spectrum[1], rtol=1e-9), f"{stages} stages: the spectrum is not flat"
        assert not np.array_equal(helper.m_sequence(stages, seed=1), bits) or stages == 3
        if stages <= 8:
            assert len({tuple(np.roll(bits, shift)) for shift in range(period)}) == period, f"{stages} stages: not a single cycle"


def test_prbs():
    for name, make, joint in SINGLE:
        space = make()
        clock = 0.25 if name == "pendulum" else 0.1
        trajectory = prbs(space, joint, clock_period_s=clock, register_stages=5, periods=3)
        check_kind(trajectory, space)
        assert_derivatives_consistent(trajectory, space)
        assert any("lowered" in note for note in notes(trajectory)) or name != "pendulum", (name, notes(trajectory))
        index = space.joint_index(joint)
        # what it is FOR: a flat spectrum to about 0.2 / clock (the -3 dB of a held level is 0.44 / clock)
        slot = int(round(clock * space.rate_hz))
        period = 31 * slot
        x = trajectory.positions[period:2 * period, index]
        power = np.abs(np.fft.rfft(x - x.mean())) ** 2
        lines = np.arange(1, int(0.2 * 31) + 1)  # harmonics of 1 / (31 clock) up to 0.2 / clock
        assert power[lines].max() / power[lines].min() < 2.0, f"{name}: lines to 0.2/clock differ by {power[lines].max() / power[lines].min():.2f}x"
        # the asked amplitude when it fits: exactly two levels, +- the asked angle about the start
        small = prbs(space, joint, clock_period_s=clock, register_stages=4, amplitude_rad=0.001, periods=2)
        check_kind(small, space)
        offsets = small.positions[:, index] - small.positions[0, index]
        held = offsets[np.abs(small.velocities[:, index]) < 1e-12]
        assert set(np.round(np.abs(held[np.abs(held) > 1e-9]), 9)) == {0.001}, set(np.round(held, 9))
    # seeded: the same seed is the same signal, another seed starts elsewhere in the period
    space = scenario.pendulum_space()
    a, b, c = (prbs(space, "can11", clock_period_s=0.25, register_stages=4, seed=s) for s in (0, 0, 1))
    assert np.array_equal(a.positions, b.positions) and not np.array_equal(a.positions, c.positions)
    try:
        prbs(space, "can11", register_stages=2)
    except ValueError as refusal:
        assert "register_stages" in str(refusal)
    else:
        raise AssertionError("2 register stages is not a supported length")


# ---------------------------------------------------------------------- band_limited_noise


def test_band_limited_noise():
    for name, make, joint in SINGLE:
        space = make()
        trajectory = band_limited_noise(space, joint, band_low_hz=0.2, band_high_hz=2.0, rms_rad=0.002, duration_s=20.0)
        check_kind(trajectory, space)
        assert_derivatives_consistent(trajectory, space)
        index = space.joint_index(joint)
        x = trajectory.positions[:, index] - trajectory.positions[0, index]
        frequencies = np.fft.rfftfreq(len(x), 1.0 / space.rate_hz)
        power = np.abs(np.fft.rfft(x)) ** 2
        inside = (frequencies >= 0.2 - 0.4) & (frequencies <= 2.0 + 0.4)  # the fade's skirts: 1 s raised cosine
        assert power[inside].sum() > 0.95 * power.sum(), f"{name}: {power[~inside].sum() / power.sum():.3f} of the power is out of band"
        middle = x[int(1.0 * space.rate_hz):-int(1.0 * space.rate_hz)]
        assert abs(np.sqrt(np.mean(middle**2)) / 0.002 - 1.0) < 0.2, f"{name}: rms {np.sqrt(np.mean(middle**2)):.5f} vs 0.002 asked"
        again = band_limited_noise(space, joint, band_low_hz=0.2, band_high_hz=2.0, rms_rad=0.002, duration_s=20.0)
        other = band_limited_noise(space, joint, band_low_hz=0.2, band_high_hz=2.0, rms_rad=0.002, duration_s=20.0, seed=1)
        assert np.array_equal(again.positions, trajectory.positions) and not np.array_equal(other.positions, trajectory.positions)
    # the limits lower a loud ask, and say so
    space = scenario.pendulum_space()
    loud = band_limited_noise(space, "can11", rms_rad=0.05, duration_s=10.0)
    check_kind(loud, space)
    assert any("asked" in note for note in notes(loud)), notes(loud)
    try:
        band_limited_noise(space, "can11", band_high_hz=300.0)
    except ValueError as refusal:
        assert "Nyquist" in str(refusal)
    else:
        raise AssertionError("a band above Nyquist was accepted")


# ---------------------------------------------------------------------- fourier_series_excitation


def _regressor_condition(space, trajectory) -> float:
    """The column-normalised condition number of the stand-in's TRUE regressor along a trajectory (the independent ruler: the kind's
    own number comes from the same function, so this one is recomputed here from the samples)."""
    stack = np.vstack([space.dynamics.regressor(q, qd, qdd) for q, qd, qdd in
                       zip(trajectory.positions[::5], trajectory.velocities[::5], trajectory.accelerations[::5], strict=True)])
    stack = stack / np.linalg.norm(stack, axis=0)
    singular = np.linalg.svd(stack, compute_uv=False)
    return float(singular[0] / singular[-1])


def test_fourier_series_excitation():
    for name, make in MULTI:
        bare = make()
        try:
            fourier_series_excitation(bare)
        except MissingRung as refusal:
            assert "dynamics" in str(refusal), refusal
        else:
            raise AssertionError(f"{name}: an unoptimised series was handed over without dynamics")
        space = bare.with_dynamics(lane_a.ArmDynamics(bare.dof))
        optimised = fourier_series_excitation(space)
        baseline = fourier_series_excitation(space, restarts=1, iterations=0)
        check_kind(optimised, space)
        check_kind(baseline, space)
        assert_derivatives_consistent(optimised, space)
        better, worse = _regressor_condition(space, optimised), _regressor_condition(space, baseline)
        assert better < 0.92 * worse, f"{name}: optimised condition {better:.3f} is not clearly below the random series' {worse:.3f}"
        assert np.array_equal(fourier_series_excitation(space).positions, optimised.positions), "not reproducible from its seed"
        # D-optimality is the other criterion
        d_opt = fourier_series_excitation(space, criterion="d_optimality", restarts=8, iterations=100)
        check_kind(d_opt, space)
        # the same arm with only torque(): the generic basis stands in, and says so
        generic_space = bare.with_dynamics(lane_a.TorqueOnlyDynamics(lane_a.ArmDynamics(bare.dof)))
        generic = fourier_series_excitation(generic_space, restarts=8, iterations=60)
        check_kind(generic, generic_space)
        assert any("generic basis" in note for note in notes(generic)), notes(generic)
    # the torque limit, when the space states one, binds the predicted torque
    five = scenario.fivebar_space()
    limited_limits = five.limits.replace(max_effort=np.array([3.0, 3.0]))
    capped = dataclasses.replace(five, limits=limited_limits).with_dynamics(lane_a.ArmDynamics(2))
    run = fourier_series_excitation(capped, restarts=8, iterations=60)
    check_kind(run, capped)
    assert np.max(np.abs(capped.torques(run))) <= 3.0 * (1 + 1e-6), np.max(np.abs(capped.torques(run)))
    # a scene it cannot avoid is refused rather than handed over
    ur5e = scenario.ur5e_space().with_dynamics(lane_a.ArmDynamics(6)).with_collision(
        robots.UR5eScene(box_min=(-3.0, -3.0, -3.0), box_max=(3.0, 3.0, 3.0)))
    try:
        fourier_series_excitation(ur5e, restarts=2, iterations=0)
    except (ValueError, UnsafeTrajectory) as refusal:
        assert "collid" in str(refusal), refusal
    else:
        raise AssertionError("a series inside an obstacle was not refused")
    free = scenario.ur5e_space().with_dynamics(lane_a.ArmDynamics(6)).with_collision(robots.UR5eScene())
    check_kind(fourier_series_excitation(free, restarts=4, iterations=20), free)


# ---------------------------------------------------------------------- biased_dither


def test_biased_dither():
    for name, make, joint in SINGLE:
        space = make()
        index = space.joint_index(joint)
        for first in (1, -1):
            trajectory = biased_dither(space, joint, first_direction=first)
            check_kind(trajectory, space)
            assert_derivatives_consistent(trajectory, space)
            qd = trajectory.velocities[:, index]
            moving = np.sign(qd[np.abs(qd) > 1e-9])
            reversals = int(np.sum(moving[1:] != moving[:-1]))
            assert reversals == 2, f"{name}: {reversals} reversals (the two turns are the only ones allowed)"
            entry = segment_info(trajectory)[0]
            speed = entry["ramp_speed_rad_s"]
            t = trajectory.t
            for number, (a, b) in enumerate(entry["slides_s"]):
                direction = first * (1.0 if number % 2 == 0 else -1.0)
                inside = (t >= a) & (t <= b)
                slid = direction * qd[inside]
                assert slid.min() >= 0.2 * speed - 1e-9, f"{name}: slides {slid.min() / speed:.3f} of the ramp speed at its slowest"
                assert slid.max() <= 1.8 * speed + 1e-9
                if b - a >= 0.4:  # a slide shorter than 0.4 s carries none (and the kind says so)
                    assert np.ptp(slid) > 0.05 * speed, f"{name}: no ripple on slide {number}"
                else:
                    assert np.ptp(slid) < 1e-9 and any("no ripple" in note for note in notes(trajectory)), (name, number)
            assert entry["slides_s"][1][1] - entry["slides_s"][1][0] > 0.4, f"{name}: the long slide is too short to dither"
            assert np.allclose(trajectory.positions[-1], trajectory.positions[0], atol=1e-9)
            went = trajectory.positions[:, index] - trajectory.positions[0, index]
            assert np.sign(went[np.argmax(np.abs(went) > 0.1 * np.max(np.abs(went)))]) == first, "went to the wrong side first"


# ---------------------------------------------------------------------- staircase


def test_staircase():
    for name, make, joint in SINGLE:
        space = make()
        index = space.joint_index(joint)
        trajectory = staircase(space, joint, levels=7, hold_s=0.5)
        check_kind(trajectory, space)
        t, q, qd = trajectory.t, trajectory.positions[:, index], trajectory.velocities[:, index]
        still = np.abs(qd) < 1e-12
        edges = np.flatnonzero(np.diff(still.astype(int)) != 0) + 1
        runs = np.split(np.arange(len(t)), edges)
        plateaus = [(float(q[run[0]]), (t[run[-1]] - t[run[0]])) for run in runs if still[run[0]] and (t[run[-1]] - t[run[0]]) > 0.4]
        assert len(plateaus) == 12, f"{name}: {len(plateaus)} holds, expected 4 x 3"
        values = np.array([value for value, _ in plateaus]) - q[0]
        step = values.max() / 3
        levels = np.round(values / step).astype(int)
        assert np.allclose(values, levels * step, atol=1e-9), f"{name}: levels are not equally spaced"
        assert all(abs(length - 0.5) < 0.01 for _, length in plateaus), "a hold is not hold_s long"
        arrived = {}
        previous = 0
        for level in levels:
            arrived.setdefault(int(level), []).append("below" if level > previous else "above")
            previous = level
        for level in range(-2, 3):
            assert sorted(arrived[level]) == ["above", "below"], f"{name}: level {level} reached {arrived[level]}"
        assert arrived[3] == ["below"] and arrived[-3] == ["above"]
        assert abs(values.max() - 0.85 * room_of(space, trajectory.positions[0])[index]) < 1e-9
        assert np.allclose(trajectory.positions[-1], trajectory.positions[0], atol=1e-9)
        two = staircase(space, joint, levels=3, hold_s=0.3, cycles=2)
        check_kind(two, space)
        assert len([band for band in bands(two) if "hold" in band.label]) == 2 * 4
    try:
        staircase(scenario.pendulum_space(), "can11", levels=6)
    except ValueError as refusal:
        assert "odd" in str(refusal)
    else:
        raise AssertionError("an even number of levels was accepted")


# ---------------------------------------------------------------------- doublet_3211


def test_doublet_3211():
    for name, make, joint in SINGLE:
        space = make()
        index = space.joint_index(joint)
        unit = 0.2
        for pattern, crossings in (("3-2-1-1", [2 * unit, 1 * unit]), ("doublet", [])):
            trajectory = doublet_3211(space, joint, unit_time_s=unit, amplitude_rad=0.001, pattern=pattern)
            check_kind(trajectory, space)
            assert_derivatives_consistent(trajectory, space)
            t = trajectory.t
            x = trajectory.positions[:, index] - trajectory.positions[0, index]
            held = x[np.abs(trajectory.velocities[:, index]) < 1e-12]
            assert set(np.round(np.abs(held[np.abs(held) > 1e-9]), 9)) == {0.001}, f"{name}: the pulse level is not +-0.001"
            big = np.flatnonzero(np.abs(x) > 1e-4)
            flips = [(a, b) for a, b in zip(big[:-1], big[1:], strict=True) if x[a] * x[b] < 0]
            times = np.array([t[a] + (t[b] - t[a]) * abs(x[a]) / (abs(x[a]) + abs(x[b])) for a, b in flips])  # the zero crossings, interpolated
            edge = 0.5 * unit  # edge_fraction 0.5 of the shortest pulse
            first = {"3-2-1-1": 3 * unit, "doublet": unit}[pattern] + edge / 2
            assert abs(times[0] - first) <= 0.01, (name, pattern, times)
            assert np.allclose(np.diff(times), crossings, atol=0.01), (name, pattern, np.diff(times))
            assert len(times) == 1 + len(crossings)
    # position-limited: asked beyond the room, the amplitude gives way and the note says so
    space = scenario.pendulum_space()
    big = doublet_3211(space, "can11", unit_time_s=2.0, amplitude_rad=2.0)
    check_kind(big, space)
    assert any("lowered" in note for note in notes(big)) and np.max(np.abs(big.positions)) <= 0.33
    try:
        doublet_3211(space, "can11", pattern="3-x")
    except ValueError as refusal:
        assert "pattern" in str(refusal)
    else:
        raise AssertionError("a pattern that is not whole units was accepted")


# ---------------------------------------------------------------------- rapid_replan


def test_rapid_replan():
    for name, make in MULTI:
        space = lane_a.with_named_home(make())
        trajectory = rapid_replan(space)
        check_kind(trajectory, space)
        assert_derivatives_consistent(trajectory, space)
        t, q, qd, qdd = trajectory.t, trajectory.positions, trajectory.velocities, trajectory.accelerations
        entry = segment_info(trajectory)[0]
        targets = np.array(entry["targets"])
        assert len(targets) == 8 and np.max(np.abs(targets - space.home)) <= 0.349 + 1e-9
        # every plan but the last hands over mid-motion: at each 0.15 s the speed is non-zero and nothing jumps
        joins = np.round(np.arange(1, 8) * 0.15 * space.rate_hz).astype(int)
        assert np.min(np.max(np.abs(qd[joins]), axis=1)) > 0.01, f"{name}: a hand-over is at rest"
        assert np.max(np.abs(q[joins] - q[joins - 1]) / (space.limits.max_velocity / space.rate_hz)) < 1.05, "position jumps at a hand-over"
        assert np.max(np.abs(qd[joins] - qd[joins - 1]) / (space.limits.max_acceleration / space.rate_hz)) < 1.05, "speed jumps at a hand-over"
        assert np.allclose(q[-1], targets[-1], atol=1e-9) and np.allclose(qdd[-1], 0.0, atol=1e-9), "does not end at rest on its last target"
        # each plan is a quintic heading for its own target: at the join the next target has not been reached yet
        assert all(np.max(np.abs(q[join] - targets[number])) > 1e-4 for number, join in enumerate(joins))
        again = rapid_replan(space)
        other = rapid_replan(space, seed=1)
        assert np.array_equal(again.positions, trajectory.positions) and not np.array_equal(other.positions, trajectory.positions)
        unnamed = dataclasses.replace(space, named_home=False)  # the same robot, its home not named
        try:
            rapid_replan(unnamed)
        except MissingRung as refusal:
            assert "home" in str(refusal), refusal
        else:
            raise AssertionError(f"{name}: ran without a named home")
        assert np.array_equal(rebuild(recipe(trajectory), space).positions, trajectory.positions)


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_lane_a"))
