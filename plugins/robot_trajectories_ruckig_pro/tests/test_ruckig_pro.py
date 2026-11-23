"""The ``ruckig`` time parameterization from the local Ruckig Pro build, a drop-in for the Community plugin, held to
robot_trajectories' TimeParameterizationBackend contract on the pendulum, the five-bar and the UR5e, through its real entry
point. A plain script:

    /home/bam/robot_trajectories/plugins/robot_trajectories_ruckig_pro/.venv/bin/python /home/bam/robot_trajectories/plugins/robot_trajectories_ruckig_pro/tests/test_ruckig_pro.py

Every test skips (never falls back to Community or the cloud) when this machine has no verified Pro build: the
``ruckig-pro`` 0.19.4 distribution installed from ~/bam_ws/src/external/ruckig-pro-v0.19.4-obfuscated-code. The jerk limits are
stand-ins (the fixtures state none). Waypoint passage is checked from the samples alone (each point against the polyline
through the samples), never from the backend's own waypoint times.
"""

from __future__ import annotations

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
from robot_trajectories import BoundaryState, JointSpace, TimeParameterization  # noqa: E402

PROVIDER = "robot-trajectories-ruckig-pro"
COMMUNITY = "robot-trajectories-ruckig"
CHECKOUT = pathlib.Path.home() / "bam_ws" / "src" / "external" / "ruckig-pro-v0.19.4-obfuscated-code"
FIVEBAR_START = np.radians([128.0, -126.0])
MM = 0.001


def _pro_here() -> str:
    """"" when the verified Pro build is installed here, else why not (the tests then skip)."""
    try:
        import robot_trajectories_ruckig_pro as plugin

        plugin.require_local_ruckig_pro()
    except Exception as error:  # noqa: BLE001 - any failure means no verified Pro build on this machine
        return f"no verified local Ruckig Pro build here ({type(error).__name__}: {error})"
    return ""


WHY_SKIP = _pro_here()


def need_pro() -> None:
    if WHY_SKIP:
        _runner.skip(WHY_SKIP)


def pendulum() -> JointSpace:
    return JointSpace.from_limits(robots.PENDULUM_LIMITS.replace(max_jerk=np.array([200.0])))


def fivebar() -> JointSpace:
    return (JointSpace.from_limits(robots.FIVEBAR_LIMITS.replace(max_jerk=np.array([300.0, 300.0])), home=np.radians([100.0, -76.0]))
            .with_reach_map(robots.fivebar_reach_map()))


def ur5e() -> JointSpace:
    return JointSpace.from_limits(scenario.ur5e_space().limits.replace(max_jerk=np.full(6, 100.0)))


def legs(space: JointSpace) -> list[tuple[np.ndarray, np.ndarray]]:
    room, home = space.room(space.home), space.home
    half = np.minimum(home - room.low, room.high - home)
    weights = np.linspace(1.0, 0.4, space.dof)
    return [(home, home + fraction * weights * half) for fraction in (0.8, -0.5, 0.15)]


def waypoints(space: JointSpace, count: int = 7) -> np.ndarray:
    """A curved path of ``count`` points around home, inside the room (every joint on its own sine)."""
    room, home = space.room(space.home), space.home
    half = np.minimum(home - room.low, room.high - home)
    s = np.linspace(0.0, 1.0, count)[:, None]
    phase = np.arange(space.dof)[None, :]
    return home + 0.6 * half * np.sin(np.pi * s * (1 + 0.5 * phase)) * np.where(phase % 2, -1.0, 1.0)


def within(arrays, limits, rate_hz: float) -> None:
    """Inside the position, velocity, acceleration and jerk limits it was handed, on the samples."""
    q, qd, qdd = np.asarray(arrays.q), np.asarray(arrays.qd), np.asarray(arrays.qdd)
    assert not np.any(q < limits.min_position - 1e-9) and not np.any(q > limits.max_position + 1e-9), "leaves its position limits"
    assert np.max(np.abs(qd) / limits.max_velocity) <= 1 + 1e-9, f"velocity {np.max(np.abs(qd) / limits.max_velocity):.6f}x"
    assert np.max(np.abs(qdd) / limits.max_acceleration) <= 1 + 1e-9, f"acceleration {np.max(np.abs(qdd) / limits.max_acceleration):.6f}x"
    jerk = np.abs(np.diff(qdd, axis=0)) * rate_hz
    assert np.max(jerk / limits.max_jerk) <= 1 + 1e-6, f"jerk {np.max(jerk / limits.max_jerk):.6f}x the limit"


def passes_in_order(q: np.ndarray, path: np.ndarray, tolerance: float) -> None:
    """Every path point lies within ``tolerance`` of the polyline through the samples, and the points are met in order."""
    a, b = q[:-1], q[1:]
    chord = b - a
    length = np.maximum(np.sum(chord * chord, axis=1), 1e-30)
    last = 0
    for index, point in enumerate(path):
        along = np.clip(np.sum((point - a) * chord, axis=1) / length, 0.0, 1.0)
        distance = np.linalg.norm(a + along[:, None] * chord - point, axis=1)
        # WHY the first passage at or after the previous point's: a path may cross a later point earlier (a 1-D path does)
        hits = np.flatnonzero(distance[last:] <= tolerance)
        assert len(hits), (f"path point {index} is not passed after point {index - 1} (closest {np.min(distance[last:]):.2e} rad; "
                           f"anywhere {np.min(distance):.2e} rad)")
        last += int(hits[0])


def fresh(script: str, *extra_path: str, env: dict | None = None) -> subprocess.CompletedProcess:
    paths = [*extra_path, str(CORE / "tests")]
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, cwd=CORE,
                          env={**os.environ, **(env or {}), "PYTHONPATH": os.pathsep.join(paths)})


# ---------------------------------------------------------------------- found by its entry point, verified at load


def test_found_through_its_entry_point_in_a_fresh_interpreter_with_its_provenance():
    need_pro()
    done = fresh(
        "from importlib import metadata\n"
        "import robot_trajectories as rt\n"
        "from robot_trajectories import plugins\n"
        "points = [p for p in metadata.entry_points(group='robot_trajectories.time_parameterizations') if p.name == 'ruckig']\n"
        "assert [p.value for p in points] == ['robot_trajectories_ruckig_pro:RuckigProBackend'], points\n"
        "rt.TimeParameterization('ruckig', peak_speed_fraction=1.0)\n"
        f"assert [e.provider for e in rt.time_parameterizations()['ruckig']] == [{PROVIDER!r}], "
        "rt.time_parameterizations()['ruckig']\n"
        "assert plugins.errors() == [], plugins.errors()\n"
        "serving = rt.time_parameterization.backend('ruckig')\n"
        "assert serving.edition == 'pro', serving\n"
        f"assert str(serving.provenance.source_checkout) == {str(CHECKOUT.resolve())!r}, serving.provenance\n"
        "print('found', serving.provenance.version, serving.provenance.module_sha256[:12])\n")
    assert done.returncode == 0 and "found 0.19.4" in done.stdout, done.stderr[-3000:]


def test_refuses_to_load_beside_the_community_distribution():
    need_pro()
    with tempfile.TemporaryDirectory() as site:
        info = pathlib.Path(site) / "ruckig-0.19.4.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: ruckig\nVersion: 0.19.4\n")
        done = fresh("import warnings\nwarnings.simplefilter('ignore')\n"
                     "import robot_trajectories as rt\nfrom robot_trajectories import plugins\nplugins.load()\n"
                     "assert 'ruckig' not in rt.time_parameterizations(), rt.time_parameterizations()\n"
                     "assert any(\"both the 'ruckig' and 'ruckig-pro'\" in e for e in plugins.errors()), plugins.errors()\n"
                     "print('refused')\n", site)
    assert done.returncode == 0 and "refused" in done.stdout, done.stderr[-3000:]


def test_refuses_to_load_a_build_from_another_checkout():
    need_pro()
    with tempfile.TemporaryDirectory() as root:
        elsewhere = pathlib.Path(root) / "ruckig-pro-v0.19.4-obfuscated-code"
        elsewhere.mkdir()
        wrong_name = pathlib.Path(root) / "ruckig-pro-copy"
        wrong_name.mkdir()
        for checkout, words in ((elsewhere, "wrong source"), (wrong_name, "must be a folder named")):
            done = fresh("import warnings\nwarnings.simplefilter('ignore')\n"
                         "import robot_trajectories as rt\nfrom robot_trajectories import plugins\nplugins.load()\n"
                         "assert 'ruckig' not in rt.time_parameterizations(), rt.time_parameterizations()\n"
                         f"assert any({words!r} in e for e in plugins.errors()), plugins.errors()\n"
                         "print('refused')\n", env={"ROBOT_TRAJECTORIES_RUCKIG_PRO_CHECKOUT": str(checkout)})
            assert done.returncode == 0 and "refused" in done.stdout, done.stderr[-3000:]


# ---------------------------------------------------------------------- the contract on three robots


def test_rest_to_rest_legs_inside_their_limits_and_no_longer_than_minimum_jerk():
    need_pro()
    for space in (pendulum(), fivebar(), ur5e()):
        for q0, q1 in legs(space):
            reference = TimeParameterization.minimum_jerk(peak_speed_fraction=1.0).leg(q0, q1, space.limits, space.rate_hz)
            distance = np.abs(q1 - q0)
            # WHY this jerk: the min-jerk leg's own peak jerk, 60 |d| / T^3, makes min-jerk one candidate of Ruckig's search
            jerk = np.where(distance > 0, 60.0 * distance / reference.t[-1] ** 3, np.max(60.0 * distance / reference.t[-1] ** 3))
            limits = space.limits.replace(max_jerk=jerk)
            arrays = TimeParameterization("ruckig", peak_speed_fraction=1.0).leg(q0, q1, limits, space.rate_hz)
            assert np.allclose(arrays.t, np.arange(len(arrays.t)) / space.rate_hz, atol=1e-12), "not on the rate's grid"
            assert np.array_equal(arrays.q[0], q0) and np.array_equal(arrays.q[-1], q1) and not np.any(arrays.qd[[0, -1]])
            assert len(arrays.t) <= len(reference.t), f"{arrays.t[-1]:.3f} s is longer than minimum-jerk {reference.t[-1]:.3f} s"
            within(arrays, limits, space.rate_hz)


def test_waypoint_paths_pass_every_point_in_order_inside_the_limits():
    need_pro()
    law = TimeParameterization("ruckig", peak_speed_fraction=1.0)
    for space in (pendulum(), fivebar(), ur5e()):
        path = waypoints(space)
        timed = law.along(path, space.limits, space.rate_hz)
        arrays = timed.arrays
        assert np.allclose(arrays.t, np.arange(len(arrays.t)) / space.rate_hz, atol=1e-12), "not on the rate's grid"
        assert np.array_equal(arrays.q[0], path[0]) and np.array_equal(arrays.q[-1], path[-1]) and not np.any(arrays.qd[[0, -1]])
        within(arrays, space.limits, space.rate_hz)
        # WHY 2e-5 rad: the polyline through 500 Hz samples is within a dt^2 a / 8 (~1e-5 rad) of the motion itself
        passes_in_order(arrays.q, path, 2e-5)
        progress = timed.progress
        assert progress[0] == 0.0 and progress[-1] == 1.0 and np.all(np.diff(progress) >= 0), "path_progress is not 0 -> 1, monotone"


def test_position_limits_are_enforced_where_community_refuses():
    need_pro()
    space = pendulum()
    # From rest at 0.25 rad, arrive at 0.30 rad moving back at 1.5 rad/s: the time-optimal swing passes the 0.33 rad limit.
    # Community refuses it (it ignores position limits); Pro keeps the limit or refuses by name, never leaves it.
    try:
        arrays = TimeParameterization("ruckig", peak_speed_fraction=1.0).leg(np.full(1, 0.25), np.full(1, 0.30), space.limits,
                                                                             space.rate_hz,
                                                                             end_state=BoundaryState(np.full(1, -1.5), np.zeros(1)))
    except ValueError as error:
        assert "position" in str(error).lower() or "limit" in str(error).lower(), error
    else:
        within(arrays, space.limits, space.rate_hz)
        assert np.array_equal(arrays.qd[-1], [-1.5]) and np.array_equal(arrays.q[-1], [0.30])
    # And through waypoints: a path hugging the upper limit stays under it.
    path = np.array([[0.0], [0.30], [0.32], [0.10]])
    arrays = TimeParameterization("ruckig", peak_speed_fraction=1.0).along(path, space.limits, space.rate_hz).arrays
    within(arrays, space.limits, space.rate_hz)
    passes_in_order(arrays.q, path, 2e-5)


def test_moving_ends_land_exactly_on_the_grid_for_legs_and_waypoint_paths():
    need_pro()
    space = fivebar()
    end = BoundaryState(np.array([0.5, -0.3]), np.zeros(2))
    q0, q1 = legs(space)[1]
    leg = TimeParameterization("ruckig", peak_speed_fraction=1.0).leg(q0, q1, space.limits, space.rate_hz, end_state=end)
    assert np.array_equal(leg.qd[-1], end.velocity) and np.array_equal(leg.q[-1], q1)
    within(leg, space.limits, space.rate_hz)
    backend = rt.time_parameterization.backend("ruckig")
    path = waypoints(space, 5)
    out = backend.time_parameterize(path, space.limits, space.rate_hz, TimeParameterization("ruckig", peak_speed_fraction=1.0),
                                    end_state=end)
    assert np.allclose(out.t, np.arange(len(out.t)) / space.rate_hz, atol=1e-12), "a moving waypoint path is not on the grid"
    assert np.array_equal(out.velocities[-1], end.velocity) and np.array_equal(out.positions[-1], path[-1])
    passes_in_order(out.positions, path, 2e-5)


def test_a_cartesian_line_through_ruckig_pro_tracks_within_a_millimetre():
    need_pro()
    space = JointSpace.from_limits(robots.FIVEBAR_LIMITS.replace(max_jerk=np.array([300.0, 300.0])),
                                   home=np.radians([100.0, -76.0])).with_kinematics(robots.FiveBarKinematics())
    tool = space.forward(FIVEBAR_START)[:3, 3]
    end = tool + (0.03, 0.0, 0.0)
    line = rt.cartesian_path(space, [end], time_parameterization=TimeParameterization("ruckig", peak_speed_fraction=0.5),
                             start=FIVEBAR_START)
    check_kind(line, space)
    reached = space.tool_points(line)
    direction = (end - tool) / np.linalg.norm(end - tool)
    offsets = reached - tool
    off = float(np.max(np.linalg.norm(offsets - np.outer(offsets @ direction, direction), axis=1)))
    assert off <= MM, f"the tool leaves the straight line by {off * 1000:.3f} mm"
    assert np.linalg.norm(reached[-1] - end) <= 1e-6, "the line does not end at its point"


def test_jerk_limited_scurve_kind_on_three_robots():
    need_pro()
    for space, joint in ((pendulum(), "can11"), (fivebar(), None), (ur5e(), None)):
        tape = rt.jerk_limited_scurve(space, joint)
        check_kind(tape, space)
        within(rt.kernels.Arrays(tape.t, tape.positions, tape.velocities, tape.accelerations), space.limits, space.rate_hz)


if __name__ == "__main__":
    if WHY_SKIP:
        print(f"  note  {WHY_SKIP}")
    raise SystemExit(_runner.run(globals(), "test_ruckig_pro (plugin robot-trajectories-ruckig-pro: the local Ruckig Pro build)"))
