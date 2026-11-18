"""The round's frozen ruler (bam_ws docs/trajectory_generator/check_scenario.py at b5e1f6965), held against this package.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_scenario.py

The gates, unchanged in substance:

- G1 robot_msgs out: every trajectory is a robot_msgs JointTrajectory with velocities and accelerations.
- G2 leaf rule: the package imports only the standard library, numpy and robot_msgs (the round also allowed its shared
  ``kernels`` / ``mini_catalog``; this package vendors the kernels, so a top-level ``kernels`` import is now a violation).
- G3 safe by construction: inside position, velocity and acceleration limits, no jumps, at rest at both ends; the five-bar
  never leaves its reachable set.
- G4 the scenario: catalog query, pendulum session joined and as segments, five-bar from a reach map, UR5e from a
  description, deterministic, and rebuilt from its configuration text alone.

What changed from the round's file: the scenario is ``tests/scenario.py`` imported directly (not a variant folder run in its
own interpreter), the fixture is ``tests/fixtures/robots.py``, and each gate is its own test. The checks inside are the
round's lines.
"""

from __future__ import annotations

import ast
import functools
import json
import pathlib
import sys

import numpy as np

import _runner

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "robot_trajectories"
ALLOWED_ROOTS = {"numpy", "robot_msgs", "robot_trajectories", "__future__"}
FORBIDDEN_ROOTS = {"bam_reach", "bam_robot_descriptions", "sim2real", "bam_rack_twin", "traj_compare", "pinocchio", "mujoco",
                   "bam_traj_gen", "scipy", "robots", "kernels", "mini_catalog", "fixtures", "scenario"}


def check_imports(package: pathlib.Path) -> list[str]:
    problems = []
    files = sorted(package.rglob("*.py"))
    if not files:
        return [f"G2: no package at {package}"]
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root in FORBIDDEN_ROOTS or (root not in ALLOWED_ROOTS and root not in sys.stdlib_module_names):
                    problems.append(f"G2: {path.relative_to(package.parent)} imports {name}")
    return problems


@functools.lru_cache(maxsize=1)
def check_one() -> tuple[str, ...]:
    from robot_msgs.trajectory import JointTrajectory

    import scenario
    from fixtures import robots

    problems = check_imports(PACKAGE)
    first = scenario.run()
    second = scenario.run()

    def fail(message: str) -> None:
        problems.append(message)

    def safe(name: str, trajectory, limits, *, max_acceleration=None) -> None:
        if not isinstance(trajectory, JointTrajectory):
            fail(f"G1: {name} is {type(trajectory).__name__}, not robot_msgs JointTrajectory")
            return
        if trajectory.velocities is None or trajectory.accelerations is None:
            fail(f"G1: {name} has no velocities or accelerations")
            return
        if tuple(trajectory.joint_names) != tuple(limits.joint_names):
            fail(f"G1: {name} joint_names {trajectory.joint_names} != {limits.joint_names}")
            return
        q, qd, qdd, t = trajectory.positions, trajectory.velocities, trajectory.accelerations, trajectory.t
        vmax = np.asarray(limits.max_velocity, float)
        amax = np.asarray(limits.max_acceleration if limits.max_acceleration is not None else max_acceleration, float)
        if np.any(q < limits.min_position - 1e-9) or np.any(q > limits.max_position + 1e-9):
            fail(f"G3: {name} leaves its position limits")
        if np.any(np.abs(qd) > vmax * (1 + 1e-6)):
            fail(f"G3: {name} exceeds velocity limits (peak {np.max(np.abs(qd) / vmax):.3f} x)")
        if np.any(np.abs(qdd) > amax * (1 + 1e-6)):
            fail(f"G3: {name} exceeds acceleration limits (peak {np.max(np.abs(qdd) / amax):.3f} x)")
        dt = np.diff(t)[:, None]
        if np.any(np.abs(np.diff(q, axis=0)) > vmax * dt * 1.05 + 1e-9):
            fail(f"G3: {name} jumps in position between samples")
        if np.any(np.abs(np.diff(qd, axis=0)) > amax * dt * 1.05 + 1e-9):
            fail(f"G3: {name} jumps in velocity between samples")
        if np.max(np.abs(qd[0])) > 1e-6 or np.max(np.abs(qd[-1])) > 1e-6:
            fail(f"G3: {name} does not start and end at rest")

    for key in ("pendulum", "fivebar", "ur5e"):
        if key not in first or "joined" not in first[key]:
            fail(f"G4: run() has no {key}['joined']")
            return tuple(problems)

    pendulum = first["pendulum"]
    safe("pendulum joined", pendulum["joined"], robots.PENDULUM_LIMITS)
    safe("five-bar joined", first["fivebar"]["joined"], robots.FIVEBAR_LIMITS)
    safe("UR5e joined", first["ur5e"]["joined"], robots.UR5E_DEMO_BOX)

    # G3: the five-bar never leaves its reachable set (fold margin 0.2 against the map's 0.35)
    off = [q for q in first["fivebar"]["joined"].positions[::5] if (p := robots.fivebar_passive(*q)) is None or p[2] < 0.2]
    if off:
        fail(f"G3: five-bar leaves the reachable set at {len(off)} checked samples")

    # G4 step 1: the catalog query
    pid = set(first.get("catalog_pid", ()))
    if not {"step_train", "chirp"} <= pid or "workspace_sweep" in pid:
        fail(f"G4: catalog query for pid_tuning returned {sorted(pid)}")

    # G4 step 2: the pendulum session as segments
    segments, labels = pendulum.get("segments", []), pendulum.get("labels", [])
    if len(segments) < 5 or len(labels) != len(segments):
        fail(f"G4: pendulum has {len(segments)} segments and {len(labels)} labels")
    else:
        for index, segment in enumerate(segments):
            safe(f"pendulum segment {index} ({labels[index]})", segment, robots.PENDULUM_LIMITS)
        text = " ".join(labels).lower()
        for word in ("step", "chirp", "sweep", "waypoint"):
            if word not in text:
                fail(f"G4: no pendulum segment label mentions {word!r}")
        chirps = [s for s, label in zip(segments, labels) if "chirp" in label.lower()]
        if not chirps or max(s.t[-1] - s.t[0] for s in chirps) < 15.0:
            fail("G4: the pendulum session has no chirp segment of 15 s or more")
        stitched = np.concatenate([s.positions if i == 0 else s.positions[1:] for i, s in enumerate(segments)])
        joined = pendulum["joined"].positions
        if stitched.shape != joined.shape or not np.allclose(stitched, joined, atol=1e-9):
            fail("G4: the pendulum segments, end to end, are not the joined trajectory")
    if pendulum["joined"].t[-1] < 45.0:
        fail(f"G4: the pendulum session lasts {pendulum['joined'].t[-1]:.1f} s (< 45 s)")

    # G4 step 3: the five-bar workspace sweep covers its reach map
    reach = robots.fivebar_reach_map().reachable_configurations()
    span = np.ptp(first["fivebar"]["joined"].positions, axis=0) / np.ptp(reach, axis=0)
    if np.any(span < 0.5) or first["fivebar"]["joined"].t[-1] < 20.0:
        fail(f"G4: five-bar covers {np.round(span, 2)} of its reachable span in {first['fivebar']['joined'].t[-1]:.1f} s")

    # G4 step 4: every UR5e joint moves
    if np.any(np.std(first["ur5e"]["joined"].positions, axis=0) < 0.02):
        fail("G4: not every UR5e joint moves")

    # G4: deterministic, and rebuilt from the configuration text alone
    for key in ("pendulum", "fivebar", "ur5e"):
        a, b = first[key]["joined"], second[key]["joined"]
        if not (np.array_equal(a.t, b.t) and np.array_equal(a.positions, b.positions)):
            fail(f"G4: {key} differs between two runs")
    try:
        json.loads(first["config"])
        rebuilt = scenario.rebuild(first["config"])
        if not (np.array_equal(rebuilt.t, pendulum["joined"].t) and np.array_equal(rebuilt.positions, pendulum["joined"].positions)):
            fail("G4: rebuild(config) is not the pendulum session")
    except Exception as error:  # noqa: BLE001 - any failure to rebuild is the finding
        fail(f"G4: config round trip failed: {type(error).__name__}: {error}")
    return tuple(problems)


def _gate(prefix: str) -> list[str]:
    return [problem for problem in check_one() if problem.startswith(prefix)]


def test_g1_robot_msgs_out():
    assert not _gate("G1:"), _gate("G1:")


def test_g2_leaf_rule():
    assert not _gate("G2:"), _gate("G2:")


def test_g3_safe_by_construction():
    assert not _gate("G3:"), _gate("G3:")


def test_g4_the_scenario():
    assert not _gate("G4:"), _gate("G4:")


def test_g2_goes_red_on_a_planted_import():
    """The ruler must be able to fail: a package file that imports the round's shared kernels by name is caught."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        planted = pathlib.Path(folder) / "robot_trajectories"
        planted.mkdir()
        (planted / "ok.py").write_text("from . import kernels as k\nimport numpy\nfrom robot_msgs.trajectory import JointLimits\n")
        (planted / "bad.py").write_text("import kernels\nfrom scipy import signal\n")
        assert check_imports(planted) == ["G2: robot_trajectories/bad.py imports kernels",
                                          "G2: robot_trajectories/bad.py imports scipy"], check_imports(planted)


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_scenario (the round's ruler, G1-G4)"))
