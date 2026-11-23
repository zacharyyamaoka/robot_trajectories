"""Plugins arrive by entry point and are treated like built-ins: a kind and a time parameterization from tests/fixtures/plugin_site.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_plugins.py

The fixture is an installed package in all but the wheel: ``robot_trajectories_test_plugin/`` beside its
``robot_trajectories_test_plugin-0.1.0.dist-info/`` (METADATA, entry_points.txt), so ``importlib.metadata`` finds it when the
folder is on sys.path, exactly as it finds a pip-installed plugin. Each check runs in a fresh interpreter (no state leaks
into the other tests), and the package itself never names the plugin (tests/test_boundary.py).
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import _runner

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "tests" / "fixtures" / "plugin_site"


def _run(script: str) -> str:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SITE), str(ROOT / "tests")])}
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, cwd=ROOT, env=env)
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


def test_a_plugin_kind_is_listed_tagged_and_guarded_like_a_built_in():
    out = _run(
        "import numpy as np\n"
        "import robot_trajectories as rt\n"
        "from fixtures import robots\n"
        "from kind_contract import check_kind\n"
        "info = rt.kinds()['smoothstep_moves']\n"
        "assert info.plugin == 'robot-trajectories-test-plugin', info.plugin\n"
        "assert 'smoothstep_moves' in rt.catalog(use_case='demo').implemented.names\n"
        "assert rt.smoothstep_moves is info.function, 'exported in the package namespace'\n"
        "space = rt.JointSpace.from_limits(robots.PENDULUM_LIMITS)\n"
        "moves = rt.smoothstep_moves(space, 'can11')\n"
        "check_kind(moves, space)\n"
        "print('kind ok', rt.labels(moves), round(float(moves.t[-1]), 3))\n"
    )
    assert "kind ok" in out, out


def test_a_plugin_time_parameterization_serves_any_path_kind():
    out = _run(
        "import numpy as np\n"
        "import robot_trajectories as rt\n"
        "from fixtures import robots\n"
        "from kind_contract import check_kind\n"
        "laws = rt.time_parameterizations()\n"
        "assert [e.provider for e in laws['smoothstep_test']] == ['robot-trajectories-test-plugin'], laws['smoothstep_test']\n"
        "space = rt.JointSpace.from_limits(robots.PENDULUM_LIMITS)\n"
        "law = rt.TimeParameterization('smoothstep_test', peak_speed_fraction=0.5)\n"
        "tour = rt.joint_waypoints(space, [[0.2], [-0.2]], time_parameterization=law)\n"
        "check_kind(tour, space)\n"
        "assert rt.recipe(tour).count('smoothstep_test') == 1\n"
        "assert np.max(np.abs(tour.velocities)) <= 0.5 * 2.0 * (1 + 1e-9), 'the speed fraction reached the backend'\n"
        "try:\n"
        "    law.leg(np.zeros(1), np.ones(1) * 0.1, space.limits, 500.0, end_state=rt.BoundaryState(np.ones(1), np.zeros(1)))\n"
        "    raise SystemExit('a rest-to-rest backend ended moving')\n"
        "except ValueError as error:\n"
        "    assert 'rest-to-rest' in str(error), error\n"
        "print('time parameterization ok')\n"
    )
    assert "time parameterization ok" in out, out


def test_without_the_plugin_its_time_parameterization_is_refused_naming_the_group():
    done = subprocess.run([sys.executable, "-c",
                           "import robot_trajectories as rt\nrt.TimeParameterization('smoothstep_test', peak_speed_fraction=0.5)"],
                          capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert done.returncode != 0 and "robot_trajectories.time_parameterizations" in done.stderr, done.stderr[-2000:]


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_plugins (entry points: kinds and time parameterizations)"))
