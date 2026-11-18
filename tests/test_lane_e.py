"""Lane E's two kinds in the core environment, where no time-law plugin is installed: registered on their catalog rows, callable
as f(space) alone, and locked (a ``MissingRung`` naming the package to install), never broken, even on a space with every rung.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_lane_e.py

Their motion is tested where their backends live, each through its real entry point:
plugins/robot_trajectories_ruckig/tests/test_ruckig.py, plugins/robot_trajectories_ruckig_pro/tests/test_ruckig_pro.py and
plugins/robot_trajectories_toppra/tests/test_toppra.py (each in its own venv).
"""

from __future__ import annotations

import numpy as np

import _runner
import scenario
from fixtures import robots
from robot_trajectories import JointSpace, MissingRung, catalog, jerk_limited_scurve, kinds, time_law, time_optimal_path


def test_registered_on_their_catalog_rows():
    rows = {name: kinds()[name].catalog_id for name in ("jerk_limited_scurve", "time_optimal_path")}
    assert rows == {"jerk_limited_scurve": "jerk_limited_scurve", "time_optimal_path": "time_optimal_path"}, rows
    todo = set(catalog().todo.names)
    assert not todo & set(rows), f"rows still listed as todo: {todo & set(rows)}"
    assert "jerk_limits" in kinds()["jerk_limited_scurve"].needs


def test_locked_without_a_plugin_naming_the_package_even_with_every_rung():
    assert not {"ruckig", "toppra"} & set(time_law.time_laws()), "a plugin is installed in the core venv: this test needs none"
    for space in (JointSpace.from_limits(robots.PENDULUM_LIMITS.replace(max_jerk=np.array([200.0]))),
                  JointSpace.from_limits(scenario.ur5e_space().limits.replace(max_jerk=np.full(6, 100.0)))):
        for call, package in ((lambda: jerk_limited_scurve(space), "robot-trajectories-ruckig"),
                              (lambda: time_optimal_path(space), "robot-trajectories-toppra"),
                              (lambda: time_optimal_path(space, [space.home, space.home]), "robot-trajectories-toppra")):
            try:
                call()
            except MissingRung as error:
                assert package in str(error) and "robot_trajectories.time_laws" in str(error) and "Add it" in str(error), error
            else:
                raise AssertionError(f"ran without {package}")


def test_a_space_without_a_jerk_limit_locks_the_scurve_on_its_rung_first():
    try:
        jerk_limited_scurve(scenario.pendulum_space(), "can11")
    except MissingRung as error:
        assert error.need == "jerk_limits", error
    else:
        raise AssertionError("jerk_limited_scurve ran without a jerk limit")


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_lane_e (lane E's kinds, locked without their plugins)"))
