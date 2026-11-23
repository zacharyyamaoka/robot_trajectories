"""The rename of 2026-11-23: ``TimeLaw`` is ``TimeParameterization`` now, and every old name still imports as the same object.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_time_parameterization_names.py

WHY (Zach, 2026-11-23): "time law" was an agent's coinage he never used. The stage is trajectory generation; the class role is
a time parameterization, MoveIt's name for the same role (a path and limits in, a timed trajectory out). The old names stay
importable, silently, because arm_motion_bench and 21 of its frozen dataset files use them, and a recipe keeps writing its
``"time_law"`` keys, so the recipes in those datasets and the ones written now read and rebuild the same.

The plugin discovery checks fake ``importlib.metadata.entry_points`` (``unittest.mock``, so this file also runs as a plain
script without pytest's ``monkeypatch``); ``tests/test_plugins.py`` holds the real thing through an installed-style fixture.
"""

from __future__ import annotations

import importlib
import inspect
import json
import types
import warnings
from importlib import metadata
from unittest import mock

import _runner
import numpy as np
from fixtures import robots

import robot_trajectories as rt
from robot_trajectories import plugins

NEW_GROUP, OLD_GROUP = "robot_trajectories.time_parameterizations", "robot_trajectories.time_laws"
WAYPOINTS = [[0.2], [-0.2]]


def _pendulum():
    return rt.JointSpace.from_limits(robots.PENDULUM_LIMITS)


# ---------------------------------------------------------------------- the names


def test_the_new_names_import_from_the_package_and_from_their_module():
    from robot_trajectories import (TimeParameterization, TimeParameterizationBackend, register_time_parameterization,
                                    time_parameterizations)
    from robot_trajectories import time_parameterization as module

    assert TimeParameterization is module.TimeParameterization
    assert TimeParameterizationBackend is module.TimeParameterizationBackend
    assert register_time_parameterization is module.register_time_parameterization
    assert time_parameterizations is module.time_parameterizations
    names = {"TimeParameterization", "TimeParameterizationBackend", "register_time_parameterization", "time_parameterizations"}
    assert names <= set(rt.__all__), names - set(rt.__all__)
    assert plugins.TIME_PARAMETERIZATION_GROUP == module.TIME_PARAMETERIZATION_GROUP == NEW_GROUP
    assert {"minimum_jerk", "trapezoid"} <= set(time_parameterizations())


def test_the_old_names_are_the_same_objects_from_the_same_places():
    from robot_trajectories import TimeLaw, TimeLawBackend, register_time_law, time_laws
    from robot_trajectories import time_law as old_module
    from robot_trajectories import time_parameterization as module
    from robot_trajectories.kinds import speed_ladder as ladder
    from robot_trajectories.time_law import TimeLaw as from_the_old_module

    assert TimeLaw is rt.TimeParameterization and from_the_old_module is rt.TimeParameterization
    assert TimeLawBackend is rt.TimeParameterizationBackend
    assert register_time_law is rt.register_time_parameterization
    assert time_laws is rt.time_parameterizations
    assert rt.time_law is old_module, "rt.time_law is still the (old) module"
    assert old_module.TimeLaw is module.TimeParameterization and old_module.TimeLawBackend is module.TimeParameterizationBackend
    assert old_module.register_time_law is module.register_time_parameterization and old_module.time_laws is module.time_parameterizations
    assert old_module.TIME_LAW_GROUP == plugins.TIME_LAW_GROUP == NEW_GROUP, "the old constant names the group a plugin declares now"
    for name in ("BUILT_IN", "BackendInfo", "BoundaryState", "BuiltInBackend", "LegTiming", "Timed", "backend", "backend_needs",
                 "legs_through"):
        assert getattr(old_module, name) is getattr(module, name), name
    assert {"TimeLaw", "TimeLawBackend", "register_time_law", "time_laws"} <= set(rt.__all__)
    assert ladder.time_laws is ladder.time_parameterizations
    law = TimeLaw.minimum_jerk(peak_speed_fraction=0.5)
    assert isinstance(law, rt.TimeParameterization) and law == rt.TimeParameterization.minimum_jerk(peak_speed_fraction=0.5)


def test_the_old_names_are_silent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_module = importlib.reload(importlib.import_module("robot_trajectories.time_law"))
        rt.TimeLaw.minimum_jerk(peak_speed_fraction=0.5)
        rt.time_laws()
        rt.joint_waypoints(_pendulum(), WAYPOINTS, time_law=rt.TimeLaw.trapezoid(peak_speed_fraction=0.5))
    loud = [str(item.message) for item in caught if issubclass(item.category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning))]
    assert not loud, loud
    assert old_module.TimeLaw is rt.TimeParameterization, "reloaded, the old module still hands out the same objects"


# ---------------------------------------------------------------------- the entry-point groups


class _Backend:
    """A plugin's backend: the built-in minimum-jerk underneath, under its own name."""

    def __init__(self, name: str):
        self.name = name

    def time_parameterize(self, path_points, limits, rate_hz, law, *, start_state=None, end_state=None):
        built_in = rt.time_parameterization.BuiltInBackend("minimum_jerk")
        return built_in.time_parameterize(path_points, limits, rate_hz, law, start_state=start_state, end_state=end_state)


class _Point:
    """An installed entry point as ``importlib.metadata`` hands it over: name, value, group, its distribution, ``load()``."""

    def __init__(self, group: str, backend: _Backend, dist: str):
        self.group, self.name, self.value = group, backend.name, f"{dist.replace('-', '_')}:{backend.name}"
        self.dist = types.SimpleNamespace(name=dist)
        self._backend = backend

    def load(self):
        return self._backend


def _discover(*points: _Point) -> list[str]:
    """``plugins.load`` scanning again, with these entry points the only ones installed."""

    def entry_points(**selection):
        return [point for point in points if point.group == selection.get("group")]

    with mock.patch.object(metadata, "entry_points", entry_points):
        return plugins.load(refresh=True)


def test_a_backend_under_the_old_entry_point_group_is_still_discovered():
    errors = _discover(_Point(OLD_GROUP, _Backend("names_test_old_group"), "old-style-plugin"))
    assert not [error for error in errors if "names_test_old_group" in error], errors
    assert [entry.provider for entry in rt.time_parameterizations()["names_test_old_group"]] == ["old-style-plugin"]
    space = _pendulum()
    law = rt.TimeParameterization("names_test_old_group", peak_speed_fraction=0.5)
    tour = rt.joint_waypoints(space, WAYPOINTS, time_parameterization=law)
    assert np.max(np.abs(tour.velocities)) <= 0.5 * float(space.limits.max_velocity[0]) * (1 + 1e-9), "the plugin timed it"


def test_the_new_group_is_read_and_wins_a_name_clash_with_the_old():
    old, new = _Backend("names_test_clash"), _Backend("names_test_clash")
    # the old group listed first: the new one wins by group, not by the order the scan meets them
    _discover(_Point(OLD_GROUP, old, "old-provider"), _Point(NEW_GROUP, new, "new-provider"),
              _Point(NEW_GROUP, _Backend("names_test_new_group"), "new-style-plugin"))
    assert [entry.provider for entry in rt.time_parameterizations()["names_test_clash"]] == ["new-provider"]
    assert rt.time_parameterization.backend("names_test_clash") is new
    assert [entry.provider for entry in rt.time_parameterizations()["names_test_new_group"]] == ["new-style-plugin"]


# ---------------------------------------------------------------------- the keyword


def test_the_old_keyword_makes_the_same_trajectory_and_the_same_recipe():
    space = _pendulum()
    law = rt.TimeParameterization.trapezoid(peak_speed_fraction=0.5)
    new = rt.joint_waypoints(space, WAYPOINTS, time_parameterization=law)
    old = rt.joint_waypoints(space, WAYPOINTS, time_law=law)
    for field in ("t", "positions", "velocities", "accelerations"):
        assert np.array_equal(getattr(new, field), getattr(old, field)), field
    assert rt.recipe(new) == rt.recipe(old)
    call = json.loads(rt.recipe(new))["calls"][0]
    # WHY the old key: arm_motion_bench's frozen datasets hold recipes written with it; they must rebuild and compare equal
    assert call["time_law"]["time_law"] == "trapezoid" and "time_parameterization" not in call, call
    assert np.array_equal(rt.rebuild(rt.recipe(old)).positions, new.positions), "a recipe with the old key rebuilds"
    assert "time_parameterization=TimeParameterization.trapezoid(" in rt.calls(new)[0], rt.calls(new)
    moved = rt.move(space, time_law=law), rt.move(space, time_parameterization=law)
    assert np.array_equal(moved[0].positions, moved[1].positions) and rt.recipe(moved[0]) == rt.recipe(moved[1])


def test_both_keywords_at_once_are_refused():
    law = rt.TimeParameterization.minimum_jerk(peak_speed_fraction=0.5)
    space = _pendulum()
    for call in (lambda: rt.joint_waypoints(space, WAYPOINTS, time_law=law, time_parameterization=law),
                 lambda: rt.pick_and_place(space, None, None, approach_time_law=law, approach_time_parameterization=law)):
        try:
            call()
        except TypeError as error:
            assert "time_parameterization=" in str(error) and "time_law=" in str(error), error
        else:
            raise AssertionError("both the old and the new keyword were accepted")


def test_signatures_show_the_new_keyword():
    for name, info in rt.kinds().items():
        if info.plugin is None:
            parameters = inspect.signature(info.function).parameters
            assert "time_law" not in parameters and "approach_time_law" not in parameters, name
    assert "time_parameterization" in inspect.signature(rt.joint_waypoints).parameters
    assert "approach_time_parameterization" in rt.kinds()["pick_and_place"].parameters


def test_the_shared_helper_keeps_the_signature_and_refuses_both():
    from robot_trajectories.time_parameterization import renamed_keywords

    @renamed_keywords(time_law="time_parameterization")
    def timed(path, *, time_parameterization=None):
        return path, time_parameterization

    assert list(inspect.signature(timed).parameters) == ["path", "time_parameterization"]
    assert timed(1, time_law=2) == timed(1, time_parameterization=2) == (1, 2)
    try:
        timed(1, time_law=2, time_parameterization=3)
    except TypeError as error:
        assert "time_parameterization=" in str(error) and "time_law=" in str(error), error
    else:
        raise AssertionError("both keywords were accepted")


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_time_parameterization_names (the rename of 2026-11-23)"))
