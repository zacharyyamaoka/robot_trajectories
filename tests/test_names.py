"""The renames of 2026-11-23: ``TimeLaw`` became ``TimeParameterization``, then ``WaypointTrajectoryGenerator``; every older name
still imports as the same object, every older keyword still works, and every older entry-point group is still read.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_names.py

WHY (Zach, 2026-11-23): "time law" was an agent's coinage he never used, so it became a time parameterization (MoveIt's name).
Then, the same day: "lets give it the name TrajectoryGenerator ... these ones are specially WaypointTrajectoryGenerators. All
TrajectoryGenerators output Trajectories". So ``TrajectoryGenerator`` is the umbrella (anything that outputs a robot_msgs
Trajectory: every kind), and ``WaypointTrajectoryGenerator`` the one kinds hand their legs to (points and joint limits in, a
JointTrajectory out). The older names stay importable, silently, because arm_motion_bench and 21 of its frozen dataset files
use them, and a recipe keeps writing its ``"time_law"`` keys, so the recipes in those datasets and the ones written now read
and rebuild the same.

The plugin discovery checks fake ``importlib.metadata.entry_points`` (``unittest.mock``, so this file also runs as a plain
script without pytest's ``monkeypatch``); ``tests/test_plugins.py`` holds the real thing through an installed-style fixture.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import itertools
import json
import types
import warnings
from importlib import metadata
from unittest import mock

import _runner
import numpy as np
from fixtures import robots
from robot_msgs.trajectory import JointTrajectory, Trajectory

import robot_trajectories as rt
from robot_trajectories import plugins

# the entry-point groups, newest first
NEW_GROUP = "robot_trajectories.waypoint_trajectory_generators"
MIDDLE_GROUP = "robot_trajectories.time_parameterizations"
OLD_GROUP = "robot_trajectories.time_laws"
# every spelling of the two renamed keywords, oldest first: a recipe records the oldest, a signature shows the newest
SPELLINGS = ("time_law", "time_parameterization", "waypoint_generator")
APPROACH_SPELLINGS = ("approach_time_law", "approach_time_parameterization", "approach_waypoint_generator")
WAYPOINTS = [[0.2], [-0.2]]


def _pendulum():
    return rt.JointSpace.from_limits(robots.PENDULUM_LIMITS)


def _arm():
    from fixtures import lane_b

    return lane_b.ur5e_space(dynamics=None)


# ---------------------------------------------------------------------- the names


def test_the_new_names_import_from_the_package_and_from_their_module():
    from robot_trajectories import (TrajectoryGenerator, WaypointTrajectoryGenerator, WaypointTrajectoryGeneratorBackend,
                                    register_waypoint_trajectory_generator, waypoint_trajectory_generators)
    from robot_trajectories import waypoint_trajectory_generator as module
    from robot_trajectories.kinds import speed_ladder as ladder
    from robot_trajectories.trajectory_generator import TrajectoryGenerator as from_its_module

    assert WaypointTrajectoryGenerator is module.WaypointTrajectoryGenerator
    assert WaypointTrajectoryGeneratorBackend is module.WaypointTrajectoryGeneratorBackend
    assert register_waypoint_trajectory_generator is module.register_waypoint_trajectory_generator
    assert waypoint_trajectory_generators is module.waypoint_trajectory_generators
    assert TrajectoryGenerator is from_its_module
    names = {"TrajectoryGenerator", "WaypointTrajectoryGenerator", "WaypointTrajectoryGeneratorBackend",
             "register_waypoint_trajectory_generator", "waypoint_trajectory_generators"}
    assert names <= set(rt.__all__), names - set(rt.__all__)
    assert plugins.WAYPOINT_TRAJECTORY_GENERATOR_GROUP == module.WAYPOINT_TRAJECTORY_GENERATOR_GROUP == NEW_GROUP
    assert {"minimum_jerk", "trapezoid"} <= set(waypoint_trajectory_generators())
    assert callable(ladder.waypoint_trajectory_generators)
    rungs = ladder.waypoint_trajectory_generators((0.25, 0.5))
    assert rungs == (WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.25),
                     WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5)), rungs


def test_the_time_parameterization_names_are_the_same_objects_from_the_same_places():
    from robot_trajectories import (TimeParameterization, TimeParameterizationBackend, register_time_parameterization,
                                    time_parameterizations)
    from robot_trajectories import time_parameterization as middle_module
    from robot_trajectories import waypoint_trajectory_generator as module
    from robot_trajectories.kinds import speed_ladder as ladder
    from robot_trajectories.time_parameterization import TimeParameterization as from_the_middle_module

    assert TimeParameterization is rt.WaypointTrajectoryGenerator and from_the_middle_module is rt.WaypointTrajectoryGenerator
    assert TimeParameterizationBackend is rt.WaypointTrajectoryGeneratorBackend
    assert register_time_parameterization is rt.register_waypoint_trajectory_generator
    assert time_parameterizations is rt.waypoint_trajectory_generators
    assert rt.time_parameterization is middle_module, "rt.time_parameterization is still the (older) module"
    assert middle_module.TimeParameterization is module.WaypointTrajectoryGenerator
    assert middle_module.TimeParameterizationBackend is module.WaypointTrajectoryGeneratorBackend
    assert middle_module.register_time_parameterization is module.register_waypoint_trajectory_generator
    assert middle_module.time_parameterizations is module.waypoint_trajectory_generators
    assert middle_module.TIME_PARAMETERIZATION_GROUP == plugins.TIME_PARAMETERIZATION_GROUP == NEW_GROUP, \
        "the older constant names the group a plugin declares now"
    for name in ("BUILT_IN", "BackendInfo", "BoundaryState", "BuiltInBackend", "LegTiming", "RENAMED_KEYWORDS", "Timed",
                 "backend", "backend_needs", "legs_through", "renamed_keywords"):
        assert getattr(middle_module, name) is getattr(module, name), name
    assert {"TimeParameterization", "TimeParameterizationBackend", "register_time_parameterization",
            "time_parameterizations"} <= set(rt.__all__)
    assert ladder.time_parameterizations is ladder.waypoint_trajectory_generators
    generator = TimeParameterization.minimum_jerk(peak_speed_fraction=0.5)
    assert isinstance(generator, rt.WaypointTrajectoryGenerator)
    assert generator == rt.WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5)


def test_the_old_names_are_the_same_objects_from_the_same_places():
    from robot_trajectories import TimeLaw, TimeLawBackend, register_time_law, time_laws
    from robot_trajectories import time_law as old_module
    from robot_trajectories import waypoint_trajectory_generator as module
    from robot_trajectories.kinds import speed_ladder as ladder
    from robot_trajectories.time_law import TimeLaw as from_the_old_module

    assert TimeLaw is rt.WaypointTrajectoryGenerator and from_the_old_module is rt.WaypointTrajectoryGenerator
    assert TimeLawBackend is rt.WaypointTrajectoryGeneratorBackend
    assert register_time_law is rt.register_waypoint_trajectory_generator
    assert time_laws is rt.waypoint_trajectory_generators
    assert rt.time_law is old_module, "rt.time_law is still the (old) module"
    assert old_module.TimeLaw is module.WaypointTrajectoryGenerator
    assert old_module.TimeLawBackend is module.WaypointTrajectoryGeneratorBackend
    assert old_module.register_time_law is module.register_waypoint_trajectory_generator
    assert old_module.time_laws is module.waypoint_trajectory_generators
    assert old_module.TIME_LAW_GROUP == plugins.TIME_LAW_GROUP == NEW_GROUP, "the old constant names the group a plugin declares now"
    for name in ("BUILT_IN", "BackendInfo", "BoundaryState", "BuiltInBackend", "LegTiming", "Timed", "backend", "backend_needs",
                 "legs_through"):
        assert getattr(old_module, name) is getattr(module, name), name
    assert {"TimeLaw", "TimeLawBackend", "register_time_law", "time_laws"} <= set(rt.__all__)
    assert ladder.time_laws is ladder.waypoint_trajectory_generators
    law = TimeLaw.minimum_jerk(peak_speed_fraction=0.5)
    assert isinstance(law, rt.WaypointTrajectoryGenerator) and law == rt.WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5)


def test_the_old_names_are_silent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_module = importlib.reload(importlib.import_module("robot_trajectories.time_law"))
        middle_module = importlib.reload(importlib.import_module("robot_trajectories.time_parameterization"))
        rt.TimeLaw.minimum_jerk(peak_speed_fraction=0.5)
        rt.TimeParameterization.minimum_jerk(peak_speed_fraction=0.5)
        rt.time_laws()
        rt.time_parameterizations()
        rt.joint_waypoints(_pendulum(), WAYPOINTS, time_law=rt.TimeLaw.trapezoid(peak_speed_fraction=0.5))
        rt.joint_waypoints(_pendulum(), WAYPOINTS, time_parameterization=rt.TimeParameterization.trapezoid(peak_speed_fraction=0.5))
    loud = [str(item.message) for item in caught if issubclass(item.category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning))]
    assert not loud, loud
    assert old_module.TimeLaw is rt.WaypointTrajectoryGenerator, "reloaded, the old module still hands out the same objects"
    assert middle_module.TimeParameterization is rt.WaypointTrajectoryGenerator, "and so does the middle one"


# ---------------------------------------------------------------------- TrajectoryGenerator: the umbrella


def test_every_kind_and_every_waypoint_trajectory_generator_output_is_a_trajectory():
    """Zach: "All TrajectoryGenerators output Trajectories". A kind is one; what a WaypointTrajectoryGenerator makes is one."""
    space = _pendulum()
    generator = rt.WaypointTrajectoryGenerator.trapezoid(peak_speed_fraction=0.5)
    made = {
        "joint_waypoints": rt.joint_waypoints(space, WAYPOINTS, waypoint_generator=generator),
        "move": rt.move(space, waypoint_generator=rt.WaypointTrajectoryGenerator.minimum_jerk(move_s=0.5)),
        "chirp": rt.chirp(space, "can11", duration_s=2.0),
        "speed_ladder": rt.speed_ladder(space, rung_fractions=(0.25, 0.5), repeats=1, held_out=1),
        "hold": rt.hold(space, 0.5),
    }
    for name, trajectory in made.items():
        assert isinstance(rt.kinds()[name].function, rt.TrajectoryGenerator), name
        assert isinstance(trajectory, Trajectory) and isinstance(trajectory, JointTrajectory), (name, type(trajectory))
    bound = functools.partial(rt.joint_waypoints, space, WAYPOINTS, waypoint_generator=generator)
    assert isinstance(bound, rt.TrajectoryGenerator) and isinstance(bound(), Trajectory), "a kind bound to its space is one too"
    assert not isinstance(0.5, rt.TrajectoryGenerator), "the runtime check is not vacuous: it needs a callable"
    # a WaypointTrajectoryGenerator's own output: points and joint limits in (sparse or dense), a JointTrajectory out
    points = np.array([[0.0], [0.2], [-0.1], [0.3]])
    for shape in rt.waypoint_trajectory_generator.BUILT_IN:
        generator = rt.WaypointTrajectoryGenerator(shape, peak_speed_fraction=0.5)
        serving = rt.waypoint_trajectory_generator.backend(shape)
        for path in (points[:2], points):
            out = serving.time_parameterize(path, space.limits, space.rate_hz, generator)
            assert isinstance(out, Trajectory) and isinstance(out, JointTrajectory), (shape, len(path), type(out))
            assert np.allclose(out.positions[[0, -1]], path[[0, -1]]), (shape, len(path))
    assert "Trajectory" in inspect.signature(rt.TrajectoryGenerator.__call__).return_annotation


# ---------------------------------------------------------------------- the entry-point groups


class _Backend:
    """A plugin's backend: the built-in minimum-jerk underneath, under its own name."""

    def __init__(self, name: str):
        self.name = name

    def time_parameterize(self, path_points, limits, rate_hz, law, *, start_state=None, end_state=None):
        built_in = rt.waypoint_trajectory_generator.BuiltInBackend("minimum_jerk")
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


def test_the_groups_are_read_newest_first():
    assert plugins.WAYPOINT_TRAJECTORY_GENERATOR_GROUPS == (NEW_GROUP, MIDDLE_GROUP, OLD_GROUP)
    assert plugins.LEGACY_TIME_PARAMETERIZATION_GROUP == MIDDLE_GROUP and plugins.LEGACY_TIME_LAW_GROUP == OLD_GROUP


def test_a_backend_under_each_older_entry_point_group_is_still_discovered():
    for group, name, dist in ((OLD_GROUP, "names_test_old_group", "old-style-plugin"),
                              (MIDDLE_GROUP, "names_test_middle_group", "middle-style-plugin")):
        errors = _discover(_Point(group, _Backend(name), dist))
        assert not [error for error in errors if name in error], errors
        assert [entry.provider for entry in rt.waypoint_trajectory_generators()[name]] == [dist], group
        space = _pendulum()
        generator = rt.WaypointTrajectoryGenerator(name, peak_speed_fraction=0.5)
        tour = rt.joint_waypoints(space, WAYPOINTS, waypoint_generator=generator)
        assert np.max(np.abs(tour.velocities)) <= 0.5 * float(space.limits.max_velocity[0]) * (1 + 1e-9), "the plugin timed it"


def test_the_newest_group_is_read_and_wins_a_name_clash():
    old, middle, new = _Backend("names_test_clash"), _Backend("names_test_clash"), _Backend("names_test_clash")
    # the older groups listed first: the newest wins by group, not by the order the scan meets them
    _discover(_Point(OLD_GROUP, old, "old-provider"), _Point(MIDDLE_GROUP, middle, "middle-provider"),
              _Point(NEW_GROUP, new, "new-provider"), _Point(NEW_GROUP, _Backend("names_test_new_group"), "new-style-plugin"))
    assert [entry.provider for entry in rt.waypoint_trajectory_generators()["names_test_clash"]] == ["new-provider"]
    assert rt.waypoint_trajectory_generator.backend("names_test_clash") is new
    assert [entry.provider for entry in rt.waypoint_trajectory_generators()["names_test_new_group"]] == ["new-style-plugin"]
    older, newer = _Backend("names_test_older_clash"), _Backend("names_test_older_clash")
    _discover(_Point(OLD_GROUP, older, "old-provider"), _Point(MIDDLE_GROUP, newer, "middle-provider"))
    assert [entry.provider for entry in rt.waypoint_trajectory_generators()["names_test_older_clash"]] == ["middle-provider"]
    assert rt.waypoint_trajectory_generator.backend("names_test_older_clash") is newer


# ---------------------------------------------------------------------- the keywords


def test_every_keyword_spelling_makes_the_same_trajectory_and_the_same_recipe():
    space = _pendulum()
    generator = rt.WaypointTrajectoryGenerator.trapezoid(peak_speed_fraction=0.5)
    made = [rt.joint_waypoints(space, WAYPOINTS, **{spelling: generator}) for spelling in SPELLINGS]
    for other in made[:-1]:
        for field in ("t", "positions", "velocities", "accelerations"):
            assert np.array_equal(getattr(made[-1], field), getattr(other, field)), field
        assert rt.recipe(made[-1]) == rt.recipe(other)
    call = json.loads(rt.recipe(made[-1]))["calls"][0]
    # WHY the oldest key: arm_motion_bench's frozen datasets hold recipes written with it; they must rebuild and compare equal
    assert call["time_law"]["time_law"] == "trapezoid", call
    assert not {"time_parameterization", "waypoint_generator"} & set(call), call
    assert np.array_equal(rt.rebuild(rt.recipe(made[0])).positions, made[-1].positions), "a recipe with the old key rebuilds"
    assert "waypoint_generator=WaypointTrajectoryGenerator.trapezoid(" in rt.calls(made[-1])[0], rt.calls(made[-1])
    moved = [rt.move(space, **{spelling: generator}) for spelling in SPELLINGS]
    for other in moved[:-1]:
        assert np.array_equal(moved[-1].positions, other.positions) and rt.recipe(moved[-1]) == rt.recipe(other)
    from robot_trajectories import registry

    assert registry.RECORDED_AS == {"time_parameterization": "time_law", "waypoint_generator": "time_law",
                                    "approach_time_parameterization": "approach_time_law",
                                    "approach_waypoint_generator": "approach_time_law"}, registry.RECORDED_AS


def test_every_approach_keyword_spelling_makes_the_same_pick_and_place():
    arm = _arm()
    tool = arm.forward(arm.home)[:3, 3]
    approach = rt.WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.4)
    made = [rt.pick_and_place(arm, tool + (0.08, -0.08, -0.12), tool + (-0.08, 0.10, -0.12), lift_m=0.08, **{spelling: approach})
            for spelling in APPROACH_SPELLINGS]
    for other in made[:-1]:
        assert np.array_equal(made[-1].positions, other.positions) and rt.recipe(made[-1]) == rt.recipe(other)
    call = json.loads(rt.recipe(made[-1]))["calls"][0]
    assert call["approach_time_law"]["peak_speed_fraction"] == 0.4 and "time_law" in call, call
    assert not {"approach_time_parameterization", "approach_waypoint_generator", "time_parameterization",
                "waypoint_generator"} & set(call), call


def test_more_than_one_spelling_is_refused():
    generator = rt.WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5)
    space = _pendulum()
    cases = [(spellings, lambda given: rt.joint_waypoints(space, WAYPOINTS, **given)) for spellings in
             (*itertools.combinations(SPELLINGS, 2), SPELLINGS)]
    cases += [(spellings, lambda given: rt.pick_and_place(space, None, None, **given)) for spellings in
              (*itertools.combinations(APPROACH_SPELLINGS, 2), APPROACH_SPELLINGS)]
    for spellings, call in cases:
        try:
            call({spelling: generator for spelling in spellings})
        except TypeError as error:
            assert all(f"{spelling}=" in str(error) for spelling in spellings), (spellings, error)
        else:
            raise AssertionError(f"{spellings} were all accepted at once")


def test_signatures_show_the_newest_keyword():
    older = {*SPELLINGS[:-1], *APPROACH_SPELLINGS[:-1]}
    for name, info in rt.kinds().items():
        if info.plugin is None:
            parameters = inspect.signature(info.function).parameters
            assert not older & set(parameters), name
    assert "waypoint_generator" in inspect.signature(rt.joint_waypoints).parameters
    assert "approach_waypoint_generator" in rt.kinds()["pick_and_place"].parameters


def test_the_shared_helper_keeps_the_signature_and_refuses_more_than_one():
    from robot_trajectories.waypoint_trajectory_generator import renamed_keywords

    @renamed_keywords(time_law="waypoint_generator", time_parameterization="waypoint_generator")
    def timed(path, *, waypoint_generator=None):
        return path, waypoint_generator

    assert list(inspect.signature(timed).parameters) == ["path", "waypoint_generator"]
    assert timed(1, time_law=2) == timed(1, time_parameterization=2) == timed(1, waypoint_generator=2) == (1, 2)
    for spellings in (*itertools.combinations(SPELLINGS, 2), SPELLINGS):
        try:
            timed(1, **{spelling: 2 for spelling in spellings})
        except TypeError as error:
            assert all(f"{spelling}=" in str(error) for spelling in spellings), (spellings, error)
        else:
            raise AssertionError(f"{spellings} were all accepted at once")


def test_a_kind_declaring_an_older_spelling_takes_every_spelling_and_records_the_oldest():
    """A plugin kind written before a rename keeps its parameter's name; it is called and recorded like a shipped kind."""
    from robot_trajectories import registry

    @registry.kind(use_cases=("demo",), family="point_to_point", starts_as="path", dimensions=("1D",))
    def names_test_older_kind(space, *, time_parameterization=rt.WaypointTrajectoryGenerator.minimum_jerk(move_s=0.5),
                              start=None):
        """Two legs timed the older way."""
        return rt.waypoint_trajectory_generator.legs_through(start, [start + 0.1, start], time_parameterization, space.limits,
                                                             space.rate_hz, dwell_s=0.0, label="older")

    try:
        space = _pendulum()
        generator = rt.WaypointTrajectoryGenerator.trapezoid(peak_speed_fraction=0.5)
        made = [names_test_older_kind(space, **{spelling: generator}) for spelling in SPELLINGS]
        for other in made[:-1]:
            assert np.array_equal(made[-1].positions, other.positions) and rt.recipe(made[-1]) == rt.recipe(other)
        call = json.loads(rt.recipe(made[0]))["calls"][0]
        assert call["time_law"]["time_law"] == "trapezoid" and "time_parameterization" not in call, call
        try:
            names_test_older_kind(space, time_parameterization=generator, waypoint_generator=generator)
        except TypeError as error:
            assert "time_parameterization=" in str(error) and "waypoint_generator=" in str(error), error
        else:
            raise AssertionError("both spellings were accepted at once")
    finally:
        registry._REGISTRY.pop("names_test_older_kind", None)  # WHY: one pytest process shares the registry


def test_a_kind_taking_two_spellings_is_refused_at_registration():
    from robot_trajectories import registry

    for pair in (("time_law", "waypoint_generator"), ("time_law", "time_parameterization"),
                 ("approach_time_parameterization", "approach_waypoint_generator")):
        namespace: dict = {"__name__": "test_names_exec"}
        exec(f"def names_test_two_spellings(space, *, {pair[0]}=None, {pair[1]}=None, start=None):\n    return []", namespace)
        try:
            registry.kind(use_cases=("demo",), family="point_to_point", starts_as="path",
                          dimensions=("1D",))(namespace["names_test_two_spellings"])
        except TypeError as error:
            assert all(name in str(error) for name in pair), (pair, error)
        else:
            registry._REGISTRY.pop("names_test_two_spellings", None)
            raise AssertionError(f"a kind taking {pair} was registered")


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_names (the renames of 2026-11-23)"))
