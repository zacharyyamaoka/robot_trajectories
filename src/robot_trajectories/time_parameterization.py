"""The module's middle name, kept so ``from robot_trajectories.time_parameterization import TimeParameterization`` still works:
``waypoint_trajectory_generator.py`` is the module now (formerly ``time_law.py``, then this one; both renamed 2026-11-23).

Every name this module had imports from here as the same object as the new module's
(``TimeParameterization is WaypointTrajectoryGenerator``); anything not listed below (``BuiltInBackend``, ``renamed_keywords``,
``PLUGIN_SHAPES``, ...) is looked up on the new module. Nothing here warns.
"""

from __future__ import annotations

from typing import Any

from . import waypoint_trajectory_generator as _module
from .waypoint_trajectory_generator import (BUILT_IN, RENAMED_KEYWORDS, WAYPOINT_TRAJECTORY_GENERATOR_GROUP, BackendInfo,
                                            BoundaryState, LegTiming, Timed, WaypointTrajectoryGenerator,
                                            WaypointTrajectoryGeneratorBackend, backend, backend_needs, legs_through,
                                            register_waypoint_trajectory_generator, renamed_keywords,
                                            waypoint_trajectory_generators)

# WHY these deprecated aliases (Zach, 2026-11-23): "lets give it the name TrajectoryGenerator ... these ones are specially
# WaypointTrajectoryGenerators. All TrajectoryGenerators output Trajectories". The names of the hours between the two renames
# stay importable, as the same objects and silently, like time_law.py's. TIME_PARAMETERIZATION_GROUP names the group a plugin
# declares now; the group it used to name is still read (plugins.LEGACY_TIME_PARAMETERIZATION_GROUP).
TimeParameterization = WaypointTrajectoryGenerator
TimeParameterizationBackend = WaypointTrajectoryGeneratorBackend
register_time_parameterization = register_waypoint_trajectory_generator
time_parameterizations = waypoint_trajectory_generators
TIME_PARAMETERIZATION_GROUP = WAYPOINT_TRAJECTORY_GENERATOR_GROUP


def __getattr__(name: str) -> Any:
    """The rest of the module (``BuiltInBackend``, ``PLUGIN_SHAPES``, ``TRAPEZOID_RAMP``, ...): the new module's own."""
    return getattr(_module, name)


__all__ = ["BUILT_IN", "RENAMED_KEYWORDS", "TIME_PARAMETERIZATION_GROUP", "BackendInfo", "BoundaryState", "LegTiming", "Timed",
           "TimeParameterization", "TimeParameterizationBackend", "backend", "backend_needs", "legs_through",
           "register_time_parameterization", "renamed_keywords", "time_parameterizations"]
