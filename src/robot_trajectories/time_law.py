"""The old module name, kept so ``from robot_trajectories.time_law import TimeLaw`` still works:
``waypoint_trajectory_generator.py`` is the module now (formerly TimeLaw, then TimeParameterization; both renamed 2026-11-23).

Every name this module had imports from here as the same object as the new module's (``TimeLaw is WaypointTrajectoryGenerator``);
anything not listed below (``BuiltInBackend``, ``PLUGIN_SHAPES``, ...) is looked up on the new module. Nothing here warns.
"""

from __future__ import annotations

from typing import Any

from . import waypoint_trajectory_generator as _module
from .waypoint_trajectory_generator import (BUILT_IN, WAYPOINT_TRAJECTORY_GENERATOR_GROUP, BackendInfo, BoundaryState, LegTiming,
                                            Timed, WaypointTrajectoryGenerator, WaypointTrajectoryGeneratorBackend, backend,
                                            backend_needs, legs_through, register_waypoint_trajectory_generator,
                                            waypoint_trajectory_generators)

# WHY these deprecated aliases (Zach, 2026-11-23): "time law" was an agent's coinage he never used; it became a time
# parameterization (MoveIt's name for the role), then a waypoint trajectory generator: "these ones are specially
# WaypointTrajectoryGenerators. All TrajectoryGenerators output Trajectories". The old names stay importable, as the same
# objects and silently, because arm_motion_bench and 21 of its frozen dataset files use them. TIME_LAW_GROUP names the group a
# plugin declares now; the old group is still read (plugins.LEGACY_TIME_LAW_GROUP).
TimeLaw = WaypointTrajectoryGenerator
TimeLawBackend = WaypointTrajectoryGeneratorBackend
register_time_law = register_waypoint_trajectory_generator
time_laws = waypoint_trajectory_generators
TIME_LAW_GROUP = WAYPOINT_TRAJECTORY_GENERATOR_GROUP


def __getattr__(name: str) -> Any:
    """The rest of the old module (``BuiltInBackend``, ``PLUGIN_SHAPES``, ``TRAPEZOID_RAMP``, ...): the new module's own."""
    return getattr(_module, name)


__all__ = ["BUILT_IN", "BackendInfo", "BoundaryState", "LegTiming", "TIME_LAW_GROUP", "TimeLaw", "TimeLawBackend", "Timed",
           "backend", "backend_needs", "legs_through", "register_time_law", "time_laws"]
