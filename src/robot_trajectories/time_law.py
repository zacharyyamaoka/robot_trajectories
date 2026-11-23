"""The old module name, kept so ``from robot_trajectories.time_law import TimeLaw`` still works: ``time_parameterization.py``
is the module now (formerly TimeLaw, renamed 2026-11-23).

Every name this module had imports from here as the same object as the new module's (``TimeLaw is TimeParameterization``);
anything not listed below (``BuiltInBackend``, ``PLUGIN_SHAPES``, ...) is looked up on the new module. Nothing here warns.
"""

from __future__ import annotations

from typing import Any

from . import time_parameterization as _module
from .time_parameterization import (BUILT_IN, TIME_PARAMETERIZATION_GROUP, BackendInfo, BoundaryState, LegTiming, Timed,
                                    TimeParameterization, TimeParameterizationBackend, backend, backend_needs, legs_through,
                                    register_time_parameterization, time_parameterizations)

# WHY these deprecated aliases (Zach, 2026-11-23): "time law" was an agent's coinage he never used; the stage is trajectory
# generation and the class role a time parameterization, MoveIt's name for it (a path and limits in, a timed trajectory out).
# The old names stay importable, as the same objects and silently, because arm_motion_bench and 21 of its frozen dataset files
# use them. TIME_LAW_GROUP names the group a plugin declares now; the old group is still read (plugins.LEGACY_TIME_LAW_GROUP).
TimeLaw = TimeParameterization
TimeLawBackend = TimeParameterizationBackend
register_time_law = register_time_parameterization
time_laws = time_parameterizations
TIME_LAW_GROUP = TIME_PARAMETERIZATION_GROUP


def __getattr__(name: str) -> Any:
    """The rest of the old module (``BuiltInBackend``, ``PLUGIN_SHAPES``, ``TRAPEZOID_RAMP``, ...): the new module's own."""
    return getattr(_module, name)


__all__ = ["BUILT_IN", "BackendInfo", "BoundaryState", "LegTiming", "TIME_LAW_GROUP", "TimeLaw", "TimeLawBackend", "Timed",
           "backend", "backend_needs", "legs_through", "register_time_law", "time_laws"]
