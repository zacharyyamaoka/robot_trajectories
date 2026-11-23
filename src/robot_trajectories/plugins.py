"""Plugins: kinds and backends other packages provide, found by Python entry points (never imported by name), cut by role.

Zach (2026-11-18): "for simple things you keep the implementation yourself but the other things other repos can implement
them... like ruckig (and if you have the professional version we can support that as a drop in backend)". So the simple kinds
live in ``kinds/*.py`` here, and a package that may import what this leaf may not (ruckig, toppra, a robot's own stack)
registers itself in its own ``pyproject.toml``:

    [project.entry-points."robot_trajectories.kinds"]
    my_kinds = "my_package.kinds"                  # a module whose @kind functions register on import

    [project.entry-points."robot_trajectories.time_parameterizations"]
    ruckig = "robot_trajectories_ruckig:RuckigBackend"   # a TimeParameterizationBackend (an instance, or a no-argument class)

The roles (prior-art note 2026-11-18, section 7: every framework cuts the same seams, and "simple vs complex" is not one):

- ``robot_trajectories.kinds``: kinds, ``f(space, ..., start=None) -> JointTrajectory``, registered with ``@kind``;
- ``robot_trajectories.time_parameterizations``: a path and limits in, a timed trajectory out
  (``time_parameterization.TimeParameterizationBackend``): Ruckig, Ruckig Pro (a drop-in for the same name, never installed
  beside Community), TOPP-RA. Its name before the rename of 2026-11-23, ``robot_trajectories.time_laws``, is still read: a
  plugin built against it still loads, and a name declared in both groups is the new group's;
- ``robot_trajectories.planners`` (reserved, not loaded yet): start, goal and the space's collision check in, joint path
  points out (caelan's motion-planners, OMPL, VAMP); a time parameterization then times the path;
- ``robot_trajectories.optimizers`` (reserved, not loaded yet): path and timing chosen together (Drake, cuRobo, TrajOpt,
  Ruckig Pro's waypoint mode); they register as kinds that need a model.

``import robot_trajectories`` loads every installed plugin once. A plugin kind is then listed, tagged, guarded and recorded
exactly like a built-in one (``kinds()[name].plugin`` names its distribution); a time parameterization serves
``TimeParameterization(name, ...)``. A plugin that fails to load is reported (``errors()`` and a warning), and the rest still
load.

WHY entry points: the standard Python plugin mechanism (``importlib.metadata``, stdlib), so a plugin is an ordinary
installable package and this leaf never names it (tests/test_boundary.py).
"""

from __future__ import annotations

import warnings
from importlib import metadata
from typing import Any

KIND_GROUP = "robot_trajectories.kinds"
TIME_PARAMETERIZATION_GROUP = "robot_trajectories.time_parameterizations"
LEGACY_TIME_LAW_GROUP = "robot_trajectories.time_laws"  # the group's name before 2026-11-23, still read
TIME_LAW_GROUP = TIME_PARAMETERIZATION_GROUP  # deprecated alias (time_law.py says why): the group a plugin declares now
PLANNER_GROUP = "robot_trajectories.planners"  # reserved: the role exists, nothing loads it yet
OPTIMIZER_GROUP = "robot_trajectories.optimizers"  # reserved

_state: dict[str, Any] = {"loaded": False, "errors": [], "loaded_points": set()}


def _provider(point: metadata.EntryPoint) -> str:
    dist = getattr(point, "dist", None)
    return getattr(dist, "name", None) or point.value.split(":")[0].split(".")[0]


def load(*, refresh: bool = False) -> list[str]:
    """Load every installed plugin kind and backend (once; ``refresh=True`` scans again for new ones). Returns the errors."""
    if _state["loaded"] and not refresh:
        return list(_state["errors"])
    _state["loaded"] = True
    from . import registry
    from .time_parameterization import register_time_parameterization

    # WHY time parameterizations first: a plugin kind may build a TimeParameterization of a plugin backend when it is imported
    current = list(metadata.entry_points(group=TIME_PARAMETERIZATION_GROUP))
    # WHY the old group too (the rename of 2026-11-23): a plugin built before it declares robot_trajectories.time_laws and must
    # still load. A name in both groups is the new group's: its old entry is skipped, never registered beside it as a second
    # provider (which backend() would refuse as ambiguous)
    declared = {point.name for point in current}
    legacy = [point for point in metadata.entry_points(group=LEGACY_TIME_LAW_GROUP) if point.name not in declared]
    for group, point in [*((TIME_PARAMETERIZATION_GROUP, point) for point in current),
                         *((LEGACY_TIME_LAW_GROUP, point) for point in legacy)]:
        key = (group, point.name, point.value)
        if key in _state["loaded_points"]:
            continue
        _state["loaded_points"].add(key)
        try:
            loaded = point.load()
            backend = loaded() if isinstance(loaded, type) else loaded
            register_time_parameterization(backend, provider=_provider(point))
        except Exception as error:  # noqa: BLE001
            _report(f"time-parameterization plugin {point.name!r} ({point.value}) failed to load: {type(error).__name__}: "
                    f"{error}")
    for point in metadata.entry_points(group=KIND_GROUP):
        key = (KIND_GROUP, point.name, point.value)
        if key in _state["loaded_points"]:
            continue
        _state["loaded_points"].add(key)
        before = set(registry.kinds())
        try:
            point.load()
        except Exception as error:  # noqa: BLE001 - one broken plugin must not take the package down
            _report(f"kind plugin {point.name!r} ({point.value}) failed to load: {type(error).__name__}: {error}")
            continue
        registry._mark_plugin(sorted(set(registry.kinds()) - before), _provider(point))
    return list(_state["errors"])


def _report(message: str) -> None:
    _state["errors"].append(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def errors() -> list[str]:
    """What failed to load, in words."""
    return list(_state["errors"])


__all__ = ["KIND_GROUP", "LEGACY_TIME_LAW_GROUP", "OPTIMIZER_GROUP", "PLANNER_GROUP", "TIME_LAW_GROUP",
           "TIME_PARAMETERIZATION_GROUP", "errors", "load"]
