"""Lane E's private helper (KINDS_TODO "Rules of the road"): a kind whose waypoint trajectory generator comes from a plugin is
locked, not broken, when no plugin serves that law. Imported by ``jerk_limited_scurve`` and ``time_optimal_path``; registers nothing."""

from __future__ import annotations

from typing import Any

from .. import plugins
from ..space import MissingRung
from ..waypoint_trajectory_generator import (WAYPOINT_TRAJECTORY_GENERATOR_GROUP, WaypointTrajectoryGenerator,
                                             waypoint_trajectory_generators)


class MissingPlugin(MissingRung):
    """The kind's waypoint trajectory generator has no backend here: a locked kind (a ``MissingRung``, so callers and the catalog
    treat it as one) whose fix is a package to install, not a rung of the space. ``need`` is None: the law is beside the needs
    vocabulary."""

    def __init__(self, who: str, law: str, package: str):
        self.who, self.need, self.rung = who, None, None
        self.fix = (f"install {package} (it registers the waypoint trajectory generator {law!r} in the "
                    f"{WAYPOINT_TRAJECTORY_GENERATOR_GROUP!r} entry-point group)")
        failed = [error for error in plugins.errors() if repr(law) in error]
        detail = f"; a plugin for it failed to load: {failed[0]}" if failed else ""
        ValueError.__init__(self, f"{who} needs the waypoint trajectory generator {law!r} (beside the ladder: a plugin); no "
                                  f"installed plugin provides it{detail}. Add it: {self.fix}")


def plugin_law(who: str, shape: str, package: str, **law: Any) -> WaypointTrajectoryGenerator:
    """``WaypointTrajectoryGenerator(shape, **law)``, or ``MissingPlugin`` naming ``package`` when nothing installed serves
    ``shape``.

    WHY built inside the kind's body, never as a default argument: a default is evaluated at import, before plugins load."""
    plugins.load()
    if shape not in waypoint_trajectory_generators():
        raise MissingPlugin(who, shape, package)
    return WaypointTrajectoryGenerator(shape, **law)


__all__ = ["MissingPlugin", "plugin_law"]
