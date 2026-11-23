"""Lane E's private helper (KINDS_TODO "Rules of the road"): a kind whose time parameterization comes from a plugin is locked,
not broken, when no plugin serves that law. Imported by ``jerk_limited_scurve`` and ``time_optimal_path``; registers nothing."""

from __future__ import annotations

from typing import Any

from .. import plugins
from ..space import MissingRung
from ..time_parameterization import TIME_PARAMETERIZATION_GROUP, TimeParameterization, time_parameterizations


class MissingPlugin(MissingRung):
    """The kind's time parameterization has no backend here: a locked kind (a ``MissingRung``, so callers and the catalog
    treat it as one) whose fix is a package to install, not a rung of the space. ``need`` is None: the law is beside the needs
    vocabulary."""

    def __init__(self, who: str, law: str, package: str):
        self.who, self.need, self.rung = who, None, None
        self.fix = (f"install {package} (it registers the time parameterization {law!r} in the {TIME_PARAMETERIZATION_GROUP!r} "
                    "entry-point group)")
        failed = [error for error in plugins.errors() if repr(law) in error]
        detail = f"; a plugin for it failed to load: {failed[0]}" if failed else ""
        ValueError.__init__(self, f"{who} needs the time parameterization {law!r} (beside the ladder: a plugin); no installed "
                                  f"plugin provides it{detail}. Add it: {self.fix}")


def plugin_law(who: str, shape: str, package: str, **law: Any) -> TimeParameterization:
    """``TimeParameterization(shape, **law)``, or ``MissingPlugin`` naming ``package`` when nothing installed serves ``shape``.

    WHY built inside the kind's body, never as a default argument: a default is evaluated at import, before plugins load."""
    plugins.load()
    if shape not in time_parameterizations():
        raise MissingPlugin(who, shape, package)
    return TimeParameterization(shape, **law)


__all__ = ["MissingPlugin", "plugin_law"]
