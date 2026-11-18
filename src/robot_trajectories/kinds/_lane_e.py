"""Lane E's private helper (KINDS_TODO "Rules of the road"): a kind whose time law comes from a plugin is locked, not broken,
when no plugin serves that law. Imported by ``jerk_limited_scurve`` and ``time_optimal_path``; registers nothing."""

from __future__ import annotations

from typing import Any

from .. import plugins
from ..space import MissingRung
from ..time_law import TIME_LAW_GROUP, TimeLaw, time_laws


class MissingPlugin(MissingRung):
    """The kind's time law has no backend here: a locked kind (a ``MissingRung``, so callers and the catalog treat it as one)
    whose fix is a package to install, not a rung of the space. ``need`` is None: the law is beside the needs vocabulary."""

    def __init__(self, who: str, law: str, package: str):
        self.who, self.need, self.rung = who, None, None
        self.fix = f"install {package} (it registers the time law {law!r} in the {TIME_LAW_GROUP!r} entry-point group)"
        failed = [error for error in plugins.errors() if repr(law) in error]
        detail = f"; a plugin for it failed to load: {failed[0]}" if failed else ""
        ValueError.__init__(self, f"{who} needs the time law {law!r} (beside the ladder: a plugin); no installed plugin provides "
                                  f"it{detail}. Add it: {self.fix}")


def plugin_law(who: str, shape: str, package: str, **law: Any) -> TimeLaw:
    """``TimeLaw(shape, **law)``, or ``MissingPlugin`` naming ``package`` when nothing installed serves ``shape``.

    WHY built inside the kind's body, never as a default argument: a default is evaluated at import, before plugins load."""
    plugins.load()
    if shape not in time_laws():
        raise MissingPlugin(who, shape, package)
    return TimeLaw(shape, **law)


__all__ = ["MissingPlugin", "plugin_law"]
