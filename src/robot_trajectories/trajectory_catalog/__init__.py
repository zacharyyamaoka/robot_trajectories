"""The trajectory catalog: every kind of trajectory worth supporting (61 rows), one Python record each.

    from robot_trajectories.trajectory_catalog import CATALOG, KINDS, USE_CASES, rows
    KINDS["linear_chirp"].entry_level      # 'joint_limits': passing joint limits is enough to generate it
    rows()                                 # catalog.json's kinds, as dicts (what generator.kinds(catalog_rows=...) grades)

Two lists, on purpose: this catalog says what is worth supporting (with its use cases, needs, sources and prior art);
``robot_trajectories.catalog()`` lists the record classes that generate today. A record claims its row with ``catalog_id``,
and both read one vocabulary (``kind.py`` here).

WHY Python records and catalog.json generated (Zach's rule, vault memory feedback_python_ground_truth): ``catalog.py`` is the
ground truth; ``make_json.py`` writes the JSON and ``tests/test_catalog.py`` fails when it is stale. WHY this subpackage is not
named ``catalog``: importing a submodule ``robot_trajectories.catalog`` would rebind the package attribute ``catalog`` and shadow
the ``catalog()`` function every caller uses.

Stdlib only, so the catalog loads anywhere (a cloud VM, a page builder) without numpy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .kind import (DEFINED_IN, DIMENSIONS, DIMENSIONS_OF_SCOPE, ENTRY_EXTRAS, ENTRY_LEVEL_OF_NEED, ENTRY_LEVELS, FAMILIES,
                   JOINS, NEEDS, OUTPUTS, PRIORITIES, SCOPES, SPACES, STARTS_AS, USE_CASE_TAGS, TrajectoryKind)

JSON_PATH = Path(__file__).resolve().with_name("catalog.json")


def load_json() -> dict[str, Any]:
    """catalog.json, the generated copy (``make_json.py`` writes it; never edit it)."""
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def rows() -> list[dict[str, Any]]:
    """The catalog's kinds as catalog.json rows: what ``generator.kinds(catalog_rows=...)`` and ``workspace.describe(...)`` grade."""
    return load_json()["kinds"]


def __getattr__(name: str) -> Any:
    # WHY lazy: ``import robot_trajectories`` reads only the vocabulary; the 61 records load when someone asks for them.
    if name in ("CATALOG", "KINDS", "USE_CASES"):
        from . import catalog

        return getattr(catalog, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CATALOG", "DEFINED_IN", "DIMENSIONS", "DIMENSIONS_OF_SCOPE", "ENTRY_EXTRAS", "ENTRY_LEVELS", "ENTRY_LEVEL_OF_NEED",
           "FAMILIES", "JOINS", "JSON_PATH", "KINDS", "NEEDS", "OUTPUTS", "PRIORITIES", "SCOPES", "SPACES", "STARTS_AS",
           "TrajectoryKind", "USE_CASES", "USE_CASE_TAGS", "load_json", "rows"]
