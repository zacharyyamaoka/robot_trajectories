"""Write catalog.json from catalog.py: the generated copy a landing page or an app reads (never edit the JSON).

    /home/bam/robot_trajectories/.venv/bin/python -m robot_trajectories.trajectory_catalog.make_json

Stdlib only. The output is stable (sorted vocabularies as declared, kinds in catalog order, two-space indent), so a re-run with an
unchanged catalog is a zero diff.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path

from . import kind as vocabulary
from .catalog import CATALOG, USE_CASES

HERE = Path(__file__).resolve().parent
SCHEMA = "trajectory_catalog/1"


def document() -> dict:
    kinds = []
    for record in CATALOG:
        row = dataclasses.asdict(record)
        row["parameters"] = [{"name": name, "unit": unit, "default": default} for name, unit, default in record.parameters]
        row["prior_art"] = [{"citation": citation, "url": url} for citation, url in record.prior_art]
        row["joint_trajectory"] = record.joint_trajectory
        # WHY add-only: the round's five landing pages read these rows through TG_DATA.catalog; new keys never rename or drop an old one.
        row["defined_in"] = record.defined_in
        row["entry_level"] = record.entry_level
        row["entry_extras"] = list(record.entry_extras)
        kinds.append(row)
    return {
        "schema": SCHEMA,
        "generated_by": "robot_trajectories/trajectory_catalog/make_json.py from catalog.py (do not edit)",
        "counts": {
            "kinds": len(CATALOG),
            "by_family": dict(Counter(record.family for record in CATALOG)),
            "by_priority": dict(Counter(record.priority for record in CATALOG)),
            "by_output": dict(Counter(record.output for record in CATALOG)),
            "by_starts_as": {value: sum(record.starts_as == value for record in CATALOG) for value in vocabulary.STARTS_AS},
            "by_dimension": {value: sum(value in record.dimensions for record in CATALOG) for value in vocabulary.DIMENSIONS},
            "by_entry_level": {value: sum(record.entry_level == value for record in CATALOG) for value in vocabulary.ENTRY_LEVELS},
        },
        "vocabulary": {
            "families": list(vocabulary.FAMILIES),
            "use_cases": USE_CASES,
            "spaces": list(vocabulary.SPACES),
            "scopes": list(vocabulary.SCOPES),
            "outputs": list(vocabulary.OUTPUTS),
            "needs": list(vocabulary.NEEDS),
            "joins": list(vocabulary.JOINS),
            "priorities": list(vocabulary.PRIORITIES),
            "starts_as": list(vocabulary.STARTS_AS),
            "defined_in": list(vocabulary.DEFINED_IN),
            "dimensions": list(vocabulary.DIMENSIONS),
            "entry_levels": list(vocabulary.ENTRY_LEVELS),
            "entry_extras": list(vocabulary.ENTRY_EXTRAS),
            "entry_level_of_need": dict(vocabulary.ENTRY_LEVEL_OF_NEED),
        },
        "kinds": kinds,
    }


def main() -> int:
    out = HERE / "catalog.json"
    out.write_text(json.dumps(document(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(CATALOG)} kinds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
