"""The catalog's rules, as plain asserts: runnable by pytest, or as a script where pytest must not run (a powered rig's host).

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_catalog.py

Ported from bam_ws docs/trajectory_generator/catalog/test_catalog.py (b5e1f6965); the catalog now lives in the package
(``robot_trajectories.trajectory_catalog``). Rules: ids (and names) unique; every tag in kind.py's vocabularies; USE_CASES describes
exactly the use-case vocabulary; every kind has a use case and either code we have or a primary source; every ``exists_in`` is
``path:line`` with the file on disk and the line inside it (relative paths are against the bam_ws root, ``$BAM_WS`` or
``~/bam_ws``; a path in another repo is skipped, and counted, only when that whole repo is absent, e.g. on a cloud VM); a
torque-space kind is never a plain position JointTrajectory; catalog.json is what make_json.py writes now, and only ever adds keys
to what the landing pages read. Paths vs trajectories (kind.py has the rules): every kind starts as a path or a time signal;
joint-space dimensions follow the scope, Cartesian ones are a contiguous run; entry_level is the lowest rung of the input ladder that
covers the needs, never higher. The kinds that generate speak the same vocabulary as the rows they claim, and ``catalog()`` lists
every row exactly once: implemented by a kind, or honestly marked not yet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import _runner
from robot_trajectories import catalog, kinds
from robot_trajectories.trajectory_catalog import JSON_PATH, make_json
from robot_trajectories.trajectory_catalog import kind as vocabulary
from robot_trajectories.trajectory_catalog.catalog import CATALOG, KINDS, USE_CASES

# WHY a root to resolve against: catalog rows cite bam_ws code as paths relative to its root (49 of 61 kinds exist there).
REPO = Path(os.environ.get("BAM_WS", Path.home() / "bam_ws"))

SKIPPED: list[str] = []


def _resolve(path_text: str) -> tuple[Path, Path]:
    """(the file, the repo root that must exist for the check to apply)."""

    path = Path(path_text)
    if path.is_absolute():
        parts = path.parts  # ('/', 'home', 'bam', '<repo>', ...)
        return path, Path(*parts[:4])
    return REPO / path, REPO


def test_ids_and_names_are_unique():
    ids = [record.id for record in CATALOG]
    names = [record.name for record in CATALOG]
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})
    assert len(names) == len(set(names)), sorted({n for n in names if names.count(n) > 1})


def test_every_tag_is_in_the_vocabulary():
    for record in CATALOG:
        assert record.family in vocabulary.FAMILIES, (record.id, record.family)
        assert record.space in vocabulary.SPACES, (record.id, record.space)
        assert record.scope in vocabulary.SCOPES, (record.id, record.scope)
        assert record.output in vocabulary.OUTPUTS, (record.id, record.output)
        assert record.segments_join in vocabulary.JOINS, (record.id, record.segments_join)
        assert record.priority in vocabulary.PRIORITIES, (record.id, record.priority)
        assert set(record.use_cases) <= set(vocabulary.USE_CASE_TAGS), (record.id, set(record.use_cases) - set(vocabulary.USE_CASE_TAGS))
        assert set(record.needs) <= set(vocabulary.NEEDS), (record.id, set(record.needs) - set(vocabulary.NEEDS))
        assert len(record.use_cases) == len(set(record.use_cases)) and len(record.needs) == len(set(record.needs)), record.id
        assert record.starts_as in vocabulary.STARTS_AS, (record.id, record.starts_as)
        assert record.defined_in in vocabulary.DEFINED_IN, (record.id, record.defined_in)
        assert set(record.dimensions) <= set(vocabulary.DIMENSIONS), (record.id, record.dimensions)
        assert record.entry_level in vocabulary.ENTRY_LEVELS, (record.id, record.entry_level)
        assert set(record.entry_extras) <= set(vocabulary.ENTRY_EXTRAS), (record.id, record.entry_extras)


def test_every_kind_starts_as_a_path_or_a_time_signal():
    halves = {value: [record.id for record in CATALOG if record.starts_as == value] for value in vocabulary.STARTS_AS}
    assert sum(len(ids) for ids in halves.values()) == len(CATALOG), halves
    assert all(halves.values()), f"an empty half: {halves}"
    # The cases the definition names outright (kind.py's docstring) must land where it says.
    for kind_id in ("linear_chirp", "log_chirp", "multisine", "step_train", "prbs", "torque_chirp", "hold_at_pose"):
        assert KINDS[kind_id].starts_as == "time_signal", kind_id
    for kind_id in ("speed_ladder_a_to_b", "joint_grid_snake", "cartesian_raster", "circular_test", "via_point_spline",
                    "space_filling_pose_set", "iso9283_pose_cube"):
        assert KINDS[kind_id].starts_as == "path", kind_id
    # A torque profile is a schedule in time by construction.
    for record in CATALOG:
        if record.output == "torque_profile":
            assert record.starts_as == "time_signal", record.id


def test_defined_in_is_the_space_field():
    assert vocabulary.DEFINED_IN == vocabulary.SPACES
    for record in CATALOG:
        assert record.defined_in == record.space, record.id


def test_dimensions_follow_the_rule():
    for record in CATALOG:
        dimensions = record.dimensions
        assert dimensions, record.id
        positions = [vocabulary.DIMENSIONS.index(value) for value in dimensions]
        assert positions == list(range(positions[0], positions[0] + len(positions))), f"{record.id}: {dimensions} is not a contiguous run"
        if record.space == "cartesian":
            continue  # the geometry's dimensions are authored (line 1D, plane 2D, volume 3D, + orientation 6D)
        assert dimensions == vocabulary.DIMENSIONS_OF_SCOPE[record.scope], (record.id, record.scope, dimensions)
    assert set(vocabulary.DIMENSIONS_OF_SCOPE) == set(vocabulary.SCOPES) - {"cartesian_path"}
    # The cases the definition names outright.
    assert KINDS["cartesian_straight_line"].dimensions == ("1D",)
    assert KINDS["cartesian_raster"].dimensions == ("2D",) and KINDS["circular_test"].dimensions == ("2D",)
    assert "3D" in KINDS["iso9283_pose_cube"].dimensions
    assert "6D" in KINDS["hand_eye_station_set"].dimensions
    # Every one of the four columns has kinds in it.
    assert all(any(value in record.dimensions for record in CATALOG) for value in vocabulary.DIMENSIONS)


def test_entry_level_is_the_lowest_rung_that_covers_the_needs():
    assert set(vocabulary.ENTRY_LEVEL_OF_NEED) == set(vocabulary.NEEDS), set(vocabulary.ENTRY_LEVEL_OF_NEED) ^ set(vocabulary.NEEDS)
    assert set(vocabulary.ENTRY_LEVEL_OF_NEED.values()) - {None} == set(vocabulary.ENTRY_LEVELS)
    rung = vocabulary.ENTRY_LEVELS.index
    for record in CATALOG:
        level = rung(record.entry_level)
        need_levels = [rung(vocabulary.ENTRY_LEVEL_OF_NEED[need]) for need in record.needs if vocabulary.ENTRY_LEVEL_OF_NEED[need]]
        assert all(need_level <= level for need_level in need_levels), f"{record.id}: a need sits above {record.entry_level}"
        assert level == 0 or level in need_levels, f"{record.id}: {record.entry_level} over-asks (no need sits on it)"
        assert (record.entry_extras == ("collision",)) == ("collision_model" in record.needs), record.id
        if record.space == "cartesian":  # a Cartesian kind cannot be generated from joint limits alone
            assert level >= rung("kinematics"), record.id
        if record.output == "torque_profile":
            assert record.entry_level == "torque_limits", record.id


def test_use_cases_describe_exactly_the_vocabulary():
    assert set(USE_CASES) == set(vocabulary.USE_CASE_TAGS), set(USE_CASES) ^ set(vocabulary.USE_CASE_TAGS)
    assert all(text.strip() for text in USE_CASES.values())


def test_every_kind_has_a_use_case_and_a_source():
    for record in CATALOG:
        assert record.use_cases, record.id
        assert record.exists_in or record.prior_art, f"{record.id}: neither code we have nor a primary source"


def test_every_exists_in_path_is_on_disk_at_that_line():
    SKIPPED.clear()
    for record in CATALOG:
        for reference in record.exists_in:
            path_text, _, line_text = reference.rpartition(":")
            assert path_text and line_text.isdigit(), f"{record.id}: {reference!r} is not path:line"
            path, root = _resolve(path_text)
            if not root.is_dir():
                SKIPPED.append(reference)
                continue
            assert path.is_file(), f"{record.id}: {path} does not exist"
            with path.open(encoding="utf-8", errors="replace") as handle:
                lines = sum(1 for _ in handle)
            assert 1 <= int(line_text) <= lines, f"{record.id}: {reference} is past the end ({lines} lines)"


def test_parameters_are_name_unit_default():
    for record in CATALOG:
        names = [name for name, _, _ in record.parameters]
        assert len(names) == len(set(names)), record.id
        for name, unit, default in record.parameters:
            assert name and unit and default is not None, (record.id, name)
            assert "deg" not in unit, f"{record.id}: {name} is in {unit}; parameters are SI, degrees are for display only"


def test_torque_kinds_are_not_position_trajectories():
    for record in CATALOG:
        if record.space == "torque":
            assert record.output != "joint_trajectory" and not record.joint_trajectory, record.id


def test_there_is_a_now_list():
    assert [record.id for record in CATALOG if record.priority == "now"]


def test_catalog_json_is_current():
    path = JSON_PATH
    assert path.is_file(), "run python -m robot_trajectories.trajectory_catalog.make_json"
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(make_json.document())), \
        "catalog.json is stale: run python -m robot_trajectories.trajectory_catalog.make_json"


# The keys catalog.json had when the five landing pages were built on it (commit 28c1cb659). WHY frozen: those pages read
# TG_DATA.catalog.kinds[*] by these names, so the JSON may gain keys but never lose or rename one.
LEGACY_TOP_KEYS = {"counts", "generated_by", "kinds", "schema", "vocabulary"}
LEGACY_ROW_KEYS = {"exists_in", "family", "id", "joint_trajectory", "name", "needs", "notes", "one_line", "output", "parameters",
                   "prior_art", "priority", "scope", "segments_join", "space", "use_cases"}
LEGACY_COUNT_KEYS = {"by_family", "by_output", "by_priority", "kinds"}
LEGACY_VOCABULARY_KEYS = {"families", "joins", "needs", "outputs", "priorities", "scopes", "spaces", "use_cases"}
NEW_ROW_KEYS = {"starts_as", "defined_in", "dimensions", "entry_level", "entry_extras"}


def test_catalog_json_only_adds_keys():
    document = make_json.document()
    assert LEGACY_TOP_KEYS <= set(document), LEGACY_TOP_KEYS - set(document)
    assert LEGACY_COUNT_KEYS <= set(document["counts"]), LEGACY_COUNT_KEYS - set(document["counts"])
    assert LEGACY_VOCABULARY_KEYS <= set(document["vocabulary"]), LEGACY_VOCABULARY_KEYS - set(document["vocabulary"])
    for row in document["kinds"]:
        assert LEGACY_ROW_KEYS | NEW_ROW_KEYS <= set(row), (row["id"], (LEGACY_ROW_KEYS | NEW_ROW_KEYS) - set(row))


def test_kinds_speak_the_catalog_vocabulary():
    """Every kind that generates uses the catalog's tags, and agrees with the row it claims on family and starts_as.
    WHY not use cases or entry level: a kind keeps its own tags (chirp serves pid_tuning; workspace_sweep reads a reach map
    with a box-grid fallback), and the catalog row stays what the sweep found."""
    for name, info in kinds().items():
        assert info.family in vocabulary.FAMILIES, (name, info.family)
        assert info.starts_as in vocabulary.STARTS_AS, (name, info.starts_as)
        assert set(info.use_cases) <= set(vocabulary.USE_CASE_TAGS), (name, set(info.use_cases) - set(vocabulary.USE_CASE_TAGS))
        assert set(info.needs) <= set(vocabulary.NEEDS), (name, set(info.needs) - set(vocabulary.NEEDS))
        if info.catalog_id is not None:
            row = KINDS[info.catalog_id]
            assert (info.family, info.starts_as) == (row.family, row.starts_as), (name, row.id)


def test_catalog_lists_every_row_once_honestly():
    """``catalog()`` = the kinds that generate, then every row no kind claims, marked not implemented: nothing twice, nothing lost."""
    listing = catalog()
    claimed = {info.catalog_id for info in kinds().values() if info.catalog_id}
    assert listing.implemented.names == list(kinds()), listing.implemented.names
    assert sorted(entry.catalog_id for entry in listing.todo) == sorted(row.id for row in CATALOG if row.id not in claimed)
    assert all(not entry.implemented for entry in listing.todo)
    assert {entry.catalog_id for entry in listing.implemented if entry.catalog_id} == claimed


def test_conveyor_use_case_finds_the_moving_target_rows():
    names = catalog(use_case="conveyor_pick_and_place").names
    for expected in ("pick_and_place", "moving_target_track", "track_and_release", "moving_pick_and_place", "touch_down_contact"):
        assert expected in names, (expected, names)


def main() -> int:
    code = _runner.run(globals(), "test_catalog")
    references = sum(len(record.exists_in) for record in CATALOG)
    print(f"  {len(CATALOG)} kinds, {references} exists_in references checked against {REPO}, {len(SKIPPED)} skipped (repo absent)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
