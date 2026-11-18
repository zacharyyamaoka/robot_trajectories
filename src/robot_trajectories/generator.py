"""JointTrajectoryGenerator: a Workspace in, records or plans in, a Program out.

    generator = JointTrajectoryGenerator(Workspace.from_limits(limits))     # or the from_* shorthands below
    generator.kinds(use_case="pid_tuning")                                   # available / fallback / locked, and why
    program = generator.generate(plans.pid_tuning(joint="can11"))           # a plan ...
    program = generator.generate([StepTrain(joint="can11"), Hold(duration_s=2.0)])   # ... or records, or both
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from robot_msgs.trajectory import JointLimits

from . import kernels as k

from .kind import Context, Kind, catalog
from .plan import Plan, PlanStep
from .workspace import Status, Workspace

TRANSITION_SPEED = 0.5  # fraction of the limits a folded-in transition uses (the kernels' join default)


class KindLocked(ValueError):
    """A record was generated on a workspace that lacks something it needs. The message names the input and the fix."""


@dataclass(frozen=True)
class KindRow:
    """One row of ``generator.kinds()``: a kind (registered, or a catalog row) and its status here."""

    id: str
    catalog_id: str | None
    use_cases: tuple[str, ...]
    entry_level: str
    starts_as: str
    status: Status
    generated: bool  # False: catalogued, not generated yet


class KindTable(tuple):
    """Rows of the catalog with their status on one workspace; ``print()`` it, ``.ids`` for the names."""

    def __new__(cls, rows: Iterable[KindRow], heading: str = ""):
        table = super().__new__(cls, rows)
        table.heading = heading
        return table

    @property
    def ids(self) -> list[str]:
        return [row.id for row in self]

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            for row in self:
                if key in (row.id, row.catalog_id):
                    return row
            raise KeyError(key)
        return super().__getitem__(key)

    def __str__(self) -> str:
        width = max((len(row.id) for row in self), default=4)
        lines = [self.heading]
        for row in self:
            state = row.status.state if row.generated else f"{row.status.state}*"
            if row.status.state != "available":
                note = f"{row.status.why} -> {row.status.fix}"
            else:
                note = f"catalog: {row.catalog_id}" if row.generated and row.catalog_id and row.catalog_id != row.id else ""
                if row.status.unchecked:
                    note += f" (not checked here: {', '.join(n.replace('_', ' ') for n in row.status.unchecked)})"
            lines.append(f"  {row.id:<{width}}  {state:<10} {row.entry_level.replace('_', ' '):<13} "
                         f"{row.starts_as.replace('_', ' '):<12} {note.strip()}".rstrip())
        if any(not row.generated for row in self):
            lines.append("  * catalogued, not generated yet: the status says whether this workspace has its inputs")
        return "\n".join(lines)


class JointTrajectoryGenerator:
    """Makes robot_msgs JointTrajectories inside one Workspace, from records or plans."""

    def __init__(self, workspace: Workspace, *, rate_hz: float = k.DEFAULT_RATE_HZ) -> None:
        if not isinstance(workspace, Workspace):
            raise TypeError(f"a generator is built on a Workspace (Workspace.from_limits(...)), not {type(workspace).__name__}")
        self.workspace = workspace
        self.rate_hz = float(rate_hz)

    # ------------------------------------------------------------------ shorthands for the three entry forms

    @classmethod
    def from_limits(cls, limits: JointLimits, *, rate_hz: float = k.DEFAULT_RATE_HZ, **workspace: Any) -> JointTrajectoryGenerator:
        return cls(Workspace.from_limits(limits, **workspace), rate_hz=rate_hz)

    @classmethod
    def from_description(cls, description: Any, *, max_acceleration: Any = None, box: Any = None, home: Any = None,
                         rate_hz: float = k.DEFAULT_RATE_HZ) -> JointTrajectoryGenerator:
        workspace = Workspace.from_description(description, max_acceleration=max_acceleration)
        workspace = workspace.within(box) if box is not None else workspace
        if home is not None:
            workspace = Workspace._build(workspace.limits, built_from=workspace.built_from, home=home, name=workspace.name,
                                         assumptions=workspace.assumptions)
        return cls(workspace, rate_hz=rate_hz)

    @classmethod
    def from_reach_map(cls, reach_map: Any, limits: JointLimits, *, home: Any = None,
                       rate_hz: float = k.DEFAULT_RATE_HZ) -> JointTrajectoryGenerator:
        return cls(Workspace.from_reach_map(reach_map, limits, home=home), rate_hz=rate_hz)

    # ------------------------------------------------------------------ what it can make here

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.workspace.joint_names

    def kinds(self, use_case: str | None = None, *, catalog_rows: Sequence[Mapping[str, Any]] | None = None) -> KindTable:
        """Every registered kind (or those serving ``use_case``) with its status here: available, fallback or locked, the
        missing input and the call that adds it. With ``catalog_rows`` (catalog.json's kinds), the rows no record
        generates yet are graded too, on the catalog's own ``needs`` and ``entry_level``."""
        rows = [KindRow(kind.id, kind.catalog_id, tuple(kind.use_cases), kind.entry_level(), kind.starts_as,
                        self.workspace.kind_status(kind), True) for kind in catalog(use_case)]
        if catalog_rows is not None:
            claimed = {kind.catalog_id for kind in catalog()}
            for entry in catalog_rows:
                if entry["id"] in claimed or (use_case is not None and use_case not in entry["use_cases"]):
                    continue
                rows.append(KindRow(entry["id"], entry["id"], tuple(entry["use_cases"]), entry["entry_level"], entry["starts_as"],
                                    self.workspace.row_status(entry), False))
        counts = {state: sum(row.generated and row.status.state == state for row in rows) for state in ("available", "fallback", "locked")}
        heading = (f"{sum(row.generated for row in rows)} kinds{' for ' + use_case if use_case else ''} on this workspace "
                   f"({self.workspace.built_from}): " + ", ".join(f"{count} {state}" for state, count in counts.items()))
        return KindTable(rows, heading)

    def generate(self, items: Kind | PlanStep | Plan | Iterable[Kind | PlanStep | Plan], *, start: Sequence[float] | None = None) -> Any:
        """The Program: one rest-to-rest segment per record, each starting where the last ended, and all of them joined.

        ``items`` is a record, a plan, or a list of either (plans flatten into their steps). ``start`` is where the robot
        is when the program begins (default the workspace's home). A record whose motion begins elsewhere (a multisine
        centred on a box, a tour's first grid point) gets a min-jerk transition folded into the front of its own segment.
        """
        from .program import Program

        steps = _flatten(items)
        for index, item in enumerate(steps):
            status = self.workspace.kind_status(type(item.record))
            if status.state == "locked":
                raise KindLocked(f"step {index} {item.record!r} is locked here: {status.why}. Fix: {status.fix}")
        here = self.workspace.home if start is None else np.asarray(start, float)
        notes: list[str] = []
        segments: list[tuple[k.Arrays, list[k.Arrays]]] = []
        for index, item in enumerate(steps):
            record = item.record
            status = self.workspace.kind_status(type(record))
            context = Context(self.joint_names, self.workspace.bounds, self.rate_hz, here, self.workspace.home, self.workspace.reach, notes)
            if status.state == "fallback":
                context.note(f"{record.id}: {status.why}")
            try:
                pieces = list(record.pieces(context))
            except ValueError as error:
                raise ValueError(f"step {index} {record!r}: {error}") from error
            if not pieces:
                pieces = [k.hold(here, 0.0, self.rate_hz, label=record.display_label)]
            if not np.allclose(pieces[0].q[0], here, atol=1e-9):
                pieces.insert(0, k.min_jerk(here, pieces[0].q[0], self.workspace.bounds, self.rate_hz, speed=TRANSITION_SPEED,
                                            label=f"transition into {record.display_label}"))
            arrays, used = k.join(pieces, self.workspace.bounds, self.rate_hz, transition_speed=TRANSITION_SPEED)
            segments.append((arrays, used))
            here = arrays.q[-1]
        return Program.build(self, steps, segments, notes=notes, start=None if start is None else [float(v) for v in np.asarray(start, float)])

    # ------------------------------------------------------------------ as configuration

    def to_config(self) -> dict[str, Any]:
        return {"rate_hz": self.rate_hz, "workspace": self.workspace.to_config()}

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_config(), sort_keys=True).encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        return f"JointTrajectoryGenerator({self.workspace.built_from}: {', '.join(self.joint_names)}; {self.rate_hz:g} Hz)"


def _flatten(items: Any) -> tuple[PlanStep, ...]:
    if isinstance(items, (Kind, PlanStep, Plan)):
        items = [items]
    steps: list[PlanStep] = []
    for index, item in enumerate(items):
        if isinstance(item, Plan):
            steps += list(item.steps)
        elif isinstance(item, PlanStep):
            steps.append(item)
        elif isinstance(item, Kind):
            steps.append(PlanStep(item))
        else:
            raise TypeError(f"item {index} is {type(item).__name__}: generate takes records (e.g. Chirp(...)) and plans")
    if not steps:
        raise ValueError("nothing to generate: pass at least one record or plan")
    return tuple(steps)
