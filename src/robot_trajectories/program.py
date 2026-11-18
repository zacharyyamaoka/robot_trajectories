"""Program (from V2): what ``generator.generate(...)`` returns. One result, three views of the same motion:

- ``program.trajectory``: one long robot_msgs JointTrajectory, every segment tied to the next;
- ``program.segments``: the same motion as rest-to-rest JointTrajectories, one per record (run one, wait, run the next);
  ``program.steps`` says, per segment, what it measures and whether to wait after it (from a plan);
- ``program.to_json()``: the records and the workspace as text; ``Program.from_json(text)`` makes it again, identically,
  and refuses a different robot.

WHY one segment per record, transitions folded in: ``zip(program.records, program.segments)`` always lines up, and a
segment is runnable on its own from where the last one stopped. A transition is still visible as a band.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from robot_msgs.std import Header
from robot_msgs.trajectory import JointTrajectory

from . import kernels as k

from .kind import Kind
from .plan import PlanStep

FORMAT = "robot_trajectories.program/2"
META_KEY = "robot_trajectories.program"  # the joined trajectory's header.meta carries its own config (a log rebuilds)
RECORD_KEY = "robot_trajectories.record"  # each segment's header.meta carries its record


@dataclass(frozen=True)
class Band:
    """One labelled stretch of the joined clock (a kernel piece: a step, a dwell, a leg, a transition)."""

    segment: int
    label: str
    start_s: float
    end_s: float

    @property
    def transition(self) -> bool:
        return self.label.startswith("transition")


@dataclass(frozen=True, eq=False)
class Program:
    generator: Any  # JointTrajectoryGenerator
    steps: tuple[PlanStep, ...]
    trajectory: JointTrajectory
    segments: tuple[JointTrajectory, ...]
    bands: tuple[Band, ...]
    notes: tuple[str, ...] = ()  # what the records reported while generating (a clamped rung, a skipped speed, a fallback)
    start: tuple[float, ...] | None = None  # None: the program starts at the workspace's home

    @classmethod
    def build(cls, generator: Any, steps: tuple[PlanStep, ...], segments: Sequence[tuple[k.Arrays, list[k.Arrays]]], *,
              notes: Sequence[str], start: Sequence[float] | None) -> Program:
        rate = generator.rate_hz
        joined, _ = k.join([k.Arrays(arrays.t - arrays.t[0], arrays.q, arrays.qd, arrays.qdd, item.record.display_label)
                            for item, (arrays, _) in zip(steps, segments, strict=True)], generator.workspace.bounds, rate)
        start = None if start is None else tuple(float(value) for value in start)
        limits, names = generator.workspace.limits, generator.workspace.joint_names
        config_text = _text(_config(generator, steps, start))
        trajectory = k.to_trajectory(joined, limits, names).replace(header=Header(meta={META_KEY: config_text}))
        trajectories, bands = [], []
        first_sample = 0
        for index, (item, (arrays, used)) in enumerate(zip(steps, segments, strict=True)):
            segment = k.to_trajectory(arrays, limits, names)
            trajectories.append(segment.replace(header=Header(meta={RECORD_KEY: json.dumps(item.record.to_config())})))
            offset = float(joined.t[first_sample])
            for piece_start, piece in zip(k.segment_starts(used), used, strict=True):
                bands.append(Band(index, piece.label, offset + float(piece_start), offset + float(piece_start + piece.t[-1])))
            first_sample += len(arrays.t) - 1
        return cls(generator, steps, trajectory, tuple(trajectories), tuple(bands), tuple(notes), start)

    # ------------------------------------------------------------------ views

    @property
    def records(self) -> tuple[Kind, ...]:
        return tuple(item.record for item in self.steps)

    @property
    def labels(self) -> list[str]:
        return [item.record.display_label for item in self.steps]

    @property
    def spans(self) -> list[tuple[float, float]]:
        """(start_s, end_s) of each segment on the joined clock."""
        spans, start = [], 0.0
        for segment in self.segments:
            spans.append((start, start + float(segment.t[-1])))
            start += float(segment.t[-1])
        return spans

    @property
    def duration_s(self) -> float:
        return float(self.trajectory.t[-1])

    def tapes(self) -> list[Program]:
        """One program per segment, each rebuilt from its record and the place it starts: a tape you can save and play
        alone. Its trajectory is that segment, sample for sample (generation is deterministic)."""
        out = []
        for item, segment in zip(self.steps, self.segments, strict=True):
            out.append(self.generator.generate(item, start=[float(value) for value in segment.positions[0]]))
        return out

    # ------------------------------------------------------------------ as configuration

    @property
    def config(self) -> dict[str, Any]:
        return _config(self.generator, self.steps, self.start)

    def to_json(self) -> str:
        """The config as JSON text, one step per line so a person can read and edit it."""
        return _text(self.config)

    @classmethod
    def from_json(cls, text: str, *, generator: Any = None, reach_map: Any = None, kinematics: Any = None,
                  dynamics: Any = None) -> Program:
        """Generate the program a ``to_json()`` text describes, identically.

        The text carries the effective limits, so a program made from joint limits or a description rebuilds from the
        text alone. A reach map (or kinematics / dynamics) is only fingerprinted: hand it back, or a generator built on it.
        A generator for another robot or other limits is refused, never silently used.
        """
        from .generator import JointTrajectoryGenerator
        from .workspace import Workspace

        config = json.loads(text)
        if config.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} text (format {config.get('format')!r})")
        steps = tuple(PlanStep.from_config(item) for item in config["steps"])
        if generator is None:
            workspace = Workspace.from_config(config["generator"]["workspace"], reach_map=reach_map, kinematics=kinematics, dynamics=dynamics)
            generator = JointTrajectoryGenerator(workspace, rate_hz=config["generator"]["rate_hz"])
        elif generator.fingerprint() != _fingerprint(config["generator"]):
            written = config["generator"]["workspace"]
            raise ValueError(f"this program was made for another robot or limits ({written['built_from']}, joints "
                             f"{written['limits']['joint_names']}); {generator!r} differs")
        return generator.generate(steps, start=config.get("start"))

    @classmethod
    def from_trajectory(cls, trajectory: JointTrajectory, **kwargs: Any) -> Program:
        """The program a joined trajectory came from (its header.meta carries the config), e.g. read back from a log."""
        if META_KEY not in trajectory.header.meta:
            raise ValueError("this trajectory carries no robot_trajectories program config")
        return cls.from_json(trajectory.header.meta[META_KEY], **kwargs)

    def to_python(self) -> str:
        """The records as the Python call that makes this program (each record shows only its changed fields)."""
        return "program = generator.generate([\n" + "".join(f"    {item.record!r},\n" for item in self.steps) + "])"

    def summary(self) -> str:
        """One line per segment: span, label, wait and the record (angles in degrees where printed), then the notes."""
        lines = [f"{self.duration_s:.1f} s in {len(self.segments)} segments on {', '.join(self.generator.workspace.joint_names)}"]
        for (start, end), item in zip(self.spans, self.steps, strict=True):
            transitions = [band for band in self.bands if band.transition and start - 1e-9 <= band.start_s < end]
            extra = f"  (+{sum(b.end_s - b.start_s for b in transitions):.1f} s transition in)" if transitions else ""
            wait = " then wait" if item.then == "wait" else ""
            lines.append(f"    {start:6.1f} - {end:6.1f} s  {item.record.display_label:<22} {item.record!r}{extra}{wait}")
        lines += [f"    note: {note}" for note in self.notes]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Program({len(self.steps)} records, {self.duration_s:.1f} s, {self.generator!r})"


def _config(generator: Any, steps: Sequence[PlanStep], start: Sequence[float] | None) -> dict[str, Any]:
    return {"format": FORMAT, "generator": generator.to_config(), "start": None if start is None else list(start),
            "steps": [item.to_config() for item in steps]}


def _text(config: dict[str, Any]) -> str:
    head = ",\n ".join(f"{json.dumps(key)}: {json.dumps(config[key])}" for key in ("format", "generator", "start"))
    steps = ",\n  ".join(json.dumps(item) for item in config["steps"])
    return "{" + head + ",\n \"steps\": [\n  " + steps + "\n ]}\n"


def _fingerprint(generator_config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(generator_config, sort_keys=True).encode()).hexdigest()[:16]
