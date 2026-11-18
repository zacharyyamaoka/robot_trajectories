"""Plan (from V3, made robot-free): a named list of steps, each a record plus why it is there and what to do after it.

    plan = plans.pid_tuning(joint="can11")       # Plan: step train, settle, chirp, settle (no robot yet)
    plan + plans.friction_id(joint="can11")      # plans add with +
    generator.generate(plan)                     # the robot enters here, and only here

WHY robot-free: a plan is configuration, not motion. It can be written, printed, saved and reviewed before anyone picks
a robot, and the same plan runs on any robot whose workspace has its inputs. The robot's numbers (the room, the speeds)
come in at ``generate``; V3's plans needed the robot first, which made every plan call carry it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal

from .kind import Kind

Then = Literal["continue", "wait"]
FORMAT = "robot_trajectories.plan/1"


@dataclass(frozen=True)
class PlanStep:
    """One step: a record, what it measures, and whether to stop and look after it."""

    record: Kind
    purpose: str = ""
    then: Then = "continue"
    plan: str = ""  # the plan it came from, so a sum of plans still says whose step it is

    def __post_init__(self) -> None:
        if not isinstance(self.record, Kind):
            raise TypeError(f"a plan step holds a record (e.g. Chirp(...)), not {type(self.record).__name__}")
        if self.then not in ("continue", "wait"):
            raise ValueError(f"then is 'continue' or 'wait', not {self.then!r}")

    def to_config(self) -> dict[str, Any]:
        return {"plan": self.plan, "purpose": self.purpose, "then": self.then, "record": self.record.to_config()}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PlanStep:
        return cls(Kind.from_config(config["record"]), config.get("purpose", ""), config.get("then", "continue"), config.get("plan", ""))


def step(record: Kind, purpose: str = "", *, then: Then = "continue") -> PlanStep:
    """``step(Chirp(joint="can11"), "frequency response: bandwidth, phase margin")``."""
    return PlanStep(record, purpose, then)


@dataclass(frozen=True)
class Plan:
    """What to run, in order, and why. Iterate it for the steps; ``generator.generate(plan)`` makes the motion."""

    name: str
    steps: tuple[PlanStep, ...]
    made_by: tuple[str, ...] = ()  # the calls that made it, e.g. "plans.pid_tuning(joint='can11')"

    def __add__(self, other: Plan | PlanStep | Kind) -> Plan:
        if isinstance(other, Kind):
            other = Plan(other.display_label, (PlanStep(other, plan=other.display_label),))
        elif isinstance(other, PlanStep):
            other = Plan(other.plan or other.record.display_label, (other,))
        if not isinstance(other, Plan):
            return NotImplemented
        # WHY no glue step: the generator starts every record where the last one ended and folds in the transition, so
        # plans add with no seam, on any robot.
        return Plan(f"{self.name} + {other.name}", self.steps + other.steps, self.made_by + other.made_by)

    def __iter__(self) -> Iterator[PlanStep]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> PlanStep:
        return self.steps[index]

    @property
    def records(self) -> tuple[Kind, ...]:
        return tuple(item.record for item in self.steps)

    @property
    def plans(self) -> list[str]:
        """The plans this one adds up, in order."""
        return list(dict.fromkeys(item.plan for item in self.steps))

    def with_then(self, index: int, then: Then) -> Plan:
        steps = list(self.steps)
        steps[index] = replace(steps[index], then=then)
        return replace(self, steps=tuple(steps))

    def describe(self) -> str:
        """The steps in words: record, purpose, wait. No durations: those need a robot (``program.summary()`` has them)."""
        lines = [f"{self.name}: {len(self.steps)} steps (configuration; no robot until generate)"]
        for index, item in enumerate(self.steps, start=1):
            wait = "  then wait" if item.then == "wait" else ""
            lines.append(f"  {index}. {item.record!r}{wait}")
            if item.purpose:
                lines.append(f"       {item.purpose}")
        return "\n".join(lines)

    def to_python(self) -> str:
        """The records as Python: the configuration this plan is."""
        return "[\n" + "".join(f"    {item.record!r},\n" for item in self.steps) + "]"

    def to_json(self) -> str:
        values = {"format": FORMAT, "name": self.name, "made_by": list(self.made_by)}
        head = ",\n ".join(f"{json.dumps(key)}: {json.dumps(value)}" for key, value in values.items())
        steps = ",\n  ".join(json.dumps(item.to_config()) for item in self.steps)
        return "{" + head + ",\n \"steps\": [\n  " + steps + "\n ]}\n"

    @classmethod
    def from_json(cls, text: str) -> Plan:
        values = json.loads(text)
        if values.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} text (format {values.get('format')!r})")
        return cls(values["name"], tuple(PlanStep.from_config(item) for item in values["steps"]), tuple(values.get("made_by", ())))

    def __repr__(self) -> str:
        return f"Plan({self.name}: {len(self.steps)} steps)"
