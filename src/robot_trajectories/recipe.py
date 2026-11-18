"""Every trajectory carries the calls that made it, as JSON in ``header.meta["recipe"]``; ``rebuild`` replays them.

    {"format": "robot_trajectories.recipe/1",
     "space": {"joint_names": ["can11"], "rate_hz": 500.0, "limits": {...}, "home": [0.0], ..., "fingerprint": "3f0c..."},
     "calls": [{"kind": "pid_tuning", "joint": "can11", "steps_fraction": [0.5, ...], ..., "start": null}, ...]}

WHY the calls and not the samples: the text is short, a person can read and edit it, and it is the call to make. WHY every
parameter, defaults included: a log must still say what ran after a default changes. WHY the space's fingerprint: amplitudes
are solved against the limits, so the same calls in another space are another trajectory; ``rebuild`` refuses that rather
than hand back something that only looks like the logged run. A space of joint limits alone is written out whole, so its
recipe rebuilds from the text alone; injected models (kinematics, dynamics, collision, a reach map) are data, so their space
is handed back: ``rebuild(text, space)``.
"""

from __future__ import annotations

import json
from typing import Any

from robot_msgs.trajectory import JointTrajectory

from .registry import kinds
from .segments import chain, notes, recipe_of, repeat, segment_info
from .space import JointSpace


def recipe(trajectory: JointTrajectory) -> str:
    """The recipe text a trajectory carries."""
    recipe_of(trajectory)
    return trajectory.header.meta["recipe"]


def calls(text_or_trajectory: str | JointTrajectory) -> list[str]:
    """The recipe as Python calls, one per line, defaults left out (for transcripts and landing pages)."""
    document = recipe_of(text_or_trajectory) if isinstance(text_or_trajectory, JointTrajectory) else json.loads(text_or_trajectory)
    registry = kinds()
    lines = []
    for call in document["calls"]:
        if call["kind"] == "repeat":
            inner = calls(json.dumps({**document, "calls": call["of"]}))
            lines.append(f"repeat(chain({', '.join(inner)}), {call['times']}, label={call['label']!r})")
        else:
            lines.append(registry[call["kind"]].call_text({key: value for key, value in call.items() if key != "kind"}))
    return lines


def rebuild(text_or_trajectory: str | JointTrajectory, space: JointSpace | None = None, *, strict: bool = True) -> JointTrajectory:
    """The trajectory a recipe describes, generated again: identical, sample for sample.

    ``space`` defaults to the one written in the recipe (joint limits alone). A space whose fingerprint differs from the
    recorded one (other limits, home, rate or injected models) is refused; ``strict=False`` regenerates the calls in it anyway."""
    text = recipe(text_or_trajectory) if isinstance(text_or_trajectory, JointTrajectory) else text_or_trajectory
    document = json.loads(text)
    recorded = document["space"]
    if space is None:
        space = JointSpace.from_record(recorded)
    if tuple(recorded["joint_names"]) != space.joint_names:
        raise ValueError(f"the recipe is for joints {recorded['joint_names']}; the space has {list(space.joint_names)}")
    if strict and recorded["fingerprint"] != space.fingerprint:
        raise ValueError(f"this recipe was made in another space ({recorded.get('source', '')!r}, fingerprint "
                         f"{recorded['fingerprint']}); {space!r} differs in its limits, home, rate or injected models. "
                         "strict=False regenerates it there anyway")
    if not document["calls"]:
        raise ValueError("the recipe has no calls (a lone transition is made by chain, not by a kind)")
    made = [_replay(call, space) for call in document["calls"]]
    return made[0] if len(made) == 1 else chain(*made, space=space)


def _replay(call: dict[str, Any], space: JointSpace) -> JointTrajectory:
    if call["kind"] == "repeat":
        inner = [_replay(item, space) for item in call["of"]]
        return repeat(inner[0] if len(inner) == 1 else chain(*inner, space=space), call["times"], label=call["label"])
    registry = kinds()
    if call["kind"] not in registry:
        raise KeyError(f"the recipe calls {call['kind']!r}, which is not registered here (a plugin kind? install the package "
                       f"that provides it); registered: {', '.join(registry)}")
    params = {key: value for key, value in call.items() if key != "kind"}
    return registry[call["kind"]].function(space, **params)


def _call_text(call: dict[str, Any] | None) -> str:
    if call is None:
        return "(a transition chain put in)"
    if call["kind"] == "repeat":
        return f"repeat({' + '.join(_call_text(item) for item in call['of'])}, {call['times']})"
    info = kinds().get(call["kind"])
    params = {key: value for key, value in call.items() if key not in ("kind", "start")}
    return info.call_text(params, brief=True) if info else call["kind"]


def summary(trajectory: JointTrajectory) -> str:
    """One line per segment (its span on the clock, label, the call that makes it, then wait), then the notes."""
    entries = segment_info(trajectory)
    ids = trajectory.channels["segment"]
    lines = [f"{float(trajectory.t[-1]):.1f} s in {len(entries)} segments on {', '.join(trajectory.joint_names)}"]
    start = 0.0
    for index, entry in enumerate(entries):
        end = float(trajectory.t[int((ids == index).nonzero()[0][-1])])
        wait = "  then wait" if entry.get("wait_after") else ""
        lines.append(f"    {start:6.1f} - {end:6.1f} s  {entry['label']:<26} {_call_text(entry.get('call'))}{wait}")
        start = end
    lines += [f"    note: {note}" for note in notes(trajectory)]
    return "\n".join(lines)


__all__ = ["calls", "rebuild", "recipe", "summary"]
