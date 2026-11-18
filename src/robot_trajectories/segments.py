"""Segments are a view of one robot_msgs JointTrajectory, not a type: ``chain`` joins, ``split`` gives the pieces back.

    session = chain(step_train(space, joint="can11"), hold(space, 2.0), chirp(space, joint="can11"))
    split(session)          # the same motion as rest-to-rest JointTrajectories, one per segment, each t from 0
    labels(session)         # ['step train', 'hold', 'chirp']
    repeat(tape, 5)         # one tape played five times, the same input each time (one segment)
    annotate(t, "what it measures", wait_after=True)

What every trajectory this package makes carries (robot_msgs fields only, so anything that keeps a JointTrajectory keeps it):

- ``channels["segment"]`` / ``channels["piece"]``: per sample, the index of the segment and of the kernel piece playing. A
  segment owns its samples after its first; its first sample is the previous one's last (the shared rest point), so the
  pieces stitch end to end exactly. A piece is a finer band inside a segment (a leg, a dwell, a rung).
- ``header.meta["segments"]``: JSON, one entry per segment: ``label``, ``call`` (the one call that makes it alone, or null for
  a transition ``chain`` put in), and optionally ``purpose``, ``wait_after``, ``notes``, ``made_by``, ``starts`` / ``ends``
  ("moving", or "anywhere" for an event-ended end), ``command``, ``until``, ``assumes``, and ``first_sample`` (below).
- WHY ``first_sample``: joined segments share one sample (the previous one's last); where a segment's own first sample
  differs from it bit for bit (a signed zero, the acceleration a fade ends on), the entry keeps it, so ``split`` gives every
  segment back exactly as its kind made it.
- ``header.meta["pieces"]``: JSON, one ``{"label", "segment"}`` per piece.
- ``header.meta["recipe"]``: JSON ``{"format", "space", "calls"}``: the calls that made it (``recipe.rebuild`` replays them).
- Optional: ``channels["tool_target"]`` (N, 3), the tool point a Cartesian kind commanded (NaN where none was), and
  ``effort`` (N, J) where a kind commands torque (NaN where none is commanded).

Boundary states: a segment starts and ends at rest unless its kind declares ``starts="moving"`` / ``ends="moving"``.
Where either side of a join is moving, ``chain`` requires the two states to match (position and velocity), and never
puts a transition there (a rest-to-rest transition cannot join moving states).

WHY a channel and not start indices (V1's decision 2): a channel is per sample, so it survives slicing, resampling (integer
channels hold) and logging next to the measured run; indices in meta would go stale the moment the trajectory is cut.

Every function here is pure: trajectories in, a new trajectory out.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from robot_msgs.std import Header
from robot_msgs.trajectory import JointLimits, JointTrajectory

from . import kernels as k

SEGMENT, PIECE, TOOL_TARGET = "segment", "piece", "tool_target"
RECIPE_FORMAT = "robot_trajectories.recipe/1"
TRANSITION_SPEED = 0.5  # WHY 0.5: kernels.join's transition speed, so a chain's transitions are the round's


class UnsafeTrajectory(ValueError):
    """A trajectory left its limits, jumped, did not rest at its ends, or collided. Kinds are built never to raise this; it
    is the guard that proves it, for kinds nobody has written yet too."""


@dataclass(frozen=True)
class Piece(k.Arrays):
    """A kernel piece (``t, q, qd, qdd, label``) that also carries per-sample channels or a commanded effort.

    WHY a subclass and not a new field on ``kernels.Arrays``: the kernels are vendored byte for byte; ``kernels.join`` keeps
    the piece objects it was given, so their extras are found again by identity after the join."""

    channels: Mapping[str, np.ndarray] = field(default_factory=dict)
    effort: np.ndarray | None = None


class Motion(NamedTuple):
    """What a kind's body may return when it has something to say: its pieces, notes for the reader (a clamp, a skip), and
    keys for its segment's entry (``assumes``: the motion model a moving-target plan relies on; ``sync``: its window)."""

    pieces: Sequence[k.Arrays]
    notes: Sequence[str] = ()
    entry: Mapping[str, Any] = {}


@dataclass(frozen=True)
class Band:
    """One labelled stretch of a trajectory's clock (a kernel piece: a step, a dwell, a leg, a transition)."""

    segment: int
    label: str
    start_s: float
    end_s: float

    @property
    def transition(self) -> bool:
        return self.label.startswith("transition")


# ====================================================================== the guard


def guard(arrays: k.Arrays, limits: JointLimits | k.Bounds, what: str, *, rest_start: bool = True, rest_end: bool = True,
          effort: np.ndarray | None = None, collides: Callable[[np.ndarray], str] | None = None,
          collision_rows: np.ndarray | None = None) -> None:
    """Refuse a motion outside its position, velocity or acceleration limits, with a jump between samples, not at rest at
    its ends, over its torque limit, or colliding (``collides(q)`` names what it hits, "" when free). Tolerances are the
    round's ruler's (check_scenario.py G3), so whatever passes here passes there."""
    bounds = limits if isinstance(limits, k.Bounds) else k.Bounds.from_limits(limits)
    names = bounds.joint_names
    q, qd, qdd, t = arrays.q, arrays.qd, arrays.qdd, arrays.t
    problems: list[str] = []
    beyond = np.maximum(np.max(bounds.low - q, axis=0), np.max(q - bounds.high, axis=0))
    for joint in np.flatnonzero(beyond > 1e-9):
        problems.append(f"{names[joint]} leaves its range by {math.degrees(beyond[joint]):.4f}°")
    for quantity, values, limit in (("velocity", qd, bounds.max_velocity), ("acceleration", qdd, bounds.max_acceleration)):
        ratio = np.max(np.abs(values), axis=0) / limit
        for joint in np.flatnonzero(ratio > 1 + 1e-6):
            problems.append(f"{names[joint]} {quantity} reaches {ratio[joint]:.4f}x its limit at t = "
                            f"{float(t[int(np.argmax(np.abs(values[:, joint])))]):.3f} s")
    if len(t) > 1:
        dt = np.diff(t)[:, None]
        for quantity, values, limit in (("position", q, bounds.max_velocity), ("velocity", qd, bounds.max_acceleration)):
            jumps = np.abs(np.diff(values, axis=0)) > limit * dt * 1.05 + 1e-9
            if np.any(jumps):
                row, joint = np.argwhere(jumps)[0]
                problems.append(f"{names[joint]} jumps in {quantity} between samples {row} and {row + 1} (t = {float(t[row]):.3f} s)")
    if rest_start and np.max(np.abs(qd[0])) > 1e-9:
        problems.append("does not start at rest")
    if rest_end and np.max(np.abs(qd[-1])) > 1e-9:
        problems.append("does not end at rest")
    max_effort = getattr(limits, "max_effort", None)
    if effort is not None and max_effort is not None:
        ratio = np.nanmax(np.abs(effort), axis=0) / np.asarray(max_effort, float)
        for joint in np.flatnonzero(ratio > 1 + 1e-6):
            problems.append(f"{names[joint]} effort reaches {ratio[joint]:.4f}x its torque limit")
    if problems:
        raise UnsafeTrajectory(f"{what}: " + "; ".join(problems))
    if collides is not None:
        rows = range(len(t)) if collision_rows is None else np.flatnonzero(collision_rows)
        for row in rows:
            hit = collides(q[row])
            if hit:
                raise UnsafeTrajectory(f"{what}: sample {row} (t = {float(t[row]):.3f} s, q = "
                                       f"{np.round(np.degrees(q[row]), 2).tolist()}°) collides with {hit}")


# ====================================================================== reading what a trajectory carries


def _meta(trajectory: JointTrajectory, key: str) -> Any:
    if key not in trajectory.header.meta:
        raise ValueError(f"this JointTrajectory was not made by robot_trajectories (no header.meta[{key!r}])")
    return json.loads(trajectory.header.meta[key])


def recipe_of(trajectory: JointTrajectory) -> dict[str, Any]:
    """The recipe a trajectory carries: ``{"format", "space", "calls"}``."""
    return _meta(trajectory, "recipe")


def segment_info(trajectory: JointTrajectory) -> list[dict[str, Any]]:
    """One dict per segment: label, call, and purpose / wait_after / notes / made_by when set."""
    return _meta(trajectory, "segments")


def labels(trajectory: JointTrajectory) -> list[str]:
    """The segment labels, in order (one per ``split`` piece)."""
    return [entry["label"] for entry in segment_info(trajectory)]


def notes(trajectory: JointTrajectory) -> list[str]:
    """What the kinds reported while generating (a clamped rung, a skipped speed, a fallback), in order."""
    found: list[str] = []
    for entry in segment_info(trajectory):
        found += [note for note in entry.get("notes", ()) if note not in found]
    return found


def _ends(ids: np.ndarray, count: int) -> list[int]:
    return [int(np.flatnonzero(ids == index)[-1]) for index in range(count)]


def bands(trajectory: JointTrajectory) -> list[Band]:
    """Every kernel piece as a labelled stretch of the clock."""
    pieces = _meta(trajectory, "pieces")
    ends = _ends(np.asarray(trajectory.channels[PIECE]), len(pieces))
    t = trajectory.t
    return [Band(entry["segment"], entry["label"], float(t[0 if index == 0 else ends[index - 1]]), float(t[end]))
            for index, (entry, end) in enumerate(zip(pieces, ends, strict=True))]


# ====================================================================== blocks: the unit chain, split and repeat move


@dataclass
class _Block:
    """A stretch of rows that moves as one: its arrays (t from 0), its pieces, its channels and effort, and its segment."""

    arrays: k.Arrays
    piece_labels: list[str]
    piece_of_row: np.ndarray  # (n,) local piece index; row 0 belongs to piece 0
    channels: dict[str, np.ndarray]
    effort: np.ndarray | None
    segment: dict[str, Any]


def _rows(trajectory: JointTrajectory, first: int, last: int, rate: float, label: str) -> k.Arrays:
    count = last + 1 - first
    return k.Arrays(np.arange(count) / rate, trajectory.positions[first:last + 1], trajectory.velocities[first:last + 1],
                    trajectory.accelerations[first:last + 1], label)


def _extras(trajectory: JointTrajectory, first: int, last: int) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    channels = {name: np.asarray(values)[first:last + 1] for name, values in trajectory.channels.items()
                if name not in (SEGMENT, PIECE)}
    effort = None if trajectory.effort is None else np.asarray(trajectory.effort)[first:last + 1]
    return channels, effort


def _segment_blocks(trajectory: JointTrajectory) -> list[_Block]:
    entries = segment_info(trajectory)
    pieces = _meta(trajectory, "pieces")
    rate = recipe_of(trajectory)["space"]["rate_hz"]
    segment_ids = np.asarray(trajectory.channels[SEGMENT])
    piece_ids = np.asarray(trajectory.channels[PIECE])
    ends = _ends(segment_ids, len(entries))
    blocks = []
    for index, entry in enumerate(entries):
        first = 0 if index == 0 else ends[index - 1]
        own = [number for number, piece in enumerate(pieces) if piece["segment"] == index]
        local = piece_ids[first:ends[index] + 1] - own[0]
        local[0] = 0
        channels, effort = _extras(trajectory, first, ends[index])
        arrays = _rows(trajectory, first, ends[index], rate, entry["label"])
        entry = dict(entry)
        own_first = entry.pop("first_sample", None)
        if own_first is not None:
            arrays = _with_first(arrays, own_first)
        blocks.append(_Block(arrays, [pieces[n]["label"] for n in own], local, channels, effort, entry))
    return blocks


def _with_first(arrays: k.Arrays, sample: Mapping[str, Sequence[float]]) -> k.Arrays:
    """``arrays`` with its first row set to a segment's own first sample (see ``first_sample`` in the module docstring)."""
    q, qd, qdd = (np.array(values, float) for values in (arrays.q, arrays.qd, arrays.qdd))
    q[0], qd[0], qdd[0] = sample["q"], sample["qd"], sample["qdd"]
    return k.Arrays(arrays.t, q, qd, qdd, arrays.label)


def _first_sample(block: k.Arrays, previous: k.Arrays) -> dict[str, list[float]] | None:
    """A block's own first sample when it differs bit for bit from the previous block's last, else None."""
    own = (block.q[0], block.qd[0], block.qdd[0])
    shared = (previous.q[-1], previous.qd[-1], previous.qdd[-1])
    if all(np.ascontiguousarray(a, float).tobytes() == np.ascontiguousarray(b, float).tobytes() for a, b in zip(own, shared)):
        return None
    return {"q": [float(v) for v in own[0]], "qd": [float(v) for v in own[1]], "qdd": [float(v) for v in own[2]]}


def _piece_blocks(trajectory: JointTrajectory) -> list[_Block]:
    pieces = _meta(trajectory, "pieces")
    rate = recipe_of(trajectory)["space"]["rate_hz"]
    ends = _ends(np.asarray(trajectory.channels[PIECE]), len(pieces))
    blocks = []
    for index, entry in enumerate(pieces):
        first = 0 if index == 0 else ends[index - 1]
        channels, effort = _extras(trajectory, first, ends[index])
        blocks.append(_Block(_rows(trajectory, first, ends[index], rate, entry["label"]), [entry["label"]],
                             np.zeros(ends[index] + 1 - first, dtype=np.int64), channels, effort, {}))
    return blocks


def _filled(part: np.ndarray | None, count: int, trailing: tuple[int, ...]) -> np.ndarray:
    return np.full((count, *trailing), np.nan) if part is None else np.asarray(part, float)


def assemble(joined: k.Arrays, used: Sequence[k.Arrays], blocks: Mapping[int, _Block], *, limits: JointLimits,
             recipe: Mapping[str, Any], transition_segment: dict[str, Any] | None, segment_entry: dict[str, Any] | None = None,
             stamp: float = 0.0) -> JointTrajectory:
    """A JointTrajectory from ``kernels.join``'s output and the pieces it used. ``blocks`` maps ``id(piece)`` to what a piece
    carries; a piece not in it is a transition ``join`` made: its own segment ``transition_segment`` (chain), or a piece of
    the one segment ``segment_entry`` (a kind)."""
    segments: list[dict[str, Any]] = []
    piece_entries: list[dict[str, Any]] = []
    segment_rows, piece_rows, effort_parts = [], [], []
    channel_parts: dict[str, list[tuple[int, np.ndarray | None]]] = {}
    trailing: dict[str, tuple[int, ...]] = {}
    for index, piece in enumerate(used):
        block = blocks.get(id(piece))
        if block is None:  # a transition kernels.join put in
            block = _Block(piece, [piece.label or "transition"], np.zeros(len(piece.t), dtype=np.int64), {}, None,
                           dict(transition_segment) if transition_segment is not None else {})
        if segment_entry is not None:
            if not segments:
                segments.append(dict(segment_entry))
            segment_index = 0
        else:
            entry = dict(block.segment)
            if index > 0:
                own_first = _first_sample(piece, used[index - 1])
                if own_first is not None:
                    entry["first_sample"] = own_first
            segments.append(entry)
            segment_index = len(segments) - 1
        keep = slice(None) if index == 0 else slice(1, None)
        count = len(piece.t) if index == 0 else len(piece.t) - 1
        offset = len(piece_entries)
        piece_entries += [{"label": label, "segment": segment_index} for label in block.piece_labels]
        piece_rows.append(block.piece_of_row[keep] + offset)
        segment_rows.append(np.full(count, segment_index, dtype=np.int64))
        for name, values in block.channels.items():
            trailing.setdefault(name, np.asarray(values).shape[1:])
        effort_parts.append((count, None if block.effort is None else np.asarray(block.effort)[keep]))
        for name in set(channel_parts) | set(block.channels):
            channel_parts.setdefault(name, [(c, None) for c, _ in effort_parts[:-1]])
        for name, parts in channel_parts.items():
            values = block.channels.get(name)
            parts.append((count, None if values is None else np.asarray(values)[keep]))
    channels = {SEGMENT: np.concatenate(segment_rows), PIECE: np.concatenate(piece_rows)}
    for name, parts in channel_parts.items():
        channels[name] = np.concatenate([_filled(part, count, trailing[name]) for count, part in parts])
    effort = None
    if any(part is not None for _, part in effort_parts):
        effort = np.concatenate([_filled(part, count, (joined.q.shape[1],)) for count, part in effort_parts])
    meta = {"segments": json.dumps(segments), "pieces": json.dumps(piece_entries), "recipe": json.dumps(dict(recipe))}
    return JointTrajectory(t=joined.t, joint_names=tuple(limits.joint_names), positions=joined.q, velocities=joined.qd,
                           accelerations=joined.qdd, effort=effort, limits=limits, channels=channels,
                           header=Header(stamp=stamp, meta=meta))


# ====================================================================== chain and split


def _flatten(items: Sequence[Any]) -> list[JointTrajectory]:
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        items = tuple(items[0])
    for index, item in enumerate(items):
        if not isinstance(item, JointTrajectory):
            raise TypeError(f"chain joins JointTrajectories this package made; item {index} is {type(item).__name__}")
    if not items:
        raise ValueError("chain needs at least one trajectory")
    return list(items)


def chain(*trajectories: JointTrajectory | Sequence[JointTrajectory], space: Any = None,
          transition_speed: float = TRANSITION_SPEED) -> JointTrajectory:
    """Join trajectories on one clock, segments kept. Where one does not start where the last ended, a min-jerk transition
    goes in as its own segment (labelled ``transition``). ``chain(chain(a, b), c)`` has segments a, b, c.

    ``space`` (optional, the one they were made in) also checks the transitions against its collision models."""
    items = _flatten(trajectories)
    records = [recipe_of(item) for item in items]
    first = records[0]["space"]
    for other in records[1:]:
        if other["space"]["fingerprint"] != first["fingerprint"]:
            raise ValueError(f"chain joins trajectories of one space; got {first['source']!r} and {other['space']['source']!r} "
                             "(different limits, home, rate or injected models)")
    if space is not None and space.fingerprint != first["fingerprint"]:
        raise ValueError(f"these trajectories were made in {first['source']!r}, not in {space!r}")
    limits = items[0].limits
    bounds = k.Bounds.from_limits(limits)
    blocks = [block for item in items for block in _segment_blocks(item)]
    _check_boundaries(blocks)
    joined, used = k.join([block.arrays for block in blocks], bounds, first["rate_hz"], transition_speed=transition_speed)
    recipe = {"format": RECIPE_FORMAT, "space": first, "calls": [call for record in records for call in record["calls"]]}
    out = assemble(joined, used, {id(block.arrays): block for block in blocks}, limits=limits, recipe=recipe,
                   transition_segment={"label": "transition", "call": None})
    entries = segment_info(out)
    segment_ids = np.asarray(out.channels[SEGMENT])
    transition_rows = np.isin(segment_ids, [index for index, entry in enumerate(entries) if entry["call"] is None])
    guard(joined, limits, "chain", rest_start=entries[0].get("starts", "at_rest") == "at_rest",
          rest_end=entries[-1].get("ends", "at_rest") == "at_rest", effort=out.effort,
          collides=None if space is None or not space.checks_collision else space.collision_with, collision_rows=transition_rows)
    return out


def _check_boundaries(blocks: Sequence[_Block]) -> None:
    """Where a segment ends moving or the next starts moving, their states must match: position and velocity."""
    for index, (before, after) in enumerate(zip(blocks[:-1], blocks[1:], strict=True), start=1):
        moving = before.segment.get("ends") == "moving" or after.segment.get("starts") == "moving"
        if not moving:
            continue
        position = float(np.max(np.abs(before.arrays.q[-1] - after.arrays.q[0])))
        velocity = float(np.max(np.abs(before.arrays.qd[-1] - after.arrays.qd[0])))
        if position > 1e-9 or velocity > 1e-9:
            raise ValueError(f"chain: segment {index - 1} ({before.segment.get('label')!r}) ends "
                             f"{before.segment.get('ends', 'at_rest').replace('_', ' ')} and segment {index} "
                             f"({after.segment.get('label')!r}) starts {after.segment.get('starts', 'at_rest').replace('_', ' ')}, "
                             f"but their states differ by {position:.3g} rad and {velocity:.3g} rad/s: a moving boundary must hand "
                             "its exact state on (start the next kind at the trajectory before it), and no transition can join it")


def split(trajectory: JointTrajectory) -> list[JointTrajectory]:
    """The segments as rest-to-rest trajectories, transitions included and labelled, each starting at t = 0 with its offset on
    the joined clock in ``header.stamp``. Each carries its own one-call recipe, so it rebuilds alone, and ``chain(*split(x))``
    is ``x`` sample for sample."""
    record = recipe_of(trajectory)
    out = []
    for block in _segment_blocks(trajectory):
        entry = block.segment
        recipe = {"format": RECIPE_FORMAT, "space": record["space"], "calls": [] if entry.get("call") is None else [entry["call"]]}
        local = {id(block.arrays): block}
        piece = assemble(block.arrays, [block.arrays], local, limits=trajectory.limits, recipe=recipe, transition_segment=None)
        out.append(piece)
    # header.stamp: each piece's offset on the joined clock
    offsets = np.concatenate([[0.0], np.cumsum([float(piece.t[-1]) for piece in out])[:-1]])
    return [piece.replace(header=Header(stamp=trajectory.header.stamp + float(offset), meta=dict(piece.header.meta)))
            for piece, offset in zip(out, offsets, strict=True)]


def annotate(trajectory: JointTrajectory, purpose: str | None = None, *, wait_after: bool | None = None) -> JointTrajectory:
    """The same trajectory with every segment saying what it measures (``purpose``) and whether to stop and look after it
    (``wait_after``: split, run one, then wait). Pure: a new trajectory."""
    entries = segment_info(trajectory)
    for entry in entries:
        if purpose is not None:
            entry["purpose"] = purpose
        if wait_after is not None:
            entry["wait_after"] = bool(wait_after)
    return trajectory.replace(header=Header(stamp=trajectory.header.stamp, frame_id=trajectory.header.frame_id,
                                            meta={**trajectory.header.meta, "segments": json.dumps(entries)}))


def with_segments(trajectory: JointTrajectory, update: Callable[[int, dict[str, Any]], dict[str, Any]],
                  *, recipe: Mapping[str, Any] | None = None) -> JointTrajectory:
    """The same trajectory with each segment entry replaced by ``update(index, entry)`` (and the recipe, when given)."""
    entries = [update(index, dict(entry)) for index, entry in enumerate(segment_info(trajectory))]
    meta = {**trajectory.header.meta, "segments": json.dumps(entries)}
    if recipe is not None:
        meta["recipe"] = json.dumps(dict(recipe))
    return trajectory.replace(header=Header(stamp=trajectory.header.stamp, frame_id=trajectory.header.frame_id, meta=meta))


# ====================================================================== repeat


def repeat(trajectory: JointTrajectory, times: int = 5, *, label: str = "repeat") -> JointTrajectory:
    """One tape played ``times`` times with the same input: the run-to-run floor every sim-vs-real gate is a multiple of
    (the pendulum contract: sim vs real <= 1.5 x the same-input repeat floor, 5 repeats).

    One segment, labelled ``label``; each play is a band ``"<label>: k/N: <piece>"``. Every play starts from the tape's start
    at rest: if the tape ends elsewhere, a min-jerk move back to its start (half speed) comes before the next play.

    WHY one segment: the floor is read from all the plays together, so it is one test; the plays are its bands. WHY the
    samples are copied, not regenerated: a pure function of its input. v6's Repeat regenerated each play from where the
    last ended, which differs only when a tape ends within 1e-9 rad of its start but not bit for bit (no ported tape does;
    tests/test_equivalence.py holds the floor tape to v6's)."""
    if times < 1:
        raise ValueError(f"repeat needs times >= 1, got {times}")
    record = recipe_of(trajectory)
    rate = record["space"]["rate_hz"]
    limits = trajectory.limits
    bounds = k.Bounds.from_limits(limits)
    start, end = np.asarray(trajectory.positions[0]), np.asarray(trajectory.positions[-1])
    tape = _piece_blocks(trajectory)
    pieces: list[k.Arrays] = []
    blocks: dict[int, _Block] = {}
    here = start
    for play in range(1, times + 1):
        prefix = f"{label}: {play}/{times}"
        if not np.allclose(here, start, atol=1e-9):
            pieces.append(k.min_jerk(here, start, bounds, rate, speed=TRANSITION_SPEED, label=f"{prefix}: back to start"))
            here = start
        for block in tape:
            arrays = k.Arrays(block.arrays.t, block.arrays.q, block.arrays.qd, block.arrays.qdd, f"{prefix}: {block.arrays.label}")
            pieces.append(arrays)
            blocks[id(arrays)] = _Block(arrays, [arrays.label], block.piece_of_row, block.channels, block.effort, {})
        here = end
    joined, used = k.join(pieces, bounds, rate, transition_speed=TRANSITION_SPEED)
    call = {"kind": "repeat", "times": int(times), "label": label, "of": record["calls"]}
    recipe = {"format": RECIPE_FORMAT, "space": record["space"], "calls": [call]}
    out = assemble(joined, used, blocks, limits=limits, recipe=recipe, transition_segment=None,
                   segment_entry={"label": label, "call": call})
    guard(joined, limits, f"repeat ({label})", effort=out.effort)
    return out


__all__ = ["Band", "Motion", "Piece", "TOOL_TARGET", "UnsafeTrajectory", "annotate", "bands", "chain", "guard", "labels", "notes",
           "recipe_of", "repeat", "segment_info", "split"]
