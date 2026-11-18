from __future__ import annotations

import dataclasses
import random
from typing import ClassVar

import numpy as np

from .. import kernels as k

from ..kind import Context, Kind, param
from ..time_law import TimeLaw

CLASSES = ("pickplace", "cycle", "jog", "sweep")


class DeploymentMotionClasses(Kind):
    """Seeded random tapes in four deployment classes: pickplace (large moves, long holds), cycle (one pair out and back
    x3), jog (small corrections) and sweep (one slow pass); other seeds are the sealed held-out set (PID tuna's
    ``tapes.evaluation``: Zach's "send it random commands and lets see the performance").

    Faithful to PID tuna: the same draws in the same order from ``random.Random(seed)`` (targets as fractions of the room,
    each leg's speed as a fraction of top speed, the holds), every leg a minimum-jerk TimeLaw at that speed, and an
    explicit way home at 0.8 x speed. A seed here is that seed's tape there, scaled to this joint's room. Inside the limits:
    a leg the acceleration limit cannot take at its drawn speed runs as fast as the limit allows."""

    catalog_id = "deployment_motion_classes"
    family = "task_cycle"
    starts_as = "path"
    use_cases = ("tracking_eval", "stop_accuracy", "pid_tuning")
    needs = ("joint_limits", "velocity_limits", "acceleration_limits")
    HELD_OUT_SEEDS: ClassVar[tuple[int, ...]] = (3, 4, 5)  # the catalog's held_out_seeds: never used to pick gains

    joint: str | int | None = param(None, "joint", "the joint that moves (name or index; optional on a 1-DOF robot)")
    classes: tuple[str, ...] = param(CLASSES, "text", "which classes, in order (each seed draws them in this order)")
    seeds: tuple[int, ...] = param((0, 1, 2), "seed", "the seeds played (the catalog's train_seeds; .held_out() for 3, 4, 5)")
    min_move_s: float = param(0.15, "s", "the shortest leg (PID tuna's _move_s floor)")

    def check(self) -> None:
        unknown = [name for name in self.classes if name not in CLASSES]
        if unknown or not self.classes:
            raise ValueError(f"DeploymentMotionClasses.classes must come from {CLASSES}, got {self.classes}")
        if not self.seeds:
            raise ValueError("DeploymentMotionClasses needs at least one seed")

    def held_out(self) -> DeploymentMotionClasses:
        """The same record on the sealed seeds: score on it, never pick on it."""
        return dataclasses.replace(self, seeds=self.HELD_OUT_SEEDS, label=f"{self.label or 'deployment classes'}, held out")

    def legs(self, motion_class: str, rng: random.Random) -> list[tuple[float, float, float]]:
        """(target as a fraction of the room, peak speed fraction, dwell s): PID tuna's draws, in its order."""
        legs: list[tuple[float, float, float]] = []
        previous = 0.0
        if motion_class == "pickplace":
            for _ in range(4):
                target = rng.uniform(0.4, 0.95) * rng.choice((-1, 1))
                legs.append((target, rng.uniform(0.6, 1.0), round(rng.uniform(0.6, 1.0), 2)))
                previous = target
        elif motion_class == "cycle":
            out = rng.uniform(0.4, 0.8) * rng.choice((-1, 1))
            for _ in range(3):
                legs += [(out, 0.8, 0.5), (0.0, 0.8, 0.5)]
        elif motion_class == "jog":
            for _ in range(6):
                target = previous + rng.uniform(0.05, 0.15) * rng.choice((-1, 1))
                target = max(-0.9, min(0.9, target))
                legs.append((target, 0.4, 0.6))
                previous = target
        else:  # sweep: one slow pass
            legs.append((0.9 * rng.choice((-1, 1)), 0.25, 0.8))
        if abs(legs[-1][0]) > 1e-12:
            legs.append((0.0, 0.8, 0.5))  # WHY explicit: PID tuna's way home, never the runner's reused last move time
        return legs

    def pieces(self, context: Context) -> list[k.Arrays]:
        joint = context.joint(self.joint)
        room = context.room_around(context.start)
        travel = float(min(context.start[joint] - room.low[joint], room.high[joint] - context.start[joint]))
        pieces: list[k.Arrays] = []
        here = context.start.copy()
        clamped = 0
        total = 0
        for seed in self.seeds:
            rng = random.Random(seed)  # WHY one generator per seed across the classes: PID tuna's draw order
            for motion_class in self.classes:
                label = f"{self.display_label}: seed {seed} {motion_class}"
                for target, speed_fraction, dwell_s in self.legs(motion_class, rng):
                    goal = context.start.copy()
                    goal[joint] += target * travel
                    law = TimeLaw.minimum_jerk(peak_speed_fraction=speed_fraction, min_move_s=self.min_move_s)
                    if not np.allclose(goal, here):
                        clamped += law.leg_timing(here, goal, context.bounds).bound == "acceleration"
                        total += 1
                        pieces.append(law.leg(here, goal, context.bounds, context.rate_hz, label=label))
                    pieces.append(k.hold(goal, dwell_s, context.rate_hz, label=f"{label}: hold"))
                    here = goal
        if clamped:
            context.note(f"{self.display_label}: {clamped} of {total} legs run slower than drawn (the acceleration limit binds)")
        return pieces
