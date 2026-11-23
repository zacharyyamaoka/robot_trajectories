from __future__ import annotations

import random

import numpy as np

from .. import kernels as k
from ..registry import kind
from ..segments import Motion
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator

CLASSES = ("pickplace", "cycle", "jog", "sweep")  # PID tuna's tapes.evaluation class names, kept as its data
TRAIN_SEEDS = (0, 1, 2)  # the catalog's train_seeds
HELD_OUT_SEEDS = (3, 4, 5)  # the catalog's held_out_seeds: score on them, never pick gains on them


def legs(motion_class: str, rng: random.Random) -> list[tuple[float, float, float]]:
    """(target as a fraction of the room, peak speed fraction, dwell s): PID tuna's draws, in its order."""
    drawn: list[tuple[float, float, float]] = []
    previous = 0.0
    if motion_class == "pickplace":  # large moves, long holds
        for _ in range(4):
            target = rng.uniform(0.4, 0.95) * rng.choice((-1, 1))
            drawn.append((target, rng.uniform(0.6, 1.0), round(rng.uniform(0.6, 1.0), 2)))
            previous = target
    elif motion_class == "cycle":  # one pair out and back x3
        out = rng.uniform(0.4, 0.8) * rng.choice((-1, 1))
        for _ in range(3):
            drawn += [(out, 0.8, 0.5), (0.0, 0.8, 0.5)]
    elif motion_class == "jog":  # small corrections
        for _ in range(6):
            target = previous + rng.uniform(0.05, 0.15) * rng.choice((-1, 1))
            target = max(-0.9, min(0.9, target))
            drawn.append((target, 0.4, 0.6))
            previous = target
    else:  # sweep: one slow pass
        drawn.append((0.9 * rng.choice((-1, 1)), 0.25, 0.8))
    if abs(drawn[-1][0]) > 1e-12:
        drawn.append((0.0, 0.8, 0.5))  # WHY explicit: PID tuna's way home, never the runner's reused last move time
    return drawn


@kind(use_cases=("tracking_eval", "stop_accuracy", "pid_tuning"), catalog_id="deployment_motion_classes")
def deployment_motion_classes(space, joint=None, *, classes=CLASSES, seeds=TRAIN_SEEDS, min_move_s=0.15,
                              label="deployment motion classes", start=None):
    """Seeded random tapes in four motion classes: pickplace (large moves, long holds), cycle (one pair out and back x3), jog
    (small corrections) and sweep (one slow pass); ``seeds=HELD_OUT_SEEDS`` (3, 4, 5) is the sealed held-out set (PID tuna's
    ``tapes.evaluation``: Zach's "send it random commands and lets see the performance").

    Faithful to PID tuna: the same draws in the same order from ``random.Random(seed)`` (targets as fractions of the room,
    each leg's speed as a fraction of top speed, the holds), every leg a minimum-jerk WaypointTrajectoryGenerator at that speed, and an explicit
    way home at 0.8 x speed. A seed here is that seed's tape there, scaled to this joint's room. Inside the limits: a leg the
    acceleration limit cannot take at its drawn speed runs as fast as the limit allows (and the notes count them)."""
    unknown = [name for name in classes if name not in CLASSES]
    if unknown or not classes:
        raise ValueError(f"deployment_motion_classes.classes must come from {CLASSES}, got {classes}")
    if not seeds:
        raise ValueError("deployment_motion_classes needs at least one seed")
    joint = space.joint_index(joint)
    room = space.room(start)
    travel = float(min(start[joint] - room.low[joint], room.high[joint] - start[joint]))
    pieces: list[k.Arrays] = []
    here = start.copy()
    clamped = total = 0
    for seed in seeds:
        rng = random.Random(seed)  # WHY one generator per seed across the classes: PID tuna's draw order
        for motion_class in classes:
            name = f"{label}: seed {seed} {motion_class}"
            for target, speed_fraction, dwell_s in legs(motion_class, rng):
                goal = start.copy()
                goal[joint] += target * travel
                law = WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=speed_fraction, min_move_s=min_move_s)
                if not np.allclose(goal, here):
                    clamped += law.leg_timing(here, goal, space.limits).bound == "acceleration"
                    total += 1
                    pieces.append(law.leg(here, goal, space.limits, space.rate_hz, label=name))
                pieces.append(k.hold(goal, dwell_s, space.rate_hz, label=f"{name}: hold"))
                here = goal
    notes = [f"{label}: {clamped} of {total} legs run slower than drawn (the acceleration limit binds)"] if clamped else []
    return Motion(pieces, notes)
