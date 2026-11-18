"""The round's scenario written against the bare kernels (no API): the spec every proposal's scenario reproduced.

Ported from bam_ws ``docs/trajectory_generator/reference/scenario.py`` (b5e1f6965), reading this package's vendored
``robot_trajectories.kernels`` (byte-identical to the round's). ``tests/test_selfcheck.py`` holds the package's pendulum and
UR5e sessions equal to these, sample for sample. Dropped: the round's ``run()`` / ``rebuild()`` (the ruler's entry points for
the reference itself) and its ``mini_catalog`` import.

1. pendulum (bare JointLimits): step train, hold 2 s, log chirp 0.2-5 Hz 20 s, hold 1 s, velocity sweep, 8 random
   waypoints (seed 1), home; one joined trajectory AND the same as rest-to-rest segments
2. five-bar (reach map + limits): a neighbour-only tour of the reachable set on a 12 deg grid, then a multisine inside the
   largest box around the start that holds no unreachable configuration
3. UR5e (description; its URDF has no acceleration, so 6 rad/s^2 is supplied, and a demo box above the table):
   multisine 0.1-1 Hz on all six joints, then a 3 x 3 x 3 grid over the first three joints
"""

from __future__ import annotations

import numpy as np

from fixtures import robots
from robot_trajectories import kernels as k

RATE_HZ = 500.0
PENDULUM_CONFIG = {
    "robot": "pendulum", "rate_hz": RATE_HZ,
    "segments": [
        {"kind": "step_train", "joint": "can11", "steps_fraction": [0.5, -0.5, 0.25, -0.25, 0.9], "dwell_s": 1.5},
        {"kind": "hold", "duration_s": 2.0},
        {"kind": "chirp", "joint": "can11", "start_hz": 0.2, "end_hz": 5.0, "duration_s": 20.0, "logarithmic": True},
        {"kind": "hold", "duration_s": 1.0},
        {"kind": "velocity_sweep", "joint": "can11", "speeds_fraction": [0.1, 0.3, 0.6]},
        {"kind": "random_waypoints", "count": 8, "seed": 1},
        {"kind": "home"},
    ],
}


def pendulum_session(config: dict):
    limits = robots.PENDULUM_LIMITS
    bounds = k.Bounds.from_limits(limits)
    centre = np.zeros(1)
    rate = config["rate_hz"]
    here = centre
    segments, labels = [], []
    for spec in config["segments"]:
        kind = spec["kind"]
        if kind == "step_train":
            pieces = k.step_train(here, 0, bounds, rate, steps_fraction=tuple(spec["steps_fraction"]), dwell_s=spec["dwell_s"])
        elif kind == "hold":
            pieces = [k.hold(here, spec["duration_s"], rate)]
        elif kind == "chirp":
            pieces = [k.chirp(here, 0, bounds, rate, start_hz=spec["start_hz"], end_hz=spec["end_hz"], duration_s=spec["duration_s"],
                              logarithmic=spec["logarithmic"])]
        elif kind == "velocity_sweep":
            pieces = k.velocity_sweep(here, 0, bounds, rate, speeds_fraction=tuple(spec["speeds_fraction"]))
        elif kind == "random_waypoints":
            pieces = k.waypoint_chain(k.random_waypoints(bounds, spec["count"], seed=spec["seed"]), bounds, rate, start=here)
        elif kind == "home":
            pieces = [k.min_jerk(here, centre, bounds, rate, speed=0.5, label="home")]
        else:
            raise ValueError(kind)
        arrays, _ = k.join(pieces, bounds, rate)
        here = arrays.q[-1]
        segments.append(k.to_trajectory(arrays, limits, limits.joint_names))
        labels.append(kind.replace("_", " "))
    joined_arrays, _ = k.join([k.Arrays(s.t - s.t[0], s.positions, s.velocities, s.accelerations) for s in segments], bounds, rate)
    return k.to_trajectory(joined_arrays, limits, limits.joint_names), segments, labels


def fivebar_sweep():
    reach = robots.fivebar_reach_map(step_deg=4.0)
    solutions, valid = reach.joint_solutions[:, 0, :], reach.valid_mask[:, 0]
    keys = np.round((solutions - solutions.min(0)) / np.radians(4.0)).astype(int)
    coarse = solutions[valid & (keys[:, 0] % 3 == 0) & (keys[:, 1] % 3 == 0)]
    bounds = k.Bounds.from_limits(robots.FIVEBAR_LIMITS)
    start = coarse[len(coarse) // 2]
    tour = k.reachable_tour(coarse, np.radians(12.0), start)
    box = k.inscribed_box(start, solutions[~valid], bounds)
    pieces = [*k.waypoint_chain(tour, bounds, RATE_HZ, start=start, speed=0.8, dwell_s=0.05, label="workspace sweep"),
              k.min_jerk(tour[-1], start, bounds, RATE_HZ, speed=0.5, label="to centre"),
              k.multisine(start, [0, 1], box, RATE_HZ)]
    joined, _ = k.join(pieces, bounds, RATE_HZ)
    return k.to_trajectory(joined, robots.FIVEBAR_LIMITS, robots.FIVEBAR_LIMITS.joint_names)


def ur5e_excitation():
    description_limits = robots.UR5E_DESCRIPTION.joint_limits()
    bounds = k.Bounds.from_limits(robots.UR5E_DEMO_BOX)  # positions tightened to the demo box, acceleration supplied
    assert np.allclose(bounds.max_velocity, description_limits.max_velocity)
    centre = bounds.centre
    grid = k.serpentine_grid(bounds, 3, joints=[0, 1, 2], rest=centre)
    pieces = [k.multisine(centre, list(range(6)), bounds, RATE_HZ, band_hz=(0.1, 1.0), period_s=20.0, periods=1), *k.waypoint_chain(grid, bounds, RATE_HZ, start=centre, label="grid"),
              k.min_jerk(grid[-1], centre, bounds, RATE_HZ, speed=0.5, label="home")]
    joined, _ = k.join(pieces, bounds, RATE_HZ)
    return k.to_trajectory(joined, robots.UR5E_DEMO_BOX, robots.UR5E_DEMO_BOX.joint_names)
