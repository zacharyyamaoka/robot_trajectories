from __future__ import annotations

import math

import numpy as np

from ..registry import LIMITS, kind
from ..segments import Motion
from . import _lane_b as b


@kind(use_cases=("inertia_id", "friction_id"), needs=(*LIMITS, "torque_limits", "dynamics_model"), catalog_id="torque_steps",
      command="effort", ends="anywhere",
      until="the last step's hold has played (hold_s each) and the joint has rung down: |qd| < 0.01 rad/s for 0.5 s")
def torque_steps(space, joint=None, *, step_torque_fraction_of_gravity=0.25, hold_s=1.0, steps=4, label="torque steps", start=None):
    """Open-loop torque steps (kp = kd = 0) on top of the torque that holds the start pose: a rest hold, then ``steps`` steps
    alternating sign, each back to the hold before the next: torque-to-acceleration gain and the friction band with no position
    loop.

    Sized like ``torque_chirp`` (J288's 0.5 G steps would leave CAN 11's window, Servo Test Schedule note F9): the catalog's
    ``min(fraction x G, G sin(0.85 x range))``, then solved down to what the prediction keeps inside the limits."""
    if steps < 1 or hold_s <= 0:
        raise ValueError(f"torque_steps needs steps >= 1 and hold_s > 0, got {steps}, {hold_s}")
    j = space.joint_index(joint)
    hold = b.hold_effort(space, start)
    headroom = b.torque_headroom(space, j, hold)
    gravity = b.gravity_scale(space, j, start)
    if gravity <= 0.0:
        raise ValueError("torque_steps is for a pendulum: the model holds no gravity torque at this pose")
    wanted = min(step_torque_fraction_of_gravity * gravity, gravity * math.sin(0.85 * b.two_sided_room(space, j, start)))
    amplitude = min(wanted, headroom)
    per = int(round(hold_s * space.rate_hz))
    levels = [0.0]
    for index in range(steps):
        levels += [(1.0 if index % 2 == 0 else -1.0), 0.0]
    shape = np.repeat(levels, per)
    shape = np.append(shape, shape[-1])
    t = np.arange(len(shape)) / space.rate_hz

    def build(gain: float) -> np.ndarray:
        effort = np.tile(hold, (len(t), 1))
        effort[:, j] += gain * amplitude * shape
        return effort

    gain, effort, motion = b.solve_gain(space, j, start, build)
    notes = [f"step torque {amplitude * gain:.3f} N.m, {steps} steps alternating sign, {hold_s:g} s each (G = {gravity:.3f} N.m)"]
    if amplitude < wanted:
        notes.append(f"cut to {amplitude:.3f} N.m by the drive's torque limit")
    if gain < 1.0:
        notes.append(f"scaled to {gain:.3f} of that: the predicted swing would pass 90 % of the joint's range, speed or acceleration limit")
    return Motion([b.effort_piece(space, t, effort, start, label, motion=motion)], notes,
                  {"step_torque_n_m": float(amplitude * gain), "hold_torque_n_m": [float(v) for v in hold]})
