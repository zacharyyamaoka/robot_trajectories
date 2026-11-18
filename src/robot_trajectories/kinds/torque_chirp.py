from __future__ import annotations

import math

import numpy as np

from ..registry import LIMITS, kind
from ..segments import Motion
from . import _lane_b as b


@kind(use_cases=("inertia_id", "friction_id", "delay_id", "bandwidth"), needs=(*LIMITS, "torque_limits", "dynamics_model"),
      catalog_id="torque_chirp", command="effort", ends="anywhere",
      until="the chirp has played (duration_s, 15 s by default) and the joint has rung down: |qd| < 0.01 rad/s for 0.5 s")
def torque_chirp(space, joint=None, *, f0_hz=0.5, f1_hz=6.0, duration_s=15.0, torque_fraction_of_gravity=0.25, fade_s=1.0,
                 settle_s=0.5, label="torque chirp", start=None):
    """With kp = kd = 0: the torque that holds the start pose plus a faded linear torque chirp, the plant with nothing else in
    the loop. The command is the effort; the positions are the model's prediction of it.

    The amplitude is the catalog's, ``min(fraction x G, G sin(0.85 x range))`` (G read from the dynamics model, a pendulum's
    gravity torque), then solved down to the largest the prediction can take inside the joint's position, speed and acceleration
    limits and the drive's torque limit (said in the notes, never silent). It ends wherever the joint rings out
    (``ends="anywhere"``): a pendulum only, gravity is what brings it back."""
    if not (0 < f0_hz < f1_hz) or duration_s <= 0 or fade_s < 0 or settle_s < 0:
        raise ValueError(f"torque_chirp needs 0 < f0_hz < f1_hz, duration_s > 0, fade_s and settle_s >= 0; got {f0_hz}, {f1_hz}, "
                         f"{duration_s}, {fade_s}, {settle_s}")
    j = space.joint_index(joint)
    hold = b.hold_effort(space, start)
    headroom = b.torque_headroom(space, j, hold)
    gravity = b.gravity_scale(space, j, start)
    if gravity <= 0.0:
        raise ValueError("torque_chirp is for a pendulum: the model holds no gravity torque at this pose, so nothing brings the "
                         "joint back")
    # WHY the catalog's amplitude (the 11-18 J288 PR: 0.25 G "because the chirp sweeps through the pendulum's natural frequency"):
    # the static deflection stays inside the window; the rack's runner sizes its chirp the same way
    wanted = min(torque_fraction_of_gravity * gravity, gravity * math.sin(0.85 * b.two_sided_room(space, j, start)))
    amplitude = min(wanted, headroom)
    t = b.sample_times(duration_s + settle_s, space.rate_hz)
    window = np.where(t <= duration_s, b.fade(t, duration_s, fade_s), 0.0)
    phase = 2 * np.pi * (f0_hz * t + (f1_hz - f0_hz) * np.minimum(t, duration_s) ** 2 / (2 * duration_s))

    def build(gain: float) -> np.ndarray:
        effort = np.tile(hold, (len(t), 1))
        effort[:, j] += gain * amplitude * window * np.sin(phase)
        return effort

    gain, effort, motion = b.solve_gain(space, j, start, build)
    notes = [f"torque amplitude {amplitude * gain:.3f} N.m (the catalog's {wanted:.3f} N.m = min({torque_fraction_of_gravity:g} G, "
             f"G sin(0.85 x range)) with G = {gravity:.3f} N.m)"]
    if amplitude < wanted:
        notes.append(f"cut to {amplitude:.3f} N.m: the drive's {float(np.asarray(space.limits.max_effort)[j]):.3f} N.m limit less the "
                     f"{abs(float(hold[j])):.3f} N.m that holds the pose")
    if gain < 1.0:
        notes.append(f"scaled to {gain:.3f} of that: the predicted motion would pass 90 % of the joint's range, speed or acceleration limit")
    return Motion([b.effort_piece(space, t, effort, start, label, motion=motion)], notes,
                  {"torque_amplitude_n_m": float(amplitude * gain), "hold_torque_n_m": [float(v) for v in hold]})
