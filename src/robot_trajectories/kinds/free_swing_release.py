from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..registry import LIMITS, kind
from ..segments import Motion, Piece
from . import _lane_b as b

SETTLE_BEFORE_RELEASE_S = 0.5


@kind(use_cases=("inertia_id", "friction_id", "gravity_id"), needs=LIMITS, catalog_id="free_swing_release", ends="anywhere",
      better_with={"dynamics_model": "the plan ends at the release; the swing that follows is not predicted"},
      until="released: kp = kd = release_kd after the lift has held 0.5 s; coasts coast_s (8 s by default), or stop when "
            "|qd| < 0.01 rad/s for 1 s")
def free_swing_release(space, joint=None, *, release_fraction_of_side=0.6, lift_speed_fraction=0.1, coast_s=8.0, release_kd=0.0,
                       label="free swing", start=None):
    """Lift the joint under its position loop to a fraction of its + side, hold, then release it (kp = kd = 0, or kd only) and
    let it coast: inertia, damping and Coulomb friction with no controller in the loop.

    The motion is the lift (a minimum-jerk leg at ``lift_speed_fraction`` of top speed, as BAM's ``lift_and_drop`` lifts at
    0.1 x speed), a short settle, and, when the space's dynamics can predict it (``forward``), the coast. It ends wherever the
    joint comes to rest (``ends="anywhere"``): the next segment starts from the measured pose. If the predicted swing would
    leave the joint's limits the lift is shortened until it does not (said in the notes)."""
    if not 0 < release_fraction_of_side <= 1 or lift_speed_fraction <= 0 or coast_s <= 0 or release_kd < 0:
        raise ValueError("free_swing_release needs 0 < release_fraction_of_side <= 1, lift_speed_fraction > 0, coast_s > 0, "
                         f"release_kd >= 0; got {release_fraction_of_side}, {lift_speed_fraction}, {coast_s}, {release_kd}")
    j = space.joint_index(joint)
    bounds = space.bounds
    side = float(bounds.high[j] - start[j])
    if side <= 1e-6:
        raise ValueError(f"free_swing_release lifts on the + side: {space.joint_names[j]} has no room above its start")
    predicts = space.dynamics is not None and callable(getattr(space.dynamics, "forward", None))
    samples = int(round(coast_s * space.rate_hz)) + 1

    def coast(release_at: np.ndarray):
        hold = b.hold_effort(space, release_at)

        def effort_at(index, q, qd):
            tau = hold.copy()
            tau[j] = -release_kd * qd[j]  # torque off (kp = 0), or kd only
            return tau

        return b.predict_until(space, release_at, effort_at, samples)

    fraction, notes = release_fraction_of_side, []
    swing = None
    if predicts:
        # WHY a shrinking lift: a pendulum released high enough exceeds its own acceleration limit (gravity alone); the guard
        # would refuse it, so the lift is the highest of a x0.85 ladder that the prediction keeps inside the limits
        for _ in range(40):
            q_release = start.copy()
            q_release[j] += fraction * side
            swing = coast(q_release)
            if b.within_margin(space, j, start, swing[0], swing[1], swing[2], 1.0 - 1e-3):
                break
            fraction *= 0.85
        else:
            raise ValueError("free_swing_release: no lift height keeps the predicted swing inside the joint's limits")
        if fraction < release_fraction_of_side:
            notes.append(f"lift shortened from {release_fraction_of_side:g} to {fraction:.3f} of the + side: the predicted swing would "
                         "pass the joint's position, speed or acceleration limit")
    target = start.copy()
    target[j] += fraction * side
    lift_speed = lift_speed_fraction * float(bounds.max_velocity[j])
    lift = k.min_jerk(start, target, bounds, space.rate_hz, duration_s=k.MIN_JERK_PEAK_VELOCITY * (target[j] - start[j]) / lift_speed,
                      label=f"{label}: lift")
    pieces = [lift, k.hold(target, SETTLE_BEFORE_RELEASE_S, space.rate_hz, label=f"{label}: hold before release")]
    entry = {"release": {"joint": space.joint_names[j], "at_rad": float(target[j]), "kp": 0.0, "kd": float(release_kd),
                         "coast_s": float(coast_s)}}
    if predicts:
        q, qd, qdd, effort = swing
        pieces.append(Piece(np.arange(len(q)) / space.rate_hz, q, qd, qdd, f"{label}: released, coasting", effort=effort))
        notes.append("the coast is the model's prediction with the torque off (kd only if release_kd > 0); the measured swing replaces it")
    return Motion(pieces, notes, entry)
