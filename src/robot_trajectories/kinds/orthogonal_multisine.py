from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..registry import kind
from ..segments import Motion
from ._lane_a import joint_room


def _relative_peak_factor(position: np.ndarray) -> float:
    """Morelli's relative peak factor: (max - min) / (2 sqrt(2) rms): 1 for a sine, small for a flat-crest multisine."""
    return float((position.max() - position.min()) / (2 * np.sqrt(2) * max(float(np.sqrt(np.mean(position**2))), 1e-300)))


def _phases(harmonics: np.ndarray, period_s: float, trials: int, rng: np.random.Generator) -> np.ndarray:
    """The phases for one joint's lines with the lowest relative peak factor of position among Schroeder's and ``trials`` seeded
    random sets (Morelli 2003: phases chosen to keep the peak factor, hence the amplitude the limits allow, as low as possible)."""
    count = len(harmonics)
    t = np.arange(max(256, 16 * int(harmonics.max()))) * (period_s / max(256, 16 * int(harmonics.max())))
    sines = np.sin(2 * np.pi * np.outer(t, harmonics) / period_s)
    cosines = np.cos(2 * np.pi * np.outer(t, harmonics) / period_s)
    amplitude = period_s / (2 * np.pi * harmonics)  # equal velocity per line: position amplitude 1 / w (scaled by period_s only)

    def peak(phi: np.ndarray) -> float:
        return _relative_peak_factor(sines @ (amplitude * np.cos(phi)) + cosines @ (amplitude * np.sin(phi)))

    best = -np.pi * np.arange(count) * (np.arange(count) + 1) / max(count, 1)  # Schroeder's
    best_peak = peak(best)
    for _ in range(trials):
        candidate = rng.uniform(0, 2 * np.pi, count)
        value = peak(candidate)
        if value < best_peak:
            best, best_peak = candidate, value
    return best


@kind(use_cases=("full_dynamics_id", "bandwidth", "state_coverage"), unchecked=("collision_model",), catalog_id="orthogonal_multisine")
def orthogonal_multisine(space, joints=None, *, period_s=10.0, periods=5, band_low_hz=0.3, band_high_hz=4.0, velocity_rms_rad_s=0.5,
                         seed=0, phase_trials=200, fade_s=1.0, margin=0.8, label="orthogonal multisine", start=None):
    """Every joint (``joints``, default all) gets its OWN disjoint set of harmonics of one fundamental (1 / ``period_s``), so all
    joints move together and each joint's response still separates in frequency (Morelli's orthogonal phase-optimised multisines).
    Line k of the band goes to joint k mod J, so every joint's lines span the whole band. Each line carries the same speed, the
    joint's speed rms is ``velocity_rms_rad_s``; ``periods`` periods, faded in and out.

    The limits can lower a joint's rms (``margin`` of the tightest of room, velocity and acceleration); a note says which.
    When the space carries a collision checker the guard checks every sample against it (it is not required: the catalog's
    ``collision_model`` need is declared unchecked, so this kind runs without one).

    WHY the phases are searched, seeded: for a given set of lines the amplitude is bounded by the signal's peak, so the set of
    phases with the lowest relative peak factor among Schroeder's and ``phase_trials`` random ones (``seed``) lets the same limits
    carry more rms (Morelli's criterion)."""
    if period_s <= 0 or periods < 1 or not 0 < band_low_hz < band_high_hz or velocity_rms_rad_s <= 0:
        raise ValueError("orthogonal_multisine needs period_s > 0, periods >= 1, 0 < band_low_hz < band_high_hz and velocity_rms_rad_s > 0")
    indices = space.joint_indices(joints)
    base = 1.0 / period_s
    harmonics = np.array([h for h in range(1, int(band_high_hz / base) + 1) if band_low_hz <= h * base <= band_high_hz])
    if len(harmonics) < len(indices):
        raise ValueError(f"orthogonal_multisine: the band {band_low_hz:g}-{band_high_hz:g} Hz holds {len(harmonics)} lines of "
                         f"{base:g} Hz for {len(indices)} joints: widen the band or lengthen period_s")
    centre = np.asarray(start, float)
    bounds = space.room(start)
    t = k._samples(period_s * periods, space.rate_hz)
    w, dw, ddw = k._fade(t, t[-1], fade_s)
    rng = np.random.default_rng(seed)
    q, qd, qdd = np.tile(centre, (len(t), 1)), np.zeros((len(t), len(centre))), np.zeros((len(t), len(centre)))
    notes, lines = [], {}
    for column, joint in enumerate(indices):
        own = harmonics[column::len(indices)]
        lines[space.joint_names[joint]] = [int(h) for h in own]
        phi = _phases(own, period_s, phase_trials, rng)
        omega = 2 * np.pi * own * base
        speed = velocity_rms_rad_s * np.sqrt(2.0 / len(own))  # per line, so the joint's speed rms is the asked one
        x = sum((speed / w_k) * np.sin(w_k * t + p) for w_k, p in zip(omega, phi, strict=True))
        dx = sum(speed * np.cos(w_k * t + p) for w_k, p in zip(omega, phi, strict=True))
        ddx = sum(-speed * w_k * np.sin(w_k * t + p) for w_k, p in zip(omega, phi, strict=True))
        x, dx, ddx = x * w, dx * w + x * dw, ddx * w + 2 * dx * dw + x * ddw
        ratios = (float(np.max(np.abs(x))) / max(joint_room(centre, bounds, joint), 1e-12),
                  float(np.max(np.abs(dx))) / float(bounds.max_velocity[joint]),
                  float(np.max(np.abs(ddx))) / float(bounds.max_acceleration[joint]))
        gain = min(1.0, margin / max(ratios))
        q[:, joint] += gain * x
        qd[:, joint] = gain * dx
        qdd[:, joint] = gain * ddx
        notes.append(f"{label}: {space.joint_names[joint]} on {len(own)} lines ({own[0] * base:g}..{own[-1] * base:g} Hz), speed rms "
                     f"{gain * velocity_rms_rad_s:.3g} rad/s" + (f" (asked {velocity_rms_rad_s:g}: x{gain:.3f} to stay inside the limits)"
                                                                 if gain < 1.0 else ""))
    return Motion([k.Arrays(t, q, qd, qdd, label)], notes, entry={"lines": lines, "fundamental_hz": base})
