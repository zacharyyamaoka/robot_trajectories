from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .. import kernels as k
from ..registry import kind

GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0
COMMENSURATE_TOLERANCE = 1e-3
SMALL_DENOMINATOR = 12


def _frequencies(frequencies_hz, count: int) -> np.ndarray:
    """One frequency per joint: the ones given, extended past them by golden-ratio multiples (the most irrational ratio)."""
    given = [float(f) for f in frequencies_hz]
    if not given or any(f <= 0 for f in given):
        raise ValueError(f"lissajous_multi_joint.frequencies_hz are positive, got {tuple(frequencies_hz)}")
    return np.array([given[joint] if joint < len(given) else given[joint % len(given)] * GOLDEN ** (1 + joint // len(given))
                     for joint in range(count)])


@kind(use_cases=("state_coverage", "workspace_coverage", "full_dynamics_id"), unchecked=("collision_model",),
      catalog_id="lissajous_multi_joint")
def lissajous_multi_joint(space, joints=None, *, frequencies_hz=(0.13, 0.17, 0.23), amplitude_fraction=0.8, duration_s=60.0,
                          fade_s=2.0, label="lissajous", start=None):
    """Each joint of ``joints`` (default all) a sine at its own frequency about where it starts, faded in and out so it starts and
    ends at rest at the start: the frequencies are near-incommensurate, so the joint-space curve fills the room densely with
    smooth periodic motion (a Lissajous curve; a multisine with one line per joint, so it inherits that analysis).

    ``frequencies_hz``: one per joint, extended past the given ones by golden-ratio multiples; a pair whose ratio is a small
    rational (p / q, q <= 12, within 0.1 %) is refused, as it would close its curve in a few cycles. Phases are spaced by the golden
    angle. ``amplitude_fraction``: each joint's amplitude is that fraction of the tightest of its room, velocity and acceleration
    limits (as ``multisine``'s margin), so no joint is driven past a limit however the frequencies fall."""
    if not 0.0 < amplitude_fraction <= 1.0:
        raise ValueError(f"lissajous_multi_joint.amplitude_fraction must lie in (0, 1], got {amplitude_fraction}")
    if duration_s <= 2.0 * fade_s or fade_s < 0:
        raise ValueError(f"lissajous_multi_joint needs duration_s > 2 fade_s >= 0, got {duration_s} s and {fade_s} s")
    chosen = space.joint_indices(joints)
    freqs = _frequencies(frequencies_hz, len(chosen))
    for a in range(len(freqs)):
        for b in range(a + 1, len(freqs)):
            ratio = freqs[a] / freqs[b]
            near = Fraction(ratio).limit_denominator(SMALL_DENOMINATOR)
            if abs(ratio - float(near)) <= COMMENSURATE_TOLERANCE * ratio:
                raise ValueError(f"lissajous_multi_joint: frequencies {freqs[a]:g} and {freqs[b]:g} Hz are commensurate ({near.numerator}:"
                                 f"{near.denominator}): the curve closes after {near.denominator} cycles; pick incommensurate ones")
    t = np.arange(max(2, int(math.ceil(duration_s * space.rate_hz - 1e-9))) + 1) / space.rate_hz
    total = t[-1]
    w, dw, ddw = k._fade(t, total, fade_s)
    shapes = [np.zeros((len(t), len(chosen))) for _ in range(3)]
    for column, frequency in enumerate(freqs):
        omega, phase = 2.0 * math.pi * frequency, 2.0 * math.pi * ((column * (2.0 - GOLDEN)) % 1.0)
        x, dx, ddx = np.sin(omega * t + phase), omega * np.cos(omega * t + phase), -(omega**2) * np.sin(omega * t + phase)
        shapes[0][:, column] = x * w
        shapes[1][:, column] = dx * w + x * dw
        shapes[2][:, column] = ddx * w + 2.0 * dx * dw + x * ddw
    q, qd, qdd = k._scaled(start, chosen, *shapes, space.room(start), amplitude_fraction)
    return k.Arrays(t, q, qd, qdd, label)
