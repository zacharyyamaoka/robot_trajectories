from __future__ import annotations

from ..registry import kind
from ..segments import annotate
from .velocity_sweep import velocity_sweep


@kind(use_cases=("friction_id",), family="quasi_static", starts_as="path", dimensions=("1D",), higher_order=True)
def friction_id(space, joint=None, *, speeds_fraction=(0.1, 0.3, 0.6), start=None):
    """How much friction does this joint have? Constant-speed passes at ``speeds_fraction`` of top speed, up and down: Coulomb
    and viscous friction, Stribeck at the slow end."""
    speeds = ", ".join(f"{100 * fraction:.0f}" for fraction in speeds_fraction)
    return annotate(velocity_sweep(space, joint, speeds_fraction=speeds_fraction, start=start),
                    f"constant-speed passes at {speeds} % of top speed, up and down: Coulomb and viscous friction, Stribeck at the "
                    "slow end", wait_after=True)
