from __future__ import annotations

from ..registry import kind
from ..segments import annotate, chain
from .home import home
from .multisine import multisine
from .workspace_sweep import workspace_sweep


@kind(use_cases=("full_dynamics_id", "inertia_id"), family="excitation", starts_as="time_signal", dimensions=("2D", "3D", "6D"),
      higher_order=True)
def dynamics_excitation(space, *, joints=None, band_hz=(0.2, 3.0), period_s=10.0, periods=2, seed=0, pose_grid_joints=None,
                        points_per_joint=3, start=None):
    """What are its inertia, gravity and coupling? A multisine over ``band_hz`` on every joint at once (inertia, coupling,
    damping); ``pose_grid_joints`` adds a grid of static poses over those joints, then home (an arm's gravity terms)."""
    excitation = annotate(multisine(space, joints, band_hz=band_hz, period_s=period_s, periods=periods, seed=seed, start=start),
                          f"{band_hz[0]:g}-{band_hz[1]:g} Hz multisine on every joint at once: inertia, coupling, damping",
                          wait_after=not pose_grid_joints)
    if not pose_grid_joints:
        return excitation
    # WHY static poses on an arm (V3): gravity loads change with configuration; a multisine about one pose barely sees them.
    grid = annotate(workspace_sweep(space, joints=pose_grid_joints, points_per_joint=points_per_joint, speed=0.6, dwell_s=0.2,
                                    label="pose grid", start=excitation),
                    f"{points_per_joint ** len(pose_grid_joints)} poses over {', '.join(pose_grid_joints)}: the gravity terms, at rest")
    back = annotate(home(space, start=grid), "back home", wait_after=True)
    return chain(excitation, grid, back)
