from __future__ import annotations

from .. import kernels as k

from ..kind import Context, Kind, param


class VelocitySweep(Kind):
    """Constant-speed passes across the range, up and down at several speeds, corners rounded at 0.8 x the acceleration
    limit: the friction curve and the gravity torque (the shared kernel).

    Catalog row: ``constant_speed_triangle`` at several speeds (the rack's own friction runs, one speed out and back
    from the origin, are ``VelocityPlateaus``)."""

    catalog_id = "constant_speed_triangle"
    family = "quasi_static"
    starts_as = "path"
    use_cases = ("friction_id", "gravity_id")

    joint: str | int | None = param(None, "joint", "the joint that sweeps (name or index; optional on a 1-DOF robot)")
    speeds_fraction: tuple[float, ...] = param((0.1, 0.3, 0.6), "1", "each pass's cruise speed as a fraction of the limit")
    span_fraction: float = param(0.8, "1", "how much of the room around the start each pass crosses")
    dwell_s: float = param(0.5, "s", "the hold at each end of a pass")

    def pieces(self, context: Context) -> list[k.Arrays]:
        return k.velocity_sweep(context.start, context.joint(self.joint), context.room_around(context.start), context.rate_hz,
                                speeds_fraction=self.speeds_fraction, span_fraction=self.span_fraction, dwell_s=self.dwell_s,
                                label=self.display_label)
