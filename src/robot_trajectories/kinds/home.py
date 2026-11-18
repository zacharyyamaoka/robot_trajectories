from __future__ import annotations

from .. import kernels as k

from ..kind import Context, Kind, param


class Home(Kind):
    """Move back to the workspace's home: the building block that ends a session where it began (no catalog row)."""

    family = "point_to_point"
    starts_as = "path"
    use_cases = ()

    speed: float = param(0.5, "1", "fraction of the velocity and acceleration limits the move may use")

    def pieces(self, context: Context) -> list[k.Arrays]:
        return [k.min_jerk(context.start, context.home, context.bounds, context.rate_hz, speed=self.speed, label=self.display_label)]
