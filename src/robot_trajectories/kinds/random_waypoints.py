from __future__ import annotations

from .. import kernels as k

from ..kind import Context, Kind, param


class RandomWaypoints(Kind):
    """Seeded random minimum-jerk moves inside the limits: everyday tracking."""

    catalog_id = "random_targets"
    family = "coverage"
    starts_as = "path"
    use_cases = ("tracking_eval", "smoke_test")
    needs = ("joint_limits", "velocity_limits", "acceleration_limits")
    unchecked = ("collision_model",)  # the catalog row checks collisions; a box (or the hole-free box) is all this sees

    count: int = param(8, "count", "how many random configurations to visit")
    seed: int = param(0, "seed", "the draw's seed (always seeded: a config must regenerate the same motion)")
    margin: float = param(0.1, "1", "fraction of each joint's range kept clear at both ends")
    speed: float = param(0.6, "1", "fraction of the velocity and acceleration limits a leg may use")
    dwell_s: float = param(0.2, "s", "the hold at each waypoint")

    def check(self) -> None:
        if self.count < 1:
            raise ValueError(f"RandomWaypoints.count must be at least 1, got {self.count}")

    def pieces(self, context: Context) -> list[k.Arrays]:
        # WHY room_around: with a reach map the draw stays inside the reachable box around the start, so no leg crosses a hole.
        targets = k.random_waypoints(context.room_around(context.start), self.count, seed=self.seed, margin=self.margin)
        return k.waypoint_chain(targets, context.bounds, context.rate_hz, start=context.start, speed=self.speed,
                                dwell_s=self.dwell_s, label=self.display_label)
