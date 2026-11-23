"""TrajectoryGenerator: anything that outputs a robot_msgs Trajectory. The umbrella every kind in this package sits under.

Zach (2026-11-23): "lets give it the name TrajectoryGenerator ... these ones are specially WaypointTrajectoryGenerators. All
TrajectoryGenerators output Trajectories... so yes there are many possible like type of generators but they all share that
commonality". So:

- a **TrajectoryGenerator** is any callable that returns a robot_msgs ``Trajectory``: every kind (``joint_waypoints``,
  ``chirp``, ``pick_and_place``, ``speed_ladder``, ...) is one, and so is a kind bound to its space
  (``functools.partial(chirp, space, joint="can11")``);
- a **WaypointTrajectoryGenerator** (``waypoint_trajectory_generator.py``) is the special one the kinds hand their legs' timing
  to: points and joint limits in, a JointTrajectory out (minimum-jerk, trapezoid, Ruckig, Ruckig Pro, TOPP-RA). Calling one,
  ``generator(points, limits, rate_hz)``, makes that trajectory, so it is a TrajectoryGenerator in the isinstance sense too;
  inside a kind the same timing runs as ``leg`` / ``along``, and a plugin serves it through its backend's ``time_parameterize``.

The stage they make up is still *trajectory generation*.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from robot_msgs.trajectory import Trajectory


@runtime_checkable
class TrajectoryGenerator(Protocol):
    """Anything that outputs a robot_msgs ``Trajectory`` when called (a JointTrajectory, a PoseTrajectory, your own subclass).

    What the runtime check can and cannot see: ``isinstance(x, TrajectoryGenerator)`` only asks whether ``x`` is callable (a
    ``runtime_checkable`` Protocol checks that ``__call__`` exists, never its parameters or what it returns), so every plain
    function passes it, and the return type is not checked structurally. The return type is the contract, held by a static
    type checker against this annotation and by ``tests/test_names.py`` against what the kinds actually return."""

    def __call__(self, *args: Any, **kwargs: Any) -> Trajectory: ...


__all__ = ["TrajectoryGenerator"]
