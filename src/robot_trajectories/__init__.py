"""robot_trajectories: a signal library for robot motion. Plain functions in, robot_msgs JointTrajectories out.

    space = JointSpace.from_limits(limits)                       # the robot, injected: + kinematics, dynamics, collision
    session = chain(step_train(space, joint="can11"), hold(space, 2.0), chirp(space, joint="can11"))
    pid_tuning(space, joint="can11")                             # a higher-order kind: lower kinds chained, one call
    cartesian_path(space.with_kinematics(fk_ik), [(0.4, 0.1, 0.3)])   # tool points in, joint motion out
    split(session), labels(session), rebuild(recipe(session))    # segments are a view; the recipe rebuilds it exactly
    catalog(use_case="pid_tuning", space=space)                  # what generates here, and what the catalog lists but nothing makes yet

Zach (2026-11-18): "Its pure its functional it feels reusable... all of them essentially outputting joint trajectories... a
common interface you can concatenate them all together". And the rule every kind obeys: robot trajectories "exist
completely independent from any actual physical manifestation or task": the robot is injected into the JointSpace (FK/IK,
dynamics, collision), the task arrives as points, and no kind knows what the motion is for. Decision record:
docs/peps/0002-signal-library.md.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import kinds as _builtin_kinds  # noqa: F401  (imports kinds/*.py, which registers every kind)
from . import plugins, trajectory_catalog
from .cartesian import Unreachable
from .moving import CannotCatch, LinearMotion, MovingTarget, SyncReport, TargetWaypoint, as_moving_target, synchronised
from .recipe import calls, rebuild, recipe, summary
from .registry import Catalog, Entry, KindInfo, catalog, kind, kinds
from .segments import Band, Motion, Piece, UnsafeTrajectory, annotate, bands, chain, labels, notes, repeat, segment_info, split
from .signals import constant_speed_chirp
from .space import JointSpace, MissingRung, ReachSet, State, Status
from .time_law import BoundaryState, LegTiming, TimeLaw, TimeLawBackend, register_time_law, time_laws

plugins.load()  # WHY last: a plugin module imports this package, which must be whole by then

# WHY the kinds are published from the registry, not an import list: a new file under kinds/ (or a plugin) is the whole change
globals().update({name: info.function for name, info in kinds().items()})

__all__ = ["Band", "BoundaryState", "CannotCatch", "Catalog", "Entry", "JointSpace", "KindInfo", "LegTiming", "LinearMotion",
           "MissingRung", "Motion", "MovingTarget", "Piece", "ReachSet", "State", "Status", "SyncReport", "TargetWaypoint", "TimeLaw",
           "TimeLawBackend", "Unreachable", "UnsafeTrajectory", "annotate", "as_moving_target", "bands", "calls", "catalog", "chain",
           "constant_speed_chirp", "kind", "kinds", "labels", "notes", "plugins", "rebuild", "recipe", "register_time_law", "repeat",
           "segment_info", "split", "summary", "synchronised", "time_laws", "trajectory_catalog", *kinds()]
