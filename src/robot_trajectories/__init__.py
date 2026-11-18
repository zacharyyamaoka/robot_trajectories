"""robot_trajectories: joint trajectories from what you have, as robot_msgs JointTrajectories inside your limits.

The recommended splice of the 2026-11-18 API round (docs/peps/0001-api-shape.md): V2's config records as the core, V3's
purpose plans as a robot-free front door, V4's workspace on Zach's joint limits -> kinematics -> dynamics ladder.

    workspace = Workspace.from_limits(limits)                     # rung 1; .with_kinematics / .with_reach_map / .with_dynamics
    generator = JointTrajectoryGenerator(workspace)
    generator.kinds(use_case="pid_tuning")                        # available / fallback / locked, with the missing input
    program = generator.generate(plans.pid_tuning(joint="can11")) # a plan, or records: [StepTrain(...), Hold(...)]
    program.trajectory      # one robot_msgs JointTrajectory, segments tied end to start
    program.segments        # the same motion, one rest-to-rest JointTrajectory per record; program.steps says why and when to wait
    program.to_json()       # the configuration as text; Program.from_json(text) makes it again, identically

``catalog(use_case=...)`` lists the record classes; ``print(Chirp.describe())`` shows one with its fields and units;
``trajectory_catalog`` holds the 61 catalog rows the records claim (``trajectory_catalog.rows()`` for catalog.json).
"""

from __future__ import annotations

__version__ = "0.0.1"

from . import plans, trajectory_catalog
from .generator import JointTrajectoryGenerator, KindLocked, KindRow, KindTable
from .kind import Catalog, Context, Kind, ReachSet, catalog, entry_level, param
from .kinds import *  # noqa: F403  (every record class, discovered: see kinds/__init__.py)
from .kinds import __all__ as _kinds
from .plan import Plan, PlanStep, step
from .program import Band, Program
from .signals import constant_speed_chirp
from .time_law import LegTiming, TimeLaw
from .workspace import MissingInput, Status, Workspace

__all__ = ["Band", "Catalog", "Context", "JointTrajectoryGenerator", "Kind", "KindLocked", "KindRow", "KindTable", "LegTiming",
           "MissingInput", "Plan", "PlanStep", "Program", "ReachSet", "Status", "TimeLaw", "Workspace", "catalog",
           "constant_speed_chirp", "entry_level", "param", "plans", "step", "trajectory_catalog", *_kinds]
