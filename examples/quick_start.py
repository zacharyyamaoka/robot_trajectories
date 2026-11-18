"""The README's quick start, runnable as is: the PID-tuning plan on the CAN 11 pendulum, from bare joint limits.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/examples/quick_start.py
"""

import numpy as np
from robot_msgs.trajectory import JointLimits

from robot_trajectories import JointTrajectoryGenerator, Workspace, plans

# Instance values live with the caller, never in the package: here, CAN 11's schedule (±0.33 rad, 2.0 rad/s, 7.3896 rad/s²).
limits = JointLimits(joint_names=("can11",), min_position=np.array([-0.33]), max_position=np.array([0.33]),
                     max_velocity=np.array([2.0]), max_acceleration=np.array([7.3896]))

workspace = Workspace.from_limits(limits)                      # rung 1 of the ladder: joint limits
generator = JointTrajectoryGenerator(workspace)
program = generator.generate(plans.pid_tuning(joint="can11"))  # a plan, or records: [StepTrain(joint="can11"), Hold()]
trajectory = program.trajectory                                # one robot_msgs JointTrajectory; program.segments: one per step

print(program.summary())
print(f"{type(trajectory).__module__}.{type(trajectory).__name__}: {len(trajectory.t)} samples, "
      f"peak {np.degrees(np.max(np.abs(trajectory.velocities))):.0f}°/s of {np.degrees(limits.max_velocity[0]):.0f}°/s")
