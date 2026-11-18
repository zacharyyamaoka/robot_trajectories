"""The README's quick start, runnable as is: the CAN 11 pendulum from bare joint limits, in the signal-library API.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/examples/quick_start.py
"""

import numpy as np
from robot_msgs.trajectory import JointLimits

from robot_trajectories import JointSpace, chain, hold, pid_tuning, rebuild, recipe, split, step_train, summary

# Instance values live with the caller, never in the package: here, CAN 11's schedule (±0.33 rad, 2.0 rad/s, 7.3896 rad/s²).
limits = JointLimits(joint_names=("can11",), min_position=np.array([-0.33]), max_position=np.array([0.33]),
                     max_velocity=np.array([2.0]), max_acceleration=np.array([7.3896]))

space = JointSpace.from_limits(limits)                      # the robot: joint limits (+ kinematics, dynamics, collision)
tuning = pid_tuning(space, joint="can11")                   # a higher-order kind: step train, hold, chirp, hold
session = chain(tuning, step_train(space, joint="can11", steps_fraction=(0.3, -0.3)), hold(space, 2.0))

print(summary(session))
print(f"{len(split(session))} rest-to-rest pieces; rebuilt from its {len(recipe(session))}-character recipe: "
      f"{'identical' if np.array_equal(rebuild(recipe(session)).positions, session.positions) else 'DIFFERENT'}")
print(f"{type(session).__module__}.{type(session).__name__}: {len(session.t)} samples, "
      f"peak {np.degrees(np.max(np.abs(session.velocities))):.0f}°/s of {np.degrees(limits.max_velocity[0]):.0f}°/s")
