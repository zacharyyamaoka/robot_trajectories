"""A waypoint-trajectory-generator backend from outside the package: smoothstep legs, s = 3 tau^2 - 2 tau^3 (a cubic: continuous
velocity, the acceleration steps at the ends). Rest-to-rest only, one leg per pair of path points (it stops at each)."""

from __future__ import annotations

import math

import numpy as np
from robot_msgs.trajectory import JointTrajectory

PEAK_SPEED, PEAK_ACCELERATION = 1.5, 6.0  # max ds/dtau, max |d2s/dtau2| of the smoothstep


class SmoothStep:
    name = "smoothstep_test"

    def can_service(self, path_points, limits, law, start_state, end_state) -> str:
        for state in (start_state, end_state):
            if state is not None and (np.any(np.asarray(state.velocity) != 0) or np.any(np.asarray(state.acceleration) != 0)):
                return "smoothstep is rest-to-rest"
        return ""

    def time_parameterize(self, path_points, limits, rate_hz, law, *, start_state=None, end_state=None) -> JointTrajectory:
        path = np.asarray(path_points, float)
        t_all, q_all, qd_all, qdd_all, progress = [np.zeros(1)], [path[:1]], [np.zeros((1, path.shape[1]))], [np.zeros((1, path.shape[1]))], [np.zeros(1)]
        offset = 0.0
        for index, (a, b) in enumerate(zip(path[:-1], path[1:])):
            distance = np.abs(b - a)
            duration = max(float(np.max(PEAK_SPEED * distance / limits.max_velocity)),
                           float(np.max(np.sqrt(PEAK_ACCELERATION * distance / limits.max_acceleration))),
                           law.move_s or 0.0, law.min_move_s)
            steps = max(2, int(math.ceil(duration * rate_hz - 1e-9)))
            total = steps / rate_hz
            tau = np.arange(1, steps + 1) / steps
            s, ds, dds = 3 * tau**2 - 2 * tau**3, 6 * tau - 6 * tau**2, 6 - 12 * tau
            delta = (b - a)[None, :]
            t_all.append(offset + tau * total)
            q_all.append(a + delta * s[:, None])
            qd_all.append(delta * ds[:, None] / total)
            qdd_all.append(delta * dds[:, None] / total**2)
            progress.append(index + s)
            offset += total
        t = np.round(np.concatenate(t_all) * rate_hz) / rate_hz
        q = np.concatenate(q_all)
        q[-1] = path[-1]
        qd, qdd = np.concatenate(qd_all), np.concatenate(qdd_all)
        qd[-1] = 0.0
        return JointTrajectory(t=t, joint_names=tuple(limits.joint_names), positions=q, velocities=qd, accelerations=qdd,
                               channels={"path_progress": np.concatenate(progress)})
