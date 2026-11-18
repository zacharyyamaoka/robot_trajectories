"""Lane B's fixtures (numpy only): stand-in dynamics with ``forward(q, qd, tau)`` for the torque and hold kinds, and the spaces
they ride on. Every number here is a STAND-IN, labelled, never a measurement of a rig the package may name.

- ``Can11Dynamics``: the pendulum of the One Dof Test Rack's CAN 11 with the values its model cards give (gravity 4.66 N.m,
  inertia 0.1219 kg.m^2 = sim2real's registry ``armature_kg_m2`` 0.12193 and ``gravity_nm`` 4.661, Coulomb 0.833 N.m = its
  ``frictionloss_nm`` "measured Coulomb"); the viscous term and the drive's 5 N.m limit are stand-ins (the registry has no damping).
  Coulomb friction sticks exactly: ``forward`` returns the deceleration that stops the joint in one sample when the friction
  can hold it there, which is why it carries the sample rate it will be stepped at.
- ``SerialGravity``: a planar serial chain's gravity torque on some joints (a stand-in for the five-bar's two drives and for the
  UR5e's lift / elbow / wrist-1 plane) with a diagonal inertia and viscous damping, so ``torque`` and ``forward`` are one model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fixtures import robots
from robot_trajectories import JointSpace

G_ACCELERATION = 9.81

# ---------------------------------------------------------------------- the pendulum (CAN 11 values)

CAN11_DRIVE_LIMIT_NM = 5.0  # STAND-IN: the rack's drive limit is not in this package


@dataclass(frozen=True)
class Can11Dynamics:
    """J qdd + b qd + Fc sign(qd) + G sin(q) = tau, with sticking Coulomb friction (stand-in values: see the module docstring)."""

    inertia: float = 0.12193  # kg m^2, sim2real cards/registry.py armature_kg_m2 (CAN 11)
    gravity_torque: float = 4.661  # N m, registry gravity_nm (CAN 11: the pendulum's m g L)
    coulomb: float = 0.833  # N m, registry frictionloss_nm (CAN 11, model card: measured Coulomb)
    damping: float = 0.02  # N m s / rad: STAND-IN
    rate_hz: float = 500.0  # the clock it is stepped at (the space's rate): the stick needs the step
    name: str = "CAN 11 pendulum stand-in (J qdd + b qd + Fc stick + G sin q)"
    cache_key: str = "can11-stand-in-1"

    def torque(self, q, qd, qdd):
        q, qd, qdd = (np.asarray(value, float) for value in (q, qd, qdd))
        return self.inertia * qdd + self.damping * qd + self.coulomb * np.tanh(qd / 0.05) + self.gravity_torque * np.sin(q)

    def forward(self, q, qd, tau):
        q, qd, tau = (np.asarray(value, float) for value in (q, qd, tau))
        dt = 1.0 / self.rate_hz
        drive = tau - self.damping * qd - self.gravity_torque * np.sin(q)  # everything but the Coulomb term
        stop = drive + self.inertia * qd / dt  # the friction that would bring qd to zero in one step
        sticks = np.abs(stop) <= self.coulomb
        slide = np.where(qd != 0.0, np.sign(qd), np.sign(stop))
        return np.where(sticks, -qd / dt, (drive - self.coulomb * slide) / self.inertia)


def pendulum_limits(max_effort_nm: float = CAN11_DRIVE_LIMIT_NM):
    return robots.PENDULUM_LIMITS.replace(max_effort=np.array([float(max_effort_nm)]))


def pendulum_space(dynamics: Can11Dynamics | None = None, max_effort_nm: float = CAN11_DRIVE_LIMIT_NM) -> JointSpace:
    """CAN 11's limits + a stand-in drive limit + the CAN 11 dynamics."""
    return JointSpace.from_limits(pendulum_limits(max_effort_nm)).with_dynamics(dynamics or Can11Dynamics())


def pendulum_home_space(dynamics: Can11Dynamics | None = None) -> JointSpace:
    """The same pendulum with a NAMED home (the gravity bottom, 0), for the kinds that need a home_pose."""
    return JointSpace.from_limits(pendulum_limits(), home=[0.0]).with_dynamics(dynamics or Can11Dynamics())


# ---------------------------------------------------------------------- arms (stand-in gravity)


@dataclass(frozen=True)
class SerialGravity:
    """Gravity torque of a planar serial chain on ``plane_joints`` (relative angles from the horizontal: link k points at the sum
    of the angles up to it), lumped masses at ``centres`` along the links; the other joints carry none. ``forward`` is
    (tau - gravity - damping qd) / inertia per joint. STAND-IN values."""

    plane_joints: tuple[int, ...]
    lengths: tuple[float, ...]
    centres: tuple[float, ...]
    masses: tuple[float, ...]
    inertia: tuple[float, ...]
    damping: float = 0.05
    name: str = "planar serial gravity stand-in"
    cache_key: str = "serial-gravity-stand-in-1"

    def gravity(self, q) -> np.ndarray:
        q = np.asarray(q, float)
        angles = np.cumsum([q[joint] for joint in self.plane_joints])
        out = np.zeros(len(q))
        for index, joint in enumerate(self.plane_joints):
            total = 0.0
            for link in range(index, len(self.plane_joints)):
                reach = sum(self.lengths[k] * math.cos(angles[k]) for k in range(index, link)) + self.centres[link] * math.cos(angles[link])
                total += self.masses[link] * G_ACCELERATION * reach
            out[joint] = total
        return out

    def torque(self, q, qd, qdd):
        return np.asarray(self.inertia) * np.asarray(qdd, float) + self.damping * np.asarray(qd, float) + self.gravity(q)

    def forward(self, q, qd, tau):
        return (np.asarray(tau, float) - self.damping * np.asarray(qd, float) - self.gravity(q)) / np.asarray(self.inertia)


FIVEBAR_DYNAMICS = SerialGravity(plane_joints=(0, 1), lengths=(0.24, 0.30), centres=(0.12, 0.15), masses=(1.2, 0.8), inertia=(0.05, 0.05),
                                 name="five-bar gravity stand-in (two lumped links)", cache_key="fivebar-gravity-stand-in-1")
UR5E_DYNAMICS = SerialGravity(plane_joints=(1, 2, 3), lengths=(0.425, 0.392, 0.1), centres=(0.21, 0.2, 0.05), masses=(8.0, 3.0, 1.0),
                              inertia=(0.5, 0.5, 0.5, 0.1, 0.1, 0.1)[:6], damping=0.2,
                              name="UR5e gravity stand-in (lift, elbow, wrist-1 plane)", cache_key="ur5e-gravity-stand-in-1")
FIVEBAR_DRIVE_LIMIT_NM = 20.0  # STAND-IN
UR5E_DRIVE_LIMIT_NM = 150.0  # STAND-IN


def fivebar_space(dynamics=FIVEBAR_DYNAMICS) -> JointSpace:
    """The five-bar with a reach map, a named home, its closed-form kinematics, the stand-in gravity and a stand-in drive limit."""
    limits = robots.FIVEBAR_LIMITS.replace(max_effort=np.full(2, FIVEBAR_DRIVE_LIMIT_NM))
    space = JointSpace.from_limits(limits, home=np.radians([100.0, -76.0])).with_reach_map(robots.fivebar_reach_map())
    space = space.with_kinematics(robots.FiveBarKinematics())
    return space if dynamics is None else space.with_dynamics(dynamics)


def ur5e_space(dynamics=UR5E_DYNAMICS) -> JointSpace:
    """The UR5e in its demo box with a NAMED home (the box's centre), its DH kinematics, the stand-in gravity and a drive limit."""
    limits = robots.UR5E_DEMO_BOX.replace(max_effort=np.full(6, UR5E_DRIVE_LIMIT_NM))
    home = 0.5 * (limits.min_position + limits.max_position)
    space = JointSpace.from_limits(limits, home=home).with_kinematics(robots.UR5eKinematics())
    return space if dynamics is None else space.with_dynamics(dynamics)
