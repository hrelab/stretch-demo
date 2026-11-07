"""Bounded differential IK at the grasp center, from mujoco-sim's
end_effector/imu_ee_postwrist.PostwristIK (ee_controller commit 85f57de).

The task Jacobian includes the full wrist-to-grasp offset, so rotations
coordinate translation instead of swinging the grasp center about the wrist.
Unchanged except the constructor's transmission loop and the base-speed bounds,
which are now their own functions.
"""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from stretch_demo.sim.base_drive import OMNI_FULL_SPEED, OMNI_SLEW_RATE

WRIST_NAMES = ("wrist_yaw", "wrist_pitch", "wrist_roll")
JOINT_NAMES = ("lift", "arm", *WRIST_NAMES)
LIFT_ARM_SPEEDS = (0.15, 0.20)  # m/s


def transmission(model, ids):
    """The actuator -> joint map, which never changes during a session.

    Returns (dof matrix nv x len(ids), qpos indices, source actuator per qpos,
    scale per qpos). A fixed tendon (the arm's telescoping `extend`) spreads
    over its joints by coefficient: the actuator is the TOTAL extension.
    """
    matrix = np.zeros((model.nv, len(ids)))
    indices, sources, scales = [], [], []
    for column, idx in enumerate(ids):
        joints, scale = _actuator_joints(model, idx)
        for joint in joints:
            matrix[model.jnt_dofadr[joint], column] = scale
            indices.append(model.jnt_qposadr[joint])
            sources.append(idx)
            scales.append(scale)
    return matrix, np.asarray(indices), np.asarray(sources), np.asarray(scales)


def _actuator_joints(model, idx):
    target = model.actuator_trnid[idx, 0]
    gear = model.actuator_gear[idx, 0]
    if model.actuator_trntype[idx] != mujoco.mjtTrn.mjTRN_TENDON:
        return [target], 1 / gear
    start = model.tendon_adr[target]
    wraps = np.arange(start, start + model.tendon_num[target])
    return model.wrap_objid[wraps], 1 / (np.sum(model.wrap_prm[wraps]) * gear)


def base_speed_bounds(motion, fraction: float, base_state, acceleration_step):
    """Available (forward, strafe) base velocity, m/s.

    Spring-zone assistance: forward/back base motion is only on offer with the
    arm extension target in its outer zones (past reach_start_fraction); away
    from them, extension supplies fore/aft motion and the base only strafes.
    Both bounds are slew-limited around the base's current speed.
    """
    start = motion.reach_start_fraction
    backward = np.clip(((1 - start) - fraction) / (1 - start), 0, 1)
    forward = np.clip((fraction - start) / (1 - start), 0, 1)
    lo = np.array([-motion.reach_base_speed_m_s * backward, -OMNI_FULL_SPEED[1]])
    hi = np.array([motion.reach_base_speed_m_s * forward, OMNI_FULL_SPEED[1]])
    previous = base_state[:2] * OMNI_FULL_SPEED[:2]
    lo = np.clip(lo, previous - acceleration_step, previous + acceleration_step)
    hi = np.clip(hi, previous - acceleration_step, previous + acceleration_step)
    return lo, hi


class PostwristIK:
    """Bounded 6x7 differential IK, in world coordinates and actuator units.

    Variables: base forward, base strafe, lift, extension, wrist yaw/pitch/roll.
    Base yaw is held. Only a private MjData is posed, never the live robot.
    """

    def __init__(self, model, act, settings):
        self.model = model
        self.settings = settings
        self.data = mujoco.MjData(model)
        self.base_id = model.body("base_link").id
        self.tool_id = model.body("grasp_center_link").id
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))
        self.jacobian = np.zeros((6, 7))
        self.achieved = np.zeros(6)
        self.success = True
        self.ids = np.array([act[name] for name in JOINT_NAMES])
        self.limits = model.actuator_ctrlrange[self.ids].copy()
        wrist = np.deg2rad(settings.motion.wrist_speed_deg_s)
        self.speeds = np.array([*LIFT_ARM_SPEEDS, wrist, wrist, wrist])
        # Position errors outweigh angular ones when a twist is infeasible:
        # don't rotate freely while the grasp center drifts.
        self.weights = np.array([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])
        self.matrix = np.zeros((13, 7))
        # Small regularizer: a nearby, low-rate solution in the redundant
        # forward/extension direction, preferring spring-zone base assistance.
        self.matrix[6:] = 1e-4 * np.diag([0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
        self.rhs = np.zeros(13)
        self.arm_idx = act["arm"]
        self.arm_lo, self.arm_hi = model.actuator_ctrlrange[self.arm_idx]
        self.acceleration = OMNI_SLEW_RATE * OMNI_FULL_SPEED[:2]
        self.transmission, self.qpos_indices, self.qpos_sources, self.qpos_scales = transmission(model, self.ids)

    def update(self, live, setpoint):
        """Jacobian at the commanded pose (setpoint: a full ctrl vector)."""
        self.data.qpos[:] = live.qpos
        self.data.qpos[self.qpos_indices] = setpoint[self.qpos_sources] * self.qpos_scales
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_jacBody(self.model, self.data, self.jp, self.jr, self.tool_id)
        jac = self.jacobian
        jac[:] = 0
        base = self.data.xmat[self.base_id].reshape(3, 3)
        jac[:3, :2] = base[:, :2]
        jac[:3, 2:] = self.jp @ self.transmission
        jac[3:, 2:] = self.jr @ self.transmission
        return jac

    def solve(self, task, setpoint, base_state, dt):
        """Rates (base forward m/s, base strafe m/s, then each joint's actuator
        units/s) that best achieve the world twist `task` within every bound."""
        fraction = np.clip((setpoint[self.arm_idx] - self.arm_lo) / (self.arm_hi - self.arm_lo), 0, 1)
        base_lo, base_hi = base_speed_bounds(self.settings.motion, fraction, base_state, self.acceleration * dt)
        positions = setpoint[self.ids]
        lo = np.r_[base_lo, np.maximum(-self.speeds, (self.limits[:, 0] - positions) / dt)]
        hi = np.r_[base_hi, np.minimum(self.speeds, (self.limits[:, 1] - positions) / dt)]
        self.matrix[:6] = self.jacobian * self.weights[:, None]
        self.rhs[:6] = task * self.weights
        fixed = hi - lo < 1e-12
        result = (lo + hi) * 0.5
        active = ~fixed
        self.success = True
        if np.any(active):
            matrix = self.matrix
            solved = lsq_linear(matrix[:, active], self.rhs - matrix[:, fixed] @ result[fixed],
                                bounds=(lo[active], hi[active]), method="bvls", tol=1e-9, max_iter=40)
            result[active] = solved.x
            self.success = bool(solved.success)
        self.achieved = self.jacobian @ result
        return result
