"""MuJoCo adapter for the compiled adapteddlo_muj discrete-elastic-rod solver."""

from __future__ import annotations

import numpy as np
import mujoco

try:
    from .cpp import Dlo_iso
except ImportError as exc:
    raise ImportError(
        "The DER extension is not built. Activate the mujoco environment and run "
        "`bash dlo/cpp/build.sh`."
    ) from exc


class DloRope:
    """Compute DER joint torques for a free capsule-link rope."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        segments: int,
        *,
        overall_rot: float = 0.0,
        alpha_bar: float = 1.345,
        beta_bar: float = 0.789,
    ) -> None:
        if segments < 3:
            raise ValueError("segments must be at least 3")
        self.model = model
        self.data = data
        self.segments = segments
        self.alpha_bar = float(alpha_bar)
        self.beta_bar = float(beta_bar)
        self.overall_rot = float(overall_rot)

        self.site_ids = np.array([
            self._named_id(mujoco.mjtObj.mjOBJ_SITE, f"S_{i}") for i in range(segments)
        ] + [self._named_id(mujoco.mjtObj.mjOBJ_SITE, "S_last")], dtype=np.int32)
        self.start_site_id = self._named_id(mujoco.mjtObj.mjOBJ_SITE, "S_first")
        self.end_site_id = self.site_ids[-1]

        body_names = ["B_first"] + [f"B_{i}" for i in range(1, segments - 1)] + ["B_last", "B_last2"]
        self.body_ids = np.array([
            self._named_id(mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names
        ], dtype=np.int32)
        if len(self.body_ids) != segments + 1:
            raise ValueError("rope body count does not match segments")

        joint_names = [f"J_{i}" for i in range(1, segments - 1)] + ["J_last"]
        joint_ids = np.array([
            self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names
        ], dtype=np.int32)
        root_joint_id = self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "freejoint_A")
        self.joint_qvel_addrs = model.jnt_dofadr[joint_ids].copy()
        self.qvel_start = int(model.jnt_dofadr[root_joint_id]) + 3
        self.qvel_stop = int(np.max(self.joint_qvel_addrs)) + 3

        expected_torques = segments * 3
        if self.qvel_start < 0 or self.qvel_stop - self.qvel_start != expected_torques:
            raise ValueError(
                "rope joint layout is incompatible with the DER torque mapping: "
                f"qvel slice [{self.qvel_start}:{self.qvel_stop}] has "
                f"{self.qvel_stop - self.qvel_start} entries, expected {expected_torques}"
            )
        if self.qvel_stop > model.nv:
            raise ValueError("rope torque slice exceeds the model velocity dimension")

        self.node_positions = np.empty((segments + 1, 3), dtype=np.float64)
        self.node_positions_flat = self.node_positions.reshape(-1)
        self.centerline_torques = np.zeros((segments + 1, 3), dtype=np.float64)
        self.centerline_torques_flat = self.centerline_torques.reshape(-1)
        self.body_quaternions = np.empty((segments + 1, 4), dtype=np.float64)
        self.body_quaternions_flat = self.body_quaternions.reshape(-1)
        self.bishop_start = np.zeros((3, 3), dtype=np.float64)
        self.bishop_start_flat = self.bishop_start.reshape(-1)
        self.bishop_end = np.zeros((3, 3), dtype=np.float64)
        self.bishop_end_flat = self.bishop_end.reshape(-1)
        self.material_start = np.empty(9, dtype=np.float64)
        self.material_end = np.empty(9, dtype=np.float64)
        self.start_quaternion = np.empty(4, dtype=np.float64)
        self.bishop_quaternion = np.empty(4, dtype=np.float64)
        self.quaternion_inverse = np.empty(4, dtype=np.float64)
        self.quaternion_error = np.empty(4, dtype=np.float64)
        self.quaternion_error_velocity = np.empty(3, dtype=np.float64)
        self.theta_n = ((self.overall_rot + np.pi) % (2.0 * np.pi)) - np.pi
        self.theta_displace = self.theta_n
        self.material_frame_offset = np.empty(4, dtype=np.float64)

        self._read_nodes()
        self._init_bishop_frame()
        self.dlo_math = Dlo_iso.DLO_iso(
            self.node_positions_flat,
            self.bishop_start_flat,
            self.theta_n,
            self.overall_rot,
            self.alpha_bar,
            self.beta_bar,
        )
        self._init_material_frame_offset()

    def _named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo model is missing {object_type.name} '{name}'")
        return int(object_id)

    def _read_nodes(self) -> None:
        for index, site_id in enumerate(self.site_ids):
            self.node_positions[index] = self.data.site_xpos[site_id]

    def _init_bishop_frame(self) -> None:
        tangent = self.node_positions[1] - self.node_positions[0]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm <= np.finfo(np.float64).eps:
            raise ValueError("first two rope nodes must not coincide")
        tangent /= tangent_norm
        normal = np.cross(tangent, (0.0, 0.0, 1.0))
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            normal = np.cross(tangent, (0.0, 1.0, 0.0))
            normal_norm = np.linalg.norm(normal)
        normal /= normal_norm
        binormal = np.cross(tangent, normal)
        self.bishop_start[0] = tangent
        self.bishop_start[1] = normal
        self.bishop_start[2] = binormal

    def _quat_error(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        if np.dot(first, second) < 0:
            first = -first
        mujoco.mju_negQuat(self.quaternion_inverse, first)
        mujoco.mju_mulQuat(self.quaternion_error, second, self.quaternion_inverse)
        mujoco.mju_quat2Vel(self.quaternion_error_velocity, self.quaternion_error, 1.0)
        return self.quaternion_error_velocity

    def _axisangle_to_quat(self, vector: np.ndarray, result: np.ndarray) -> None:
        angle = np.linalg.norm(vector)
        if angle == 0.0:
            result[:] = (1.0, 0.0, 0.0, 0.0)
        else:
            mujoco.mju_axisAngle2Quat(result, vector / angle, angle)

    def _init_material_frame_offset(self) -> None:
        mujoco.mju_mat2Quat(self.start_quaternion, self.data.site_xmat[self.start_site_id])
        mujoco.mju_mat2Quat(self.bishop_quaternion, self.bishop_start.T.reshape(-1))
        self._init_local_rotation(self.start_quaternion, self.bishop_quaternion, self.material_frame_offset)
        self.dlo_math.initQe_o2m_loc(self.material_frame_offset)

    def _init_local_rotation(self, first: np.ndarray, second: np.ndarray, result: np.ndarray) -> None:
        error = self._quat_error(first, second)
        self._axisangle_to_quat(error, self.quaternion_error)
        mujoco.mju_negQuat(self.quaternion_inverse, first)
        mujoco.mju_mulQuat(self.bishop_quaternion, self.quaternion_inverse, self.quaternion_error)
        mujoco.mju_mulQuat(result, self.bishop_quaternion, first)

    def _update_start_material_frame(self) -> None:
        self.dlo_math.calculateOf2Mf(
            self.data.site_xmat[self.start_site_id],
            self.material_start,
        )
        self.bishop_start[:] = self.material_start.reshape((3, 3)).T

    def _get_theta_n(self) -> float:
        self.dlo_math.calculateOf2Mf(
            self.data.site_xmat[self.end_site_id],
            self.material_end,
        )
        material_end = self.material_end.reshape((3, 3))
        bishop_end = self.bishop_end.T
        theta = self.dlo_math.angBtwn3(bishop_end[:, 1], material_end[:, 1], bishop_end[:, 0])
        return float(theta + self.theta_displace)

    def update_torque(self) -> None:
        """Update DER state and add its torques to MuJoCo's passive generalized forces."""
        self._read_nodes()
        self._update_start_material_frame()
        self.dlo_math.updateVars(self.node_positions_flat, self.bishop_start_flat, self.bishop_end_flat)
        self.theta_n = self._get_theta_n()
        self.overall_rot = float(self.dlo_math.updateTheta(self.theta_n))

        for index, body_id in enumerate(self.body_ids):
            self.body_quaternions[index] = self.data.xquat[body_id]
        self.dlo_math.calculateCenterlineTorq(
            self.centerline_torques_flat,
            self.body_quaternions_flat,
            0,
        )
        self.data.qfrc_passive[self.qvel_start:self.qvel_stop] += self.centerline_torques[:-1].reshape(-1)

    def change_stiffness(self, alpha_bar: float, beta_bar: float) -> None:
        self.alpha_bar = float(alpha_bar)
        self.beta_bar = float(beta_bar)
        self.dlo_math.changeAlphaBeta(self.alpha_bar, self.beta_bar)

    def reset(self, overall_rot: float = 0.0) -> None:
        self.overall_rot = float(overall_rot)
        theta_n = ((overall_rot + np.pi) % (2.0 * np.pi)) - np.pi
        self.theta_displace = theta_n
        self.theta_n = theta_n
        self.dlo_math.resetTheta(theta_n, overall_rot)
