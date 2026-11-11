"""Franka FR3 grasp-and-lift task configuration and environment."""

from typing import Any, Dict, Optional, Union

import mujoco
import numpy as np
from ml_collections import config_dict

from mujoco_plain.fr3 import Fr3Base


_ARM_JOINTS = [f"fr3v2_joint{i}" for i in range(1, 8)]
_GRIPPER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]


def default_config() -> config_dict.ConfigDict:
    """Return the grasp-lift curriculum config for a Franka FR3."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=250,
        action_repeat=1,
        action_scale=0.04,
        # The Panda hand actuator uses 0=open and 255=closed.
        gripper_close_delta=-85.0,
        gripper_open_delta=60.0,
        reward_config=config_dict.create(
            reward_scales=config_dict.create(
                tcp_cube=3.0,
                approach_open=1.0,
                touch=0.5,
                grasp_incentive=4.0,
                post_grasp_raise=4.0,
                lift_pose=6.0,
                lift=20.0,
                hold=6.0,
                self_collision=-1.0,
                floor_collision=-0.25,
            ),
            reached_cube_threshold=0.04,
            lift_target_height=0.18,
            lift_success_height=0.18,
            post_grasp_raise_height=0.12,
            grasp_close_threshold=0.01,
            action_rate=-0.0005,
            no_soln_reward=-0.01,
            success_reward=2.0,
        ),
        exploration_reset_prob=0.0,
        randomize_box=False,
        box_range=config_dict.create(
            low=[-0.015, -0.015, 0.0],
            high=[0.015, 0.015, 0.0],
        ),
        box_range_easy=config_dict.create(
            low=[-0.015, -0.015, 0.0],
            high=[0.015, 0.015, 0.0],
        ),
        box_range_hard_prob=1.0,
        vision=False,
        arm_joints=list(_ARM_JOINTS),
        gripper_joints=list(_GRIPPER_JOINTS),
        reset_keyframe="pick_home",
        pregrasp_box_offset=[-0.005, 0.0, 0.028],
        lift_box_offset=[0.02, 0.0, 0.028],
    )


def default_pick_place_config() -> config_dict.ConfigDict:
    """Return the default FR3 pick-and-place config."""
    return default_config()


class Fr3PickPlace(Fr3Base):
    """FR3 grasp-lift task: approach, grasp, and lift a cube."""

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, float, bool, list[Any]]]] = None,
    ):
        super().__init__(config or default_config(), config_overrides)
        self._vision = self._config.vision
        self._init_task()

    @property
    def observation_size(self) -> int:
        return super().observation_size + self.action_size

    def _init_task(self) -> None:
        """Prepare keyframe state, Jacobian shift map and collision ids."""
        self._rng = np.random.default_rng()
        self._action_scale = float(self._config.action_scale)
        self.n_substeps = max(
            1,
            int(round(float(self._config.ctrl_dt) / float(self._config.sim_dt)))
            * int(self._config.action_repeat),
        )
        self._lowers = self._model.actuator_ctrlrange[:, 0].copy()
        self._uppers = self._model.actuator_ctrlrange[:, 1].copy()

        key_qpos = self._model.key_qpos[self._keyframe_id].copy()
        key_ctrl = self._model.key_ctrl[self._keyframe_id].copy()
        self._default_box_pos = key_qpos[self._object_qpos_addr : self._object_qpos_addr + 3].copy()
        self._pregrasp_arm_q = key_qpos[self._arm_qpos_addrs].copy()
        self._init_q = key_qpos
        self._init_ctrl = key_ctrl

        # Linearize TCP position at the configured pickup pose. The damped
        # pseudoinverse maps a small cube translation to an arm joint shift.
        mujoco.mj_resetDataKeyframe(self._model, self._data, self._keyframe_id)
        mujoco.mj_forward(self._model, self._data)
        jac_pos = np.zeros((3, self._model.nv), dtype=np.float64)
        jac_rot = np.zeros((3, self._model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self._model, self._data, jac_pos, jac_rot, self._gripper_site_id
        )
        arm_dof_addrs = self._model.jnt_dofadr[self._arm_joint_ids]
        arm_jac = jac_pos[:, arm_dof_addrs]
        damping = 1e-4
        self._pregrasp_shift_qmat = arm_jac.T @ np.linalg.solve(
            arm_jac @ arm_jac.T + damping * np.eye(3), np.eye(3)
        )

        self.lift_target_arm_q = np.clip(
            self._pregrasp_arm_q
            + self._pregrasp_shift_qmat
            @ np.array([0.0, 0.0, float(self._config.reward_config.lift_target_height)]),
            self._lowers[:7],
            self._uppers[:7],
        )
        self.lift_pose_reward_offset = float(
            np.linalg.norm(self.lift_target_arm_q - self._pregrasp_arm_q)
        )
        self._floor_geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self._box_geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom"
        )
        robot_body_ids = set(range(1, self._model.nbody)) - {self._object_body_id}
        self._robot_geom_ids = {
            geom_id
            for geom_id in range(self._model.ngeom)
            if int(self._model.geom_bodyid[geom_id]) in robot_body_ids
        }
        self._prev_action = np.zeros(self.action_size, dtype=np.float64)
        self._reached_cube = False
        self._grasped = False
        self._lifted = False
        self._grasp_site_z = 0.0

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the task, optionally randomizing the cube reproducibly."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self._model, self._data, self._keyframe_id)

        offset = np.zeros(3, dtype=np.float64)
        if self._config.randomize_box:
            bounds = self._config.box_range
            offset = self._rng.uniform(
                np.asarray(bounds.low, dtype=np.float64),
                np.asarray(bounds.high, dtype=np.float64),
            )
        box_pos = self._default_box_pos + offset
        init_arm_q = np.clip(
            self._pregrasp_arm_q + self._pregrasp_shift_qmat @ offset,
            self._lowers[:7],
            self._uppers[:7],
        )

        self._episode_box_pos = box_pos.copy()
        self._data.qpos[self._object_qpos_addr : self._object_qpos_addr + 3] = box_pos
        self._data.qpos[self._arm_qpos_addrs] = init_arm_q
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = self._init_ctrl
        self._data.ctrl[self._arm_actuator_ids] = init_arm_q
        self._data.ctrl[self._gripper_actuator_id] = self._uppers[self._gripper_actuator_id]
        mujoco.mj_forward(self._model, self._data)

        self._step_count = 0
        self._prev_action.fill(0.0)
        self._reached_cube = False
        self._grasped = False
        self._lifted = False
        self._grasp_site_z = 0.0
        hard_sample = float(bool(self._config.randomize_box))
        metrics = {
            "out_of_bounds": 0.0,
            **{f"reward/{key}": 0.0 for key in self._config.reward_config.reward_scales},
            "reward/success": 0.0,
            "floor_collision": 0.0,
            "dist_to_cube": 0.0,
            "ik_no_solution": 0.0,
            "exploration_reset": 0.0,
            "reached_cube": 0.0,
            "touching_cube": 0.0,
            "both_contact": 0.0,
            "grasped": 0.0,
            "lifted": 0.0,
            "self_collision": 0.0,
            "curriculum_hard_spawn": hard_sample,
        }
        info = {
            "metrics": metrics,
            "prev_reward": 0.0,
            "current_pos": self._data.site_xpos[self._gripper_site_id].copy(),
            "current_rot": self._data.site_xmat[self._gripper_site_id].ravel().copy(),
            "newly_reset": False,
            "prev_action": self._prev_action.copy(),
            "reached_cube": 0.0,
            "touching_cube": 0.0,
            "grasped": 0.0,
            "grasp_site_z": 0.0,
            "lifted": 0.0,
            "lift_target_arm_q": self.lift_target_arm_q.copy(),
            "lift_pose_reward_offset": self.lift_pose_reward_offset,
            "curriculum_hard_spawn": hard_sample,
            "_steps": 0,
        }
        return self._get_obs(info), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply joint-delta and gripper actions and advance the simulation."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,):
            raise ValueError(f"action must have shape ({self.action_size},), got {action.shape}")
        action = np.clip(np.nan_to_num(action, nan=0.0), -1.0, 1.0)

        ctrl = self._data.ctrl.copy()
        ctrl[self._arm_actuator_ids] += action[:7] * self._action_scale
        close_gripper = action[7] < 0.0
        gripper_delta = float(
            self._config.gripper_close_delta
            if close_gripper
            else self._config.gripper_open_delta
        )
        ctrl[self._gripper_actuator_id] += gripper_delta
        self._data.ctrl[:] = np.clip(ctrl, self._lowers, self._uppers)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self._model, self._data)

        (raw_rewards, dist_to_cube, success, floor_collision,
         self_collision, reached_cube, touching_cube, grasped,
         grasp_site_z, lifted, both_contact) = self._get_reward(
            self._data,
            self._reached_cube,
            self._grasped,
            self._grasp_site_z,
            self._lifted,
            self.lift_target_arm_q,
            self.lift_pose_reward_offset,
        )
        scales = self._config.reward_config.reward_scales
        rewards = {key: value * float(scales.get(key, 0.0)) for key, value in raw_rewards.items()}
        total_reward = float(np.clip(sum(rewards.values()), -1e4, 1e4))
        total_reward += float(success) * float(self._config.reward_config.success_reward)
        total_reward += float(self._config.reward_config.action_rate) * float(action @ action)

        self._reached_cube = reached_cube
        self._grasped = grasped
        self._grasp_site_z = grasp_site_z
        self._lifted = lifted
        self._step_count += 1
        self._prev_action = action.copy()
        object_pos = self._data.xpos[self._object_body_id].copy()
        out_of_bounds = bool(
            object_pos[2] < 0.0
            or abs(object_pos[0]) > 1.0
            or abs(object_pos[1]) > 1.0
        )
        terminated = bool(success or out_of_bounds)
        truncated = self._step_count >= int(self._config.episode_length)

        metrics = {
            **{f"reward/{key}": float(value) for key, value in rewards.items()},
            "reward/success": float(success),
            "out_of_bounds": float(out_of_bounds),
            "floor_collision": float(floor_collision),
            "dist_to_cube": float(dist_to_cube),
            "ik_no_solution": 0.0,
            "exploration_reset": 0.0,
            "reached_cube": float(reached_cube),
            "touching_cube": float(touching_cube),
            "both_contact": float(both_contact),
            "grasped": float(grasped),
            "lifted": float(lifted),
            "self_collision": float(self_collision),
            "curriculum_hard_spawn": float(bool(self._config.randomize_box)),
        }
        info = {
            "metrics": metrics,
            "prev_reward": total_reward,
            "current_pos": self._data.site_xpos[self._gripper_site_id].copy(),
            "current_rot": self._data.site_xmat[self._gripper_site_id].ravel().copy(),
            "newly_reset": False,
            "prev_action": action.copy(),
            "reached_cube": float(reached_cube),
            "touching_cube": float(touching_cube),
            "grasped": float(grasped),
            "grasp_site_z": float(grasp_site_z),
            "lifted": float(lifted),
            "lift_target_arm_q": self.lift_target_arm_q.copy(),
            "lift_pose_reward_offset": self.lift_pose_reward_offset,
            "curriculum_hard_spawn": float(bool(self._config.randomize_box)),
            "_steps": self._step_count,
            "is_grasped": self.is_grasped(),
            "object_height": float(object_pos[2]),
            "tcp_distance": float(dist_to_cube),
        }
        return self._get_obs(info), total_reward, terminated, truncated, info

    def get_arm_joint_angles(self, data: mujoco.MjData) -> np.ndarray:
        """Return the seven FR3 arm joint positions."""
        return data.qpos[self._arm_qpos_addrs].copy()

    def _get_obs(self, info: dict[str, Any]) -> np.ndarray:
        """Return the base observation concatenated with the prior action."""
        del info
        return np.concatenate((super()._observation(), self._prev_action)).astype(np.float32)

    def _get_reward(
        self,
        data: mujoco.MjData,
        reached_cube: bool,
        grasped: bool,
        grasp_site_z: float,
        lifted: bool,
        lift_target_arm_q: np.ndarray,
        lift_pose_reward_offset: float,
    ) -> tuple[dict[str, float], float, bool, bool, bool, bool, bool, bool, float, bool, bool]:
        """Calculate tutorial-shaped reward terms and latched grasp/lift flags."""
        del data  # The task uses the environment's live model and data.
        distance = self._tcp_distance()
        cube_pos = self._data.xpos[self._object_body_id]
        threshold = max(float(self._config.reward_config.reached_cube_threshold), 1e-6)
        reached_cube = bool(reached_cube or distance <= threshold)
        left_contact, right_contact = self._finger_contacts()
        both_contact = bool(left_contact and right_contact)
        touching_cube = bool(left_contact or right_contact)
        new_grasp = bool(self.is_grasped())
        if new_grasp and not grasped:
            grasp_site_z = float(self._data.site_xpos[self._gripper_site_id, 2])
        grasped = bool(grasped or new_grasp)
        object_height = float(cube_pos[2])
        object_lift = max(0.0, object_height - float(self._episode_box_pos[2]))
        lifted = bool(
            lifted
            or (grasped and object_lift >= float(self._config.reward_config.lift_success_height))
        )
        success = bool(lifted)
        self_collision, floor_collision = self._collision_flags()
        is_open = float(self._data.ctrl[self._gripper_actuator_id] > 127.5)
        post_grasp_raise = 0.0
        if grasped and grasp_site_z > 0.0:
            post_grasp_raise = float(np.clip(
                (self._data.site_xpos[self._gripper_site_id, 2] - grasp_site_z)
                / max(float(self._config.reward_config.post_grasp_raise_height), 1e-6),
                0.0,
                1.0,
            ))
        lift_pose_error = float(np.linalg.norm(
            self.get_arm_joint_angles(self._data) - lift_target_arm_q
        ))
        lift_pose = float(np.exp(-lift_pose_error / max(lift_pose_reward_offset, 1e-6)))
        lift_progress = float(np.clip(
            object_lift / max(float(self._config.reward_config.lift_target_height), 1e-6),
            0.0,
            1.0,
        ))
        hold = float(grasped and new_grasp and object_lift > 0.01)
        scales = self._config.reward_config.reward_scales
        raw_rewards = {
            "tcp_cube": float(np.exp(-distance / threshold)),
            "approach_open": float(np.exp(-distance / threshold) * is_open),
            "touch": float(touching_cube),
            "grasp_incentive": float(new_grasp and not self._grasped),
            "post_grasp_raise": post_grasp_raise,
            "lift_pose": float(lift_pose * grasped),
            "lift": lift_progress,
            "hold": hold,
            "self_collision": float(self_collision),
            "floor_collision": float(floor_collision),
        }
        # Include scale keys added by a tutorial/config so reward aggregation
        # remains total and metrics retain a stable shape.
        for key in scales:
            raw_rewards.setdefault(key, 0.0)
        return (
            raw_rewards,
            distance,
            success,
            floor_collision,
            self_collision,
            reached_cube,
            touching_cube,
            grasped,
            grasp_site_z,
            lifted,
            both_contact,
        )

    def _finger_contacts(self) -> tuple[bool, bool]:
        contacts = [False, False]
        finger_ids = [int(body_id) for body_id in self._finger_body_ids]
        for contact_id in range(self._data.ncon):
            contact = self._data.contact[contact_id]
            body1 = int(self._model.geom_bodyid[contact.geom1])
            body2 = int(self._model.geom_bodyid[contact.geom2])
            if body1 == self._object_body_id:
                finger_body = body2
            elif body2 == self._object_body_id:
                finger_body = body1
            else:
                continue
            if finger_body in finger_ids and contact.dist <= 0.0:
                contacts[finger_ids.index(finger_body)] = True
        return bool(contacts[0]), bool(contacts[1])

    def _collision_flags(self) -> tuple[bool, bool]:
        self_collision = False
        floor_collision = False
        for contact_id in range(self._data.ncon):
            contact = self._data.contact[contact_id]
            if contact.dist > 0.0:
                continue
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == self._floor_geom_id and geom2 in self._robot_geom_ids:
                floor_collision = True
            elif geom2 == self._floor_geom_id and geom1 in self._robot_geom_ids:
                floor_collision = True
            elif geom1 in self._robot_geom_ids and geom2 in self._robot_geom_ids:
                self_collision = True
        return self_collision, floor_collision
