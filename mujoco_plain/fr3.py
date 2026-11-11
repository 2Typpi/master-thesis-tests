"""MuJoCo base environment for a Franka FR3 grasp-and-lift task."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import mujoco
import numpy as np
from ml_collections import config_dict


PROJECT_ROOT = Path(__file__).resolve().parent
ARM_JOINTS = [f"fr3v2_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
ARM_ACTUATORS = ARM_JOINTS
GRIPPER_ACTUATOR = "panda_actuator8"
OBJECT_BODY = "box"
GRIPPER_SITE = "panda_gripper"
FINGER_BODIES = ["panda_left_finger", "panda_right_finger"]
PICKUP_ARM_QPOS = [0.2897, 0.496673, -0.142836, -2.14746, -0.0295746, 2.52378, -0.492496]
PICKUP_BOX_QPOS = [0.529553, 0.0731702, 0.0299388, 0.94209, 0.0, 0.0, -0.335361]


def build_model(sim_dt: float) -> mujoco.MjModel:
    """Compose the FR3, Panda hand and cube into one MuJoCo model."""
    fr3_path = PROJECT_ROOT / "franka_fr3_v2" / "fr3v2.xml"
    hand_path = PROJECT_ROOT / "franka_emika_panda" / "hand.xml"

    spec = mujoco.MjSpec.from_file(str(fr3_path.parent / "scene.xml"))
    hand = mujoco.MjSpec.from_file(str(hand_path))
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    hand.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # The FR3 link8 is already offset from the flange by 0.107 m.
    hand_body = hand.body("hand")
    hand_body.pos = [0.0, 0.0, 0.0]
    hand_body.quat = [0.9238795, 0.0, 0.0, -0.3826834]
    hand_body.add_site(name="gripper", pos=[0.0, 0.0, 0.1])
    for geom in hand.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM):
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.contype == 0 and geom.conaffinity == 0:
            continue
        geom.contype = 1
        geom.conaffinity = 1
    spec.attach(
        hand,
        prefix="panda_",
        frame=spec.body("fr3v2_link8").add_frame(name="gripper_mount"),
    )

    box = spec.worldbody.add_body(name=OBJECT_BODY, pos=[0.5, 0.0, 0.03])
    box.add_freejoint(name="box_free")
    box.add_geom(
        name="box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.03],
        condim=3,
        friction=[2.0, 0.03, 0.003],
        rgba=[0.0, 1.0, 0.0, 1.0],
        contype=2,
        conaffinity=1,
        solref=[0.01, 1.0],
    )
    spec.option.timestep = sim_dt

    # Extend the FR3 home keyframe with the hand and the free cube state.
    home_arm_qpos = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
    home_qpos = home_arm_qpos + [0.04, 0.04, 0.5, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0]
    home_ctrl = home_arm_qpos + [255.0]
    home = spec.key("home")
    home.qpos = home_qpos
    home.ctrl = home_ctrl
    pick_qpos = PICKUP_ARM_QPOS + [0.04, 0.04] + PICKUP_BOX_QPOS
    pick_ctrl = PICKUP_ARM_QPOS + [255.0]
    spec.add_key(name="pick_home", qpos=pick_qpos, ctrl=pick_ctrl)
    return spec.compile()


class Fr3Base:
    """Joint-delta control environment for an FR3 with a parallel gripper."""

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, float, bool, list[Any]]]] = None,
    ):
        self._config = config.copy_and_resolve_references() if config else config_dict.ConfigDict()
        if config_overrides:
            self._config.update(config_overrides)
        if not self._config.get("reward_config"):
            self._config.reward_config = config_dict.ConfigDict()
        if not self._config.reward_config.get("reward_scales"):
            self._config.reward_config.reward_scales = config_dict.ConfigDict()
        self._model = build_model(float(self._config.get("sim_dt", 0.005)))
        self._data = mujoco.MjData(self._model)
        self._gripper_actuator_id = self._named_id(mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
        arm_joints = self._config.get("arm_joints", ARM_JOINTS)
        finger_joints = self._config.get("gripper_joints", FINGER_JOINTS)
        self._arm_joint_ids = self._named_ids(mujoco.mjtObj.mjOBJ_JOINT, arm_joints)
        self._arm_qpos_addrs = self._model.jnt_qposadr[self._arm_joint_ids].copy()
        self._arm_actuator_ids = self._named_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, arm_joints)
        self._finger_joint_ids = self._named_ids(mujoco.mjtObj.mjOBJ_JOINT, finger_joints)
        self._finger_qpos_addrs = self._model.jnt_qposadr[self._finger_joint_ids].copy()
        self._object_body_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
        self._object_joint_id = self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "box_free")
        self._object_qpos_addr = int(self._model.jnt_qposadr[self._object_joint_id])
        self._gripper_site_id = self._named_id(mujoco.mjtObj.mjOBJ_SITE, GRIPPER_SITE)
        self._finger_body_ids = self._named_ids(mujoco.mjtObj.mjOBJ_BODY, FINGER_BODIES)
        self._keyframe_id = self._named_id(
            mujoco.mjtObj.mjOBJ_KEY, str(self._config.get("reset_keyframe", "pick_home"))
        )
        self._step_count = 0
        self._last_action = np.zeros(self.action_size, dtype=np.float64)

    @property
    def action_size(self) -> int:
        return 8

    @property
    def observation_size(self) -> int:
        return 14

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    def _named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        result = mujoco.mj_name2id(self._model, object_type, name)
        if result < 0:
            raise ValueError(f"MuJoCo model is missing {object_type.name} named {name!r}")
        return result

    def _named_ids(self, object_type: mujoco.mjtObj, names: list[str]) -> np.ndarray:
        return np.asarray([self._named_id(object_type, name) for name in names], dtype=int)

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to the configured keyframe and return observation and info."""
        mujoco.mj_resetDataKeyframe(self._model, self._data, self._keyframe_id)
        self._step_count = 0
        self._last_action.fill(0.0)
        if seed is not None and self._config.get("randomize_box", False):
            rng = np.random.default_rng(seed)
            bounds = self._config.box_range
            offset = rng.uniform(np.asarray(bounds.low), np.asarray(bounds.high))
            self._data.qpos[self._object_qpos_addr : self._object_qpos_addr + 3] += offset
        mujoco.mj_forward(self._model, self._data)
        return self._observation(), {"is_grasped": self.is_grasped()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply [seven joint deltas, gripper command] and advance simulation."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,):
            raise ValueError(f"action must have shape ({self.action_size},), got {action.shape}")
        action = np.clip(np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

        max_delta = float(self._config.get("action_scale", 0.04))
        targets = self._data.ctrl[self._arm_actuator_ids] + action[:7] * max_delta
        limits = self._model.actuator_ctrlrange[self._arm_actuator_ids]
        self._data.ctrl[self._arm_actuator_ids] = np.clip(targets, limits[:, 0], limits[:, 1])
        open_command = action[7] > 0.0
        self._data.ctrl[self._gripper_actuator_id] = 255.0 if open_command else 0.0

        dt = float(self._config.get("ctrl_dt", 0.02))
        sim_dt = float(self._model.opt.timestep)
        repeat = int(self._config.get("action_repeat", 1))
        for _ in range(max(1, int(round(dt / sim_dt))) * repeat):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        reward = self._reward(action)
        terminated = self._success()
        truncated = self._step_count >= int(self._config.get("episode_length", 250))
        info = {
            "is_grasped": self.is_grasped(),
            "object_height": float(self._data.xpos[self._object_body_id, 2]),
            "tcp_distance": self._tcp_distance(),
        }
        self._last_action = action.copy()
        return self._observation(), reward, terminated, truncated, info

    def is_grasped(self, min_normal_force: float = 0.5) -> bool:
        """Return whether both finger bodies contact the cube under load."""
        finger_ids = set(int(body_id) for body_id in self._finger_body_ids)
        forces = np.zeros(2, dtype=np.float64)
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_id in range(self._data.ncon):
            contact = self._data.contact[contact_id]
            if contact.dist > 0:
                continue
            body1 = int(self._model.geom_bodyid[contact.geom1])
            body2 = int(self._model.geom_bodyid[contact.geom2])
            object_body_id = self._object_body_id
            finger_body_id = body2 if body1 == object_body_id else body1 if body2 == object_body_id else -1
            if finger_body_id not in finger_ids:
                continue
            mujoco.mj_contactForce(self._model, self._data, contact_id, contact_force)
            forces[list(self._finger_body_ids).index(finger_body_id)] += abs(float(contact_force[0]))
        return bool(np.all(forces >= min_normal_force))

    def _observation(self) -> np.ndarray:
        return np.concatenate((
            self._data.qpos[self._arm_qpos_addrs],
            self._data.qpos[self._finger_qpos_addrs[:1]],
            self._data.xpos[self._object_body_id],
            self._data.site_xpos[self._gripper_site_id],
        )).astype(np.float32)

    def _tcp_distance(self) -> float:
        return float(np.linalg.norm(
            self._data.site_xpos[self._gripper_site_id] - self._data.xpos[self._object_body_id]
        ))

    def _reward(self, action: np.ndarray) -> float:
        scales = self._config.get("reward_config", {}).get("reward_scales", {})
        distance = self._tcp_distance()
        object_height = float(self._data.xpos[self._object_body_id, 2])
        grasped = self.is_grasped()
        reward = float(scales.get("tcp_cube", 3.0)) * np.exp(-distance / 0.1)
        reward += float(scales.get("approach_open", 1.0)) * np.exp(-distance / 0.1) * float(self._data.ctrl[self._gripper_actuator_id] > 0)
        if grasped:
            reward += float(scales.get("grasp_incentive", 4.0))
        lift = max(0.0, object_height - 0.03)
        reward += float(scales.get("lift", 20.0)) * min(
            lift / max(float(self._config.reward_config.lift_target_height), 1e-6), 1.0
        )
        if grasped and lift >= float(self._config.reward_config.lift_success_height):
            reward += float(self._config.reward_config.success_reward)
        reward += float(self._config.reward_config.action_rate) * float(np.dot(action, action))
        return float(reward)

    def _success(self) -> bool:
        object_height = float(self._data.xpos[self._object_body_id, 2])
        return self.is_grasped() and object_height >= float(self._config.reward_config.lift_success_height)
