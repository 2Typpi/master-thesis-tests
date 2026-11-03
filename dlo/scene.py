"""Compile the Franka Panda, cube, basket, and DER cable MJCF scene."""

from __future__ import annotations

from pathlib import Path

import mujoco

from .rope_xml import RopeConfig, write_rope_xml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "franka_emika_panda" / "mjx_single_cube_dlo.xml"
ROPE_PATH = PROJECT_ROOT / "franka_emika_panda" / "dlo_rope.xml"


def build_model(
    *,
    config: RopeConfig = RopeConfig(),
    timestep: float = 0.0015,
    iterations: int = 30,
) -> mujoco.MjModel:
    """Generate the matching rope asset and compile the independent DLO scene."""
    if timestep <= 0:
        raise ValueError("timestep must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if config.segments != 40:
        raise ValueError("scene keyframes currently require exactly 40 rope segments")
    write_rope_xml(ROPE_PATH, config)
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    model.opt.timestep = timestep
    model.opt.iterations = iterations
    rope_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "freejoint_A")
    rope_qpos_address = int(model.jnt_qposadr[rope_joint_id])
    for key_id in range(model.nkey):
        model.key_qpos[key_id, rope_qpos_address:rope_qpos_address + 3] = config.initial_position
        model.key_qpos[key_id, rope_qpos_address + 3:rope_qpos_address + 7] = config.initial_quaternion
    return model
