"""Fast smoke checks for the DER cable code (run with ``python -m dlo.check``)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .rope_xml import RopeConfig, generate_rope_xml, write_rope_xml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_xml_generator() -> None:
    config = RopeConfig(segments=6, length=1.2, radius=0.02)
    xml = generate_rope_xml(config)
    root = ET.fromstring(xml)
    body_names = [body.attrib["name"] for body in root.findall(".//body")]
    site_names = {site.attrib["name"] for site in root.findall(".//site")}
    joint_names = {joint.attrib["name"] for joint in root.findall(".//joint")}
    assert "stiffrope" in body_names
    assert "B_first" in body_names and "B_last" in body_names and "B_last2" in body_names
    assert {f"S_{index}" for index in range(config.segments)} <= site_names
    assert {"S_first", "S_last"} <= site_names
    assert ({f"J_{index}" for index in range(1, config.segments - 1)} | {"J_last"}) <= joint_names
    assert len(root.findall("./contact/exclude")) >= config.segments - 1

    with np.testing.assert_raises(ValueError):
        RopeConfig(length=0.1, segments=10, radius=0.01)


def check_scene_compile() -> None:
    rope_path = PROJECT_ROOT / "franka_emika_panda" / "dlo_rope.xml"
    scene_path = PROJECT_ROOT / "franka_emika_panda" / "mjx_single_cube_dlo.xml"
    write_rope_xml(rope_path, RopeConfig())
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.nq == 179, f"expected nq=179 for the scene, got {model.nq}"
    assert model.nv == 138, f"expected nv=138 for the scene, got {model.nv}"
    assert model.nkey == 3
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object1") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stiffrope") >= 0

    data = mujoco.MjData(model)
    rope_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "freejoint_A")
    rope_qpos_address = int(model.jnt_qposadr[rope_joint_id])
    arm_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link{index}") for index in range(8)
    ] + [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("hand", "left_finger", "right_finger")
    ]
    arm_body_ids = np.asarray([body_id for body_id in arm_body_ids if body_id >= 0], dtype=np.int32)
    rope_geoms = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"G{index}") for index in range(40)
    }
    for key_id in range(model.nkey):
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        assert np.all(np.isfinite(data.qpos))
        np.testing.assert_allclose(data.qpos[rope_qpos_address:rope_qpos_address + 3], RopeConfig().initial_position)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if contact.geom1 in rope_geoms:
                other_geom = int(contact.geom2)
            elif contact.geom2 in rope_geoms:
                other_geom = int(contact.geom1)
            else:
                continue
            other_body = int(model.geom_bodyid[other_geom])
            assert other_body not in arm_body_ids, "cable starts in contact with the Panda arm"


def check_compiled_controller() -> None:
    from .controller import DloRope

    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "franka_emika_panda" / "mjx_single_cube_dlo.xml"))
    model.opt.timestep = 0.0015
    model.opt.iterations = 30
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    rope = DloRope(model, data, segments=40)
    for _ in range(100):
        mujoco.mj_step1(model, data)
        rope.update_torque()
        mujoco.mj_step2(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    assert math.isfinite(rope.dlo_math.calculateEnergy())
    length = np.linalg.norm(np.diff(data.site_xpos[rope.site_ids], axis=0), axis=1).sum()
    assert abs(length - 2.0) < 0.05, f"rope length changed to {length} m"


def main() -> None:
    check_xml_generator()
    check_scene_compile()
    check_compiled_controller()
    print("DER cable checks passed")


if __name__ == "__main__":
    main()
