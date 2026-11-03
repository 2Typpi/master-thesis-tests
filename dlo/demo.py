"""Run the free DER cable in the Panda single-cube scene on CPU."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from .controller import DloRope
from .rope_xml import RopeConfig
from .scene import build_model
from .sim import step


def main() -> None:
    config = RopeConfig()
    model = build_model(config=config)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise ValueError("scene is missing the 'home' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    rope = DloRope(model, data, config.segments)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            step(model, data, rope)
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
