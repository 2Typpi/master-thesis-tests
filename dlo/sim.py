"""CPU stepping helpers for MuJoCo models containing a DER cable.

MuJoCo Warp/MJX stepping cannot invoke this Python controller hook; use standard MuJoCo on CPU.
"""

from __future__ import annotations

import mujoco

from .controller import DloRope


def step(model: mujoco.MjModel, data: mujoco.MjData, rope: DloRope) -> None:
    """Advance one simulation step with DER torques injected at the correct stage."""
    mujoco.mj_step1(model, data)
    rope.update_torque()
    mujoco.mj_step2(model, data)


def step_n(model: mujoco.MjModel, data: mujoco.MjData, rope: DloRope, steps: int) -> None:
    """Advance ``steps`` simulation steps with DER torques."""
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    for _ in range(steps):
        step(model, data, rope)
