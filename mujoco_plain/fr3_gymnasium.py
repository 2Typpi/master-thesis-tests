import gymnasium as gym
from gymnasium import spaces
import numpy as np

from mujoco_plain.pickplace import Fr3PickPlace


class Fr3GymnasiumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config=None, render_mode=None):
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.task = Fr3PickPlace(config)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.task.action_size,),  # 7 arm deltas + gripper
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.task.observation_size,),  # currently 22 values
            dtype=np.float32,
        )
        self._viewer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation, info = self.task.reset(seed=seed)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.task.step(action)
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(
                self.task.model, self.task.data
            )
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None