import numpy as np
import mujoco

class PandaEnvController:
    def __init__(self, mjm: mujoco.MjModel, d: mujoco.MjData):
        self.mjm = mjm
        self.d = d
        
        # Define scaling factor
        self.max_delta_q = 0.05  # Max joint angle change per step (radians)
        
        self.num_arm_actions = 7

    def step_action(self, action: np.ndarray):
        """
        Applies an 8-dim action vector [delta_q(7), gripper(1)] to the simulation.
        """
        # copy current positions
        current_ctrl = self.d.ctrl.copy()

        # Scale and apply changes
        arm_action = action[:self.num_arm_actions]
        current_ctrl[:self.num_arm_actions] += arm_action * self.max_delta_q

        # action > 0 -> open, if < 0 -> close
        gripper_command = 0.04 if action[7] > 0 else 0.0
        current_ctrl[7] = gripper_command
        current_ctrl[8] = gripper_command

        self.d.ctrl[:] = current_ctrl

        mujoco.mj_step(self.mjm, self.d)