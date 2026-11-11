# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Top-level Panda cable-pile simulation and frame stepping."""

import numpy as np
import warp as wp

import newton

from .camera import RGBDCameraMixin
from .constants import HOME_Q, FINGER_OPEN, CABLE_RADIUS, FRAME_DT, SIM_SUBSTEPS, SIM_DT
from .controller import ControllerMixin
from .kernels import _snapshot_hand_poses, _update_hand_mirrors, _drive_attached, _capsule_xforms
from .scene import SceneMixin

class Panda(SceneMixin, ControllerMixin, RGBDCameraMixin):
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args
        self.pull = getattr(args, 'pull', 'knot')
        self.grasp = getattr(args, 'grasp', 'friction')
        self.num_cables = max(4, getattr(args, 'num_cables', 6))
        self.seed = getattr(args, 'seed', 0)
        self.settle_time = getattr(args, 'settle_time', 2.0)
        self.filter_mirrors = getattr(args, 'filter_mirrors', False)
        self.unlink_control = getattr(args, 'unlink_control', False)
        self.device = viewer.device if hasattr(viewer, 'device') else wp.get_device()
        self._build_panda()
        self._setup_ik()
        self.reset(self.seed)
        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        self.viewer.set_camera(wp.vec3(0.75, -0.85, 0.55), pitch=-28.0, yaw=135.0)

    def reset(self, seed: int | None=None):
        """Rebuild the pile for ``seed`` and restart the controller."""
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)
        self.joint_q_ik.assign(np.array([HOME_Q], dtype=np.float32))
        self.set_arm(FINGER_OPEN)
        self._build_cables(rng)
        self._init_controller()

    def apply_command(self, tcp, yaw, width):
        """Arm side of the step: IK to the commanded TCP pose, fingers to ``width``."""
        wp.launch(_snapshot_hand_poses, dim=len(self.hand_bodies), inputs=[self.state.body_q, self.hand_bodies, self.hand_prev_q])
        self.cmd = (np.asarray(tcp, dtype=float), yaw, width)
        self.solve_ik(self.cmd[0], yaw)
        self.set_arm(width)

    def advance_physics(self):
        """Cable side of the step: sweep the mirrors and run the VBD substeps."""
        for sub in range(SIM_SUBSTEPS):
            wp.launch(_update_hand_mirrors, dim=len(self.hand_bodies), inputs=[self.state.body_q, self.hand_bodies, self.hand_prev_q, (sub + 1) / SIM_SUBSTEPS, self.cable_state_0.body_q, self.mirror_bodies])
            if self.attached is not None:
                seg_body, local = self.attached
                wp.launch(_drive_attached, dim=1, inputs=[self.state.body_q, self.hand_body, local, self.cable_state_0.body_q, self.cable_state_0.body_qd, seg_body])
            self.cable_state_0.clear_forces()
            self.cable_model.collide(self.cable_state_0, self.cable_contacts)
            self.cable_solver.step(self.cable_state_0, self.cable_state_1, self.cable_control, self.cable_contacts, SIM_DT)
            self.cable_state_0, self.cable_state_1 = (self.cable_state_1, self.cable_state_0)

    def step(self):
        if self.PHASES[self.phase_idx][0] == 'settle':
            self.apply_command(*self.cmd)
        else:
            self.apply_command(*self.controller_command())
        self.advance_physics()
        if self.settled_at is not None and (not self.rgbd_saved):
            self.save_settled_rgbd()
        self.sim_time += FRAME_DT
        self.frame += 1
        self.phase_time += FRAME_DT
        self._advance_phase()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        for k, c in enumerate(self.cables):
            xf, sc, col = self.capsule_buffers[k]
            wp.launch(_capsule_xforms, dim=len(c['bodies']), inputs=[self.cable_state_0.body_q, c['bodies'][0], 0.5 * c['seg'], xf])
            self.viewer.log_capsules(f"/cables/{c['name']}", '/geometry/capsule', xf, sc, col, None)
        self.viewer.end_frame()

    def test_final(self):
        q = self.cable_state_0.body_q.numpy()
        qd = self.cable_state_0.body_qd.numpy()
        assert np.isfinite(q).all() and np.isfinite(qd).all(), 'non-finite cable state'
        for c in self.cables:
            p = q[c['bodies'], :3]
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            assert (np.abs(seg - c['seg']) < 0.1 * c['seg']).all(), f"{c['name']}: segment length off by >10%"
            assert p[:, 2].min() > -0.02, f"{c['name']} went through the ground"
