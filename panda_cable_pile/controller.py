# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scripted grasp/pull controller and pile metrics."""

import numpy as np
import warp as wp

from .constants import FRAME_DT, SIM_DT, FINGER_OPEN, GRIP_TARGET_NORMAL_FORCE

class ControllerMixin:
    PHASES = [('settle', None), ('pregrasp', 1.5), ('descend', 1.0), ('close', 0.8), ('hold', 0.3), ('pull', 2.5), ('lift', 1.0), ('release', 0.5), ('retreat', 0.6)]

    MIN_SETTLE = 0.5

    REST_LIN = 0.005

    REST_ANG = 0.05

    REST_HOLD = 0.3

    PREGRASP_HEIGHT = 0.12

    GRASP_HEIGHT = 0.008

    PULL_LENGTH = 0.25

    FINGER_CLOSED = 0.002
    GRIP_CLOSE_RATE = 0.1
    GRIP_RECLOSE_RATE = 0.005  # m/s; gently tighten if pinch force drops while pulling

    LIFT_HEIGHT = 0.30

    def _init_controller(self):
        self.phase_idx = 0
        self.phase_time = 0.0
        self.settle_duration = self.settle_time
        self.rest_time = 0.0
        self.settled_at = None
        self.target = None
        self.attached = None
        self.grip_width = FINGER_OPEN
        self.grip_force = 0.0
        self.grip_force_by_finger = {}
        self.grip_force_reached = False
        self.ik_residuals = {}
        self.cmd = (self.tcp_position(), self.yaw_for_cable_direction((1.0, 0.0)), FINGER_OPEN)
        self.metrics_before = None
        self.metrics_after = None
        self.links_after = None
        self.links_before = None

    def max_cable_speed(self) -> tuple[float, float]:
        """(max linear [m/s], max angular [rad/s]) body speed over all non-kinematic cable bodies."""
        qd = self.cable_state_0.body_qd.numpy()
        idx = self.dynamic_cable_bodies
        return (float(np.abs(qd[idx, :3]).max()), float(np.abs(qd[idx, 3:]).max()))

    def cable_by_name(self, name):
        return next((c for c in self.cables if c['name'] == name))

    def cable_starts(self, cable) -> np.ndarray:
        q = self.cable_state_0.body_q.numpy()
        return q[cable['bodies'], :3]

    def _plan_grasp(self):
        cable = self.cable_by_name('cable_A' if self.pull == 'knot' else 'cable_free')
        starts = self.cable_starts(cable)
        n = len(starts)
        point = starts[n - 4].copy()
        direction = starts[n - 4] - starts[n - 7]
        direction[2] = 0.0
        direction /= np.linalg.norm(direction)
        point[2] = self.GRASP_HEIGHT
        self.target = {'cable': cable, 'point': point, 'dir': direction, 'seg_index': n - 4}
        self.target['yaw'] = self.yaw_for_cable_direction(direction)
        self.metrics_before = self.pile_metrics()
        self.links_before = self.link_distances()

    def update_grip_force(self, body0, body1, force_on_body1, normals, count):
        """Update per-finger normal force and stop/recover closing at the force target."""
        phase = self.PHASES[self.phase_idx][0]
        if phase not in ('close', 'pull') or self.target is None:
            return

        cable_bodies = set(self.target['cable']['bodies'])
        finger_bodies = set(self.finger_mirrors)
        finger_forces = {body: 0.0 for body in finger_bodies}
        for i in range(count):
            b0, b1 = int(body0[i]), int(body1[i])
            if b0 in finger_bodies and b1 in cable_bodies:
                finger_body = b0
            elif b1 in finger_bodies and b0 in cable_bodies:
                finger_body = b1
            else:
                continue

            normal = normals[i]
            normal_length = np.linalg.norm(normal)
            if normal_length > 0.0:
                finger_forces[finger_body] += abs(float(np.dot(force_on_body1[i], normal / normal_length)))

        self.grip_force_by_finger = finger_forces
        self.grip_force = sum(finger_forces.values())
        if not finger_forces:
            return

        weakest_finger_force = min(finger_forces.values())
        if phase == 'close':
            if weakest_finger_force >= GRIP_TARGET_NORMAL_FORCE:
                self.grip_force_reached = True
        elif weakest_finger_force < GRIP_TARGET_NORMAL_FORCE and self.grip_width > self.FINGER_CLOSED:
            # The simulated fingers are kinematic: compensate for a lost pinch by
            # incrementally reducing the commanded joint width during the pull.
            self.grip_width = max(
                self.FINGER_CLOSED,
                self.grip_width - self.GRIP_RECLOSE_RATE * SIM_DT,
            )

    def controller_command(self):
        """Scripted state machine -> (tcp [3], hand yaw, finger width)."""
        name, duration = self.PHASES[self.phase_idx]
        if name == 'settle':
            return self.cmd
        T = self.target
        u = min(self.phase_time / duration, 1.0)
        s = u * u * (3.0 - 2.0 * u)
        above = T['point'] + np.array([0.0, 0.0, self.PREGRASP_HEIGHT - self.GRASP_HEIGHT])
        pulled = T['point'] + T['dir'] * self.PULL_LENGTH + np.array([0.0, 0.0, 0.01])
        lifted = pulled + np.array([0.0, 0.0, self.LIFT_HEIGHT])
        yaw = T['yaw']
        if name == 'pregrasp':
            tcp = self.home_tcp * (1 - s) + above * s
            yaw = self.home_yaw * (1 - s) + yaw * s
            return (tcp, yaw, FINGER_OPEN)
        if name == 'descend':
            return (above * (1 - s) + T['point'] * s, yaw, FINGER_OPEN)
        if name == 'close':
            if not self.grip_force_reached:
                self.grip_width = max(self.FINGER_CLOSED, self.grip_width - self.GRIP_CLOSE_RATE * FRAME_DT)
            return (T['point'], yaw, self.grip_width)
        if name == 'hold':
            return (T['point'], yaw, self.grip_width)
        if name == 'pull':
            return (T['point'] * (1 - u) + pulled * u, yaw, self.grip_width)
        if name == 'lift':
            return (pulled * (1 - s) + lifted * s, yaw, self.grip_width)
        if name == 'release':
            width = self.grip_width + (FINGER_OPEN - self.grip_width) * s
            return (lifted, yaw, width)
        return (lifted + np.array([0.0, 0.0, 0.15 * s]), yaw, FINGER_OPEN)

    def _advance_phase(self):
        name, duration = self.PHASES[self.phase_idx]
        if name == 'settle':
            lin, ang = self.max_cable_speed()
            if lin < self.REST_LIN and ang < self.REST_ANG:
                self.rest_time += FRAME_DT
            else:
                self.rest_time = 0.0
            at_rest = self.rest_time >= self.REST_HOLD and self.phase_time >= self.MIN_SETTLE
            if not at_rest:
                if self.phase_time < self.settle_duration:
                    return
                print(f'WARNING: pile not at rest after {self.settle_duration:.1f}s (max v={lin:.4f} m/s, w={ang:.3f} rad/s); starting the arm anyway')
            self.settled_at = self.sim_time
        elif name == 'close':
            at_force_target = self.grip_force_reached
            at_closure_limit = self.grip_width <= self.FINGER_CLOSED + 1e-06
            if not (at_force_target or at_closure_limit):
                return
            if not at_force_target:
                print(
                    f'WARNING: grip target {GRIP_TARGET_NORMAL_FORCE:.1f} N per finger not reached; '
                    f'closure limit reached at {self.grip_width:.4f} m '
                    f'(measured per-finger forces: {self.grip_force_by_finger})'
                )
        elif self.phase_time + 1e-09 < duration:
            return
        elif self.phase_idx == len(self.PHASES) - 1:
            return
        if name in ('pregrasp', 'descend'):
            self.ik_residuals[name] = float(np.linalg.norm(self.tcp_position() - self.cmd[0]))
        if name == 'hold' and self.grasp == 'sticky':
            self._attach_segment()
        if name == 'pull':
            self.metrics_after = self.pile_metrics()
            self.links_after = self.link_distances()
            self.held_distance = self._held_distance()
        if name == 'release':
            self.attached = None
        self.phase_idx += 1
        self.phase_time = 0.0
        new = self.PHASES[self.phase_idx][0]
        if new == 'close':
            self.grip_width = FINGER_OPEN
            self.grip_force = 0.0
            self.grip_force_by_finger = {}
            self.grip_force_reached = False
        if new == 'pregrasp':
            self._plan_grasp()
            self.home_tcp = self.tcp_position()
            self.home_yaw = self.cmd[1]
        elif new == 'descend':
            before = (self.metrics_before, self.links_before)
            self._plan_grasp()
            self.metrics_before, self.links_before = before

    def _attach_segment(self):
        seg_body = self.target['cable']['bodies'][self.target['seg_index']]
        hand_q = wp.transform(*self.state.body_q.numpy()[self.hand_body])
        seg_q = wp.transform(*self.cable_state_0.body_q.numpy()[seg_body])
        self.attached = (seg_body, wp.transform_multiply(wp.transform_inverse(hand_q), seg_q))

    def _held_distance(self) -> float:
        """Distance between the grasped segment and the TCP (small => still held)."""
        T = self.target
        start = self.cable_starts(T['cable'])[T['seg_index']]
        return float(np.linalg.norm(start - self.tcp_position()))

    def cable_metrics(self, cable) -> dict:
        """End-to-end distance / length of one cable."""
        p = self.cable_starts(cable)
        length = cable['seg'] * (len(p) - 1)
        e2e = float(np.linalg.norm(p[-1] - p[0]))
        return {'ratio': e2e / length, 'e2e': e2e, 'length': length}

    def centroids(self) -> dict:
        """Centroid [x, y, z] of every cable, by name."""
        return {c['name']: self.cable_starts(c).mean(axis=0) for c in self.cables}

    def pile_metrics(self) -> dict:
        """Per-cable centroid positions, used to see which cables a pull dragged along."""
        return {'centroids': self.centroids(), 'A': self.cable_metrics(self.cable_by_name('cable_A'))}

    def link_distances(self) -> dict:
        """Closest approach between the centrelines of A-B and B-C.

            Cables that are hooked into each other stay within about one eye radius of each other
            while they are linked; once a link has opened the distance grows without bound.
            """
        pts = {n: self.cable_starts(self.cable_by_name(n)) for n in ('cable_A', 'cable_B', 'cable_C')}

        def closest(a, b):
            return float(np.linalg.norm(a[:, None] - b[None], axis=2).min())
        return {'A-B': closest(pts['cable_A'], pts['cable_B']), 'B-C': closest(pts['cable_B'], pts['cable_C'])}
