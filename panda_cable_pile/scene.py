# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Robot and cable-pile model construction."""

import numpy as np
import warp as wp
import newton
import newton.ik as ik

from .constants import (
    HOME_Q, PANDA_BASE, HAND_COLLISION_LINKS, TCP_OFFSET, FINGER_OPEN,
    CABLE_RADIUS, CABLE_LENGTH, CABLE_SEGMENTS, YOUNGS_MODULUS, CABLE_MODULI,
    BEND_SCALE, CABLE_MU, CABLE_KE, CABLE_KD, BEND_DAMPING, STRETCH_DAMPING,
    CONTACT_GAP, HAND_MU, HAND_KE, COLOR_KNOT, COLOR_LINKED, COLOR_FREE, COLOR_OTHER,
    PILE_CENTER, SIM_ITERATIONS, FRICTION_EPS,
)
from .geometry import chain_points, free_cable_points, random_cable_points, _zero_mass
from .kernels import quat_to_vec4

class SceneMixin:
    def _build_panda(self):
        builder = newton.ModelBuilder()
        asset = newton.utils.download_asset('franka_emika_panda')
        builder.add_urdf(str(asset / 'urdf' / 'fr3_franka_hand.urdf'), xform=wp.transform(PANDA_BASE, wp.quat_identity()), floating=False, enable_self_collisions=False, collapse_fixed_joints=True)
        builder.joint_q[:len(HOME_Q)] = HOME_Q
        builder.add_ground_plane()
        self._panda_builder = builder
        self.hand_body = next((i for i, l in enumerate(builder.body_label) if l.endswith('/fr3_link7')))
        self.model = builder.finalize()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def _setup_ik(self):
        self.pos_obj = ik.IKObjectivePosition(link_index=self.hand_body, link_offset=wp.vec3(*TCP_OFFSET), target_positions=wp.array([wp.vec3(0.0, 0.0, 0.5)], dtype=wp.vec3))
        self.rot_obj = ik.IKObjectiveRotation(link_index=self.hand_body, link_offset_rotation=wp.quat_identity(), target_rotations=wp.array([wp.vec4(1.0, 0.0, 0.0, 0.0)], dtype=wp.vec4))
        self.limit_obj = ik.IKObjectiveJointLimit(joint_limit_lower=self.model.joint_limit_lower, joint_limit_upper=self.model.joint_limit_upper)
        self.joint_q_ik = wp.array(np.array([HOME_Q], dtype=np.float32), dtype=wp.float32)
        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(model=self.model, n_problems=1, objectives=[self.pos_obj, self.rot_obj, self.limit_obj], lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)

    @staticmethod
    def hand_rotation(yaw_a: float) -> wp.quat:
        """Gripper pointing down; ``yaw_a`` is the rotation about the tool z."""
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi)
        return wp.mul(q, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw_a))

    @staticmethod
    def yaw_for_cable_direction(direction_xy) -> float:
        """Hand angle that makes the fingers open perpendicular to a cable running along ``direction_xy``.

            Measured: with this hand_rotation the finger axis points at 3*pi/4 - yaw in the xy plane.
            """
        phi = float(np.arctan2(direction_xy[1], direction_xy[0]))
        return np.pi / 4 - phi

    def solve_ik(self, tcp: np.ndarray, yaw_a: float, iters: int | None=None):
        self.pos_obj.set_target_positions(wp.array([wp.vec3(*tcp)], dtype=wp.vec3))
        self.rot_obj.set_target_rotations(wp.array([quat_to_vec4(self.hand_rotation(yaw_a))], dtype=wp.vec4))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=iters or self.ik_iters)

    def tcp_position(self) -> np.ndarray:
        xf = wp.transform(*self.state.body_q.numpy()[self.hand_body])
        return np.array(wp.transform_point(xf, wp.vec3(*TCP_OFFSET)))

    def set_arm(self, finger_width: float):
        """Write IK solution + finger width into the Panda model and update FK."""
        q = self.joint_q_ik.numpy()[0].copy()
        q[7] = q[8] = finger_width
        self.model.joint_q.assign(q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def _add_cable(self, builder, pts, label, youngs_modulus=YOUNGS_MODULUS):
        stretch, bend = newton.utils.create_cable_stiffness_from_elastic_moduli(youngs_modulus, CABLE_RADIUS)
        bend *= BEND_SCALE.get(label, 1.0)
        bodies, _ = builder.add_rod(positions=[wp.vec3(*p) for p in pts], radius=CABLE_RADIUS, bend_stiffness=bend, bend_damping=BEND_DAMPING, stretch_stiffness=stretch, stretch_damping=STRETCH_DAMPING, label=label)
        seg = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        return (bodies, seg)

    def _build_cables(self, rng):
        center = PILE_CENTER
        chain, _ = chain_points(-0.03, 0.0)
        if self.unlink_control:
            chain['B'] = chain['B'] + np.array([0.12, 0.0, 0.0])
            chain['C'] = chain['C'] + np.array([0.12, 0.0, 0.0])
        knot_pts = chain['A']
        free_pts = free_cable_points(center + np.array([0.11, -0.03]))
        specs = [(chain['A'], 'cable_A', COLOR_KNOT), (chain['B'], 'cable_B', COLOR_LINKED), (chain['C'], 'cable_C', COLOR_LINKED), (free_pts, 'cable_free', COLOR_FREE)]
        self.grasp_info = {'knot': self._grasp_from_tail(knot_pts, -1), 'free': self._grasp_from_tail(free_pts, -1)}
        keep_clear = [self.grasp_info[k]['point'][:2] for k in self.grasp_info]
        for k in range(max(0, self.num_cables - 4)):
            for _ in range(50):
                pts = random_cable_points(rng, center, z0=0.09 + 0.014 * k)
                if all((np.min(np.linalg.norm(pts[:, :2] - c, axis=1)) > 0.06 for c in keep_clear)):
                    break
            specs.append((pts, f'cable_{k}', COLOR_OTHER))
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = CABLE_KE
        builder.default_shape_cfg.kd = CABLE_KD
        builder.default_shape_cfg.mu = CABLE_MU
        builder.rigid_gap = CONTACT_GAP
        self.cables = []
        for pts, label, colour in specs:
            modulus = CABLE_MODULI.get(label, YOUNGS_MODULUS)
            bodies, seg = self._add_cable(builder, pts, label, modulus)
            self.cables.append({'name': label, 'bodies': bodies, 'seg': seg, 'colour': colour, 'pts0': pts})
        self._add_hand_mirrors(builder)
        builder.add_ground_plane()
        builder.color()
        cable_body_ids = {int(body) for cable in self.cables for body in cable['bodies']}
        for shape_id, body_id in enumerate(builder.shape_body):
            if int(body_id) not in cable_body_ids:
                builder.shape_flags[shape_id] &= ~int(newton.ShapeFlags.VISIBLE)
        self.cable_model = builder.finalize()
        self._setup_rgbd_camera()
        self.cable_solver = newton.solvers.SolverVBD(self.cable_model, iterations=SIM_ITERATIONS, friction_epsilon=FRICTION_EPS)
        self.cable_state_0 = self.cable_model.state()
        self.cable_state_1 = self.cable_model.state()
        self.cable_control = self.cable_model.control()
        self.cable_contacts = self.cable_model.contacts()
        inv_mass = self.cable_model.body_inv_mass.numpy()
        self.dynamic_cable_bodies = [b for c in self.cables for b in c['bodies'] if inv_mass[b] > 0.0]
        self.capsule_buffers = []
        for c in self.cables:
            n = len(c['bodies'])
            self.capsule_buffers.append((wp.empty(n, dtype=wp.transform), wp.full(n, wp.vec3(CABLE_RADIUS, CABLE_RADIUS, 0.5 * c['seg']), dtype=wp.vec3), wp.full(n, wp.vec3(*c['colour']), dtype=wp.vec3)))
        self.sim_time = 0.0
        self.frame = 0

    @staticmethod
    def _grasp_from_tail(pts: np.ndarray, end_index: int) -> dict:
        """Grasp ~3 segments behind the tip; pull direction continues along the tail."""
        i_tip = len(pts) - 1 if end_index == -1 else 0
        step = -1 if end_index == -1 else 1
        i_grasp = i_tip + 3 * step
        i_ref = i_tip + 6 * step
        point = pts[i_grasp].copy()
        direction = pts[i_grasp] - pts[i_ref]
        direction[2] = 0.0
        direction /= np.linalg.norm(direction)
        return {'point': point, 'dir': direction, 'index': i_grasp}

    def _add_hand_mirrors(self, builder):
        panda = self._panda_builder
        panda_q = self.state.body_q.numpy()
        mirror_shapes, hand_bodies, mirror_bodies = ([], [], [])
        hand_cfg = builder.default_shape_cfg.copy()
        hand_cfg.mu = HAND_MU
        hand_cfg.ke = HAND_KE
        for suffix in HAND_COLLISION_LINKS:
            pb = next((i for i, l in enumerate(panda.body_label) if l.endswith(suffix)))
            mirror = builder.add_body(xform=wp.transform(*panda_q[pb]), is_kinematic=True, label=f'mirror_{suffix}')
            for sh in range(panda.shape_count):
                if panda.shape_body[sh] != pb or not panda.shape_flags[sh] & int(newton.ShapeFlags.COLLIDE_SHAPES):
                    continue
                st = int(panda.shape_type[sh])
                if st == int(newton.GeoType.MESH):
                    st = int(newton.GeoType.CONVEX_MESH)
                mirror_shapes.append(builder.add_shape(mirror, st, xform=panda.shape_transform[sh], scale=panda.shape_scale[sh], src=panda.shape_source[sh], cfg=hand_cfg))
            _zero_mass(builder, mirror)
            hand_bodies.append(pb)
            mirror_bodies.append(mirror)
        for i, a in enumerate(mirror_shapes):
            for b in mirror_shapes[i + 1:]:
                builder.add_shape_collision_filter_pair(a, b)
        if self.filter_mirrors:
            for c in self.cables:
                for body in c['bodies']:
                    for cs in builder.body_shapes[body]:
                        for ms in mirror_shapes:
                            builder.add_shape_collision_filter_pair(cs, ms)
        self.hand_bodies = wp.array(hand_bodies, dtype=int)
        self.mirror_bodies = wp.array(mirror_bodies, dtype=int)
        self.hand_prev_q = wp.array(panda_q[hand_bodies], dtype=wp.transform)
        self.mirror_shapes = mirror_shapes
