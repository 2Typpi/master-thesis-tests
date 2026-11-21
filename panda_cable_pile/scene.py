# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Robot and cable-pile model construction."""

import numpy as np
import warp as wp
import newton
import newton.ik as ik

from .constants import (
    CABLE_RADIUS, YOUNGS_MODULUS, CABLE_MODULI,
    BEND_SCALE, CABLE_MU, CABLE_KE, CABLE_KD, BEND_DAMPING, STRETCH_DAMPING,
    CONTACT_GAP, HAND_MU, HAND_KE, COLOR_KNOT, COLOR_LINKED, COLOR_FREE, COLOR_OTHER,
    SIM_ITERATIONS, FRICTION_EPS,
    CABLE_PALETTE, UNIFORM_COLOR_NAME, ROLE_BY_LABEL, ROLE_DEFAULT, ROLE_COLOR_NAMES,
)
from .geometry import chain_points, free_cable_points, random_cable_points, _zero_mass
from .kernels import quat_to_vec4
from .robots import SEGMENT_LENGTH

class SceneMixin:
    def _build_panda(self):
        """Build the robot model (name kept for history; it loads whichever ``self.robot`` is selected)."""
        robot = self.robot
        builder = newton.ModelBuilder()
        robot.load(builder, wp.transform(robot.scene.base_pos, wp.quat_identity()))
        builder.joint_q[:len(robot.home_q)] = robot.home_q
        builder.add_ground_plane()
        self._panda_builder = builder
        self.hand_body = next((i for i, l in enumerate(builder.body_label) if l.endswith(robot.hand_body_suffix)))
        self.model = builder.finalize()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def _setup_ik(self):
        robot = self.robot
        assert robot.ik_mode == 'pose6d', f'unsupported ik_mode {robot.ik_mode!r}'
        self.pos_obj = ik.IKObjectivePosition(link_index=self.hand_body, link_offset=wp.vec3(*robot.tcp_offset), target_positions=wp.array([wp.vec3(0.0, 0.0, 0.5)], dtype=wp.vec3))
        self.rot_obj = ik.IKObjectiveRotation(link_index=self.hand_body, link_offset_rotation=wp.quat_identity(), target_rotations=wp.array([wp.vec4(1.0, 0.0, 0.0, 0.0)], dtype=wp.vec4), weight=robot.ik_rot_weight)
        self.limit_obj = ik.IKObjectiveJointLimit(joint_limit_lower=self.model.joint_limit_lower, joint_limit_upper=self.model.joint_limit_upper)
        self.joint_q_ik = wp.array(np.array([robot.home_q], dtype=np.float32), dtype=wp.float32)
        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(model=self.model, n_problems=1, objectives=[self.pos_obj, self.rot_obj, self.limit_obj], lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)

    def hand_rotation(self, yaw_a: float) -> wp.quat:
        """Tool orientation for ``yaw_a`` (robot-specific convention)."""
        return self.robot.hand_rotation(yaw_a)

    def yaw_for_cable_direction(self, direction_xy) -> float:
        """Hand angle that makes the gripper open perpendicular to a cable running along ``direction_xy``."""
        return self.robot.yaw_for_cable_direction(direction_xy)

    def select_yaw_branch(self, tcp: np.ndarray, yaw_a: float) -> float:
        """Pick the equivalent yaw (shifted by multiples of ``yaw_period``) whose wrist roll is closest to the current one.

        Only for grippers with ``yaw_period`` set; returns ``yaw_a`` unchanged otherwise (Panda path).
        """
        r = self.robot
        if r.yaw_period is None:
            return yaw_a
        base = r.scene.base_pos
        azimuth = float(np.arctan2(tcp[1] - base[1], tcp[0] - base[0]))
        cur = float(self.joint_q_ik.numpy()[0][r.roll_dof])
        lo, hi = (r.roll_limits[0] + 0.25, r.roll_limits[1] - 0.25)  # margin covers the ~0.1 rad azimuth-model error
        best = None
        for k in range(-4, 5):
            for m in (-1, 0, 1):
                roll = yaw_a + k * r.yaw_period - azimuth - r.roll_offset + 2.0 * np.pi * m
                if lo <= roll <= hi and (best is None or abs(roll - cur) < abs(best - cur)):
                    best = roll
        if best is None:  # no branch inside the limits; fall back to the nearest allowed roll
            best = float(np.clip(yaw_a - azimuth - r.roll_offset, lo, hi))
        return best + azimuth + r.roll_offset

    def solve_ik(self, tcp: np.ndarray, yaw_a: float, iters: int | None=None):
        yaw_a = self.select_yaw_branch(tcp, yaw_a)
        self.pos_obj.set_target_positions(wp.array([wp.vec3(*tcp)], dtype=wp.vec3))
        self.rot_obj.set_target_rotations(wp.array([quat_to_vec4(self.hand_rotation(yaw_a))], dtype=wp.vec4))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=iters or self.ik_iters)

    def tcp_position(self) -> np.ndarray:
        xf = wp.transform(*self.state.body_q.numpy()[self.hand_body])
        return np.array(wp.transform_point(xf, wp.vec3(*self.robot.tcp_offset)))

    def set_arm(self, finger_width: float):
        """Write IK solution + gripper command into the robot model and update FK."""
        q = self.joint_q_ik.numpy()[0].copy()
        for dof in self.robot.gripper_dofs:
            q[dof] = finger_width
        self.model.joint_q.assign(q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def _add_cable(self, builder, pts, label, youngs_modulus=YOUNGS_MODULUS):
        stretch, bend = newton.utils.create_cable_stiffness_from_elastic_moduli(youngs_modulus, CABLE_RADIUS)
        bend *= BEND_SCALE.get(label, 1.0)
        bodies, _ = builder.add_rod(positions=[wp.vec3(*p) for p in pts], radius=CABLE_RADIUS, bend_stiffness=bend, bend_damping=BEND_DAMPING, stretch_stiffness=stretch, stretch_damping=STRETCH_DAMPING, label=label)
        seg = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        return (bodies, seg)

    def _cable_attrs(self, label, role_rgb, attr_rng) -> dict:
        """Visual attributes of one cable. ``role`` is simulator-internal; ``colour_name``/``rgb`` is what the camera shows.

        Draw order follows the cable order, so a seed always gives the same colours. The draw happens in every
        render mode, so switching modes never changes the colours of the other modes.
        """
        role = ROLE_BY_LABEL.get(label, ROLE_DEFAULT)
        names = list(CABLE_PALETTE)
        drawn = names[int(attr_rng.integers(len(names)))]
        if self.render_mode == 'role':
            return {'role': role, 'colour_name': ROLE_COLOR_NAMES[role], 'rgb': tuple(float(c) for c in role_rgb)}
        name = UNIFORM_COLOR_NAME if self.render_mode == 'uniform' else drawn
        return {'role': role, 'colour_name': name, 'rgb': CABLE_PALETTE[name]}

    def _build_cables(self, rng):
        sc = self.robot.scene
        center = np.array(sc.pile_center)
        n_seg = sc.cable_segments
        chain, _ = chain_points(*sc.chain_origin, R=sc.chain_R, tail=sc.chain_tail, body_ab=sc.chain_body_ab, tail_c=sc.chain_tail_c, seg=SEGMENT_LENGTH)
        if self.unlink_control:
            chain['B'] = chain['B'] + np.array([sc.unlink_shift, 0.0, 0.0])
            chain['C'] = chain['C'] + np.array([sc.unlink_shift, 0.0, 0.0])
        knot_pts = chain['A']
        free_pts = free_cable_points(center + np.array(sc.free_offset), length=sc.cable_length, n_segments=n_seg, reach=sc.free_reach)
        specs = [(chain['A'], 'cable_A', COLOR_KNOT), (chain['B'], 'cable_B', COLOR_LINKED), (chain['C'], 'cable_C', COLOR_LINKED), (free_pts, 'cable_free', COLOR_FREE)]
        self.grasp_info = {'knot': self._grasp_from_tail(knot_pts, -1), 'free': self._grasp_from_tail(free_pts, -1)}
        keep_clear = [self.grasp_info[k]['point'][:2] for k in self.grasp_info]
        for k in range(max(0, self.num_cables - 4)):
            for _ in range(50):
                pts = random_cable_points(rng, center, z0=0.09 + 0.014 * k, length=sc.cable_length, n_segments=n_seg, spread=sc.random_spread)
                if all((np.min(np.linalg.norm(pts[:, :2] - c, axis=1)) > 0.06 for c in keep_clear)):
                    break
            specs.append((pts, f'cable_{k}', COLOR_OTHER))
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = CABLE_KE
        builder.default_shape_cfg.kd = CABLE_KD
        builder.default_shape_cfg.mu = CABLE_MU
        builder.rigid_gap = CONTACT_GAP
        self.cables = []
        attr_rng = np.random.default_rng([self.seed, 1])  # separate stream: colours must not change the pile layout
        for pts, label, colour in specs:
            modulus = CABLE_MODULI.get(label, YOUNGS_MODULUS)
            bodies, seg = self._add_cable(builder, pts, label, modulus)
            self.cables.append({'name': label, 'bodies': bodies, 'seg': seg, 'colour': colour, 'pts0': pts, 'attrs': self._cable_attrs(label, colour, attr_rng)})
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
            self.capsule_buffers.append((wp.empty(n, dtype=wp.transform), wp.full(n, wp.vec3(CABLE_RADIUS, CABLE_RADIUS, 0.5 * c['seg']), dtype=wp.vec3), wp.full(n, wp.vec3(*c['attrs']['rgb']), dtype=wp.vec3)))
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
        finger_mirrors = []
        hand_cfg = builder.default_shape_cfg.copy()
        hand_cfg.mu = HAND_MU
        hand_cfg.ke = HAND_KE
        for suffix in self.robot.mirror_links:
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
            if suffix in self.robot.finger_links:
                finger_mirrors.append(mirror)
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
        self.finger_mirrors = finger_mirrors
        self.hand_prev_q = wp.array(panda_q[hand_bodies], dtype=wp.transform)
        self.mirror_shapes = mirror_shapes
