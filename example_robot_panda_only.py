# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Display only the Franka Panda robot in the Newton viewer."""

import numpy as np
import warp as wp

import newton
import newton.examples


INITIAL_JOINTS = [
    -3.6802115e-03,
    2.3901723e-02,
    3.6804110e-03,
    -2.3683236e00,
    -1.2918962e-04,
    2.3922248e00,
    7.8549200e-01,
    0.05,
    0.05,
]

CABLE_NUM_SEGMENTS = 24
CABLE_LENGTH = 0.3
CABLE_RADIUS = 0.005
CABLE_BEND_STIFFNESS = 1.0e-5
CABLE_BEND_DAMPING = 1.0e-2
CABLE_STRETCH_STIFFNESS = 1.0e9
SIM_SUBSTEPS = 10
SIM_ITERATIONS = 5
FRAME_DT = 1.0 / 60.0
SIM_DT = FRAME_DT / SIM_SUBSTEPS
ARM_MOVE_DELAY = 0.5
ARM_RAISE_DURATION = 0.5
ARM_RAISE_JOINT = 3
ARM_RAISE_ANGLE = 0.8
WRIST_SPIN_START = ARM_MOVE_DELAY + ARM_RAISE_DURATION
WRIST_SPIN_DURATION = 1.0
WRIST_SPIN_JOINT = 4
WRIST_SPIN_ANGLE = 2.0
CABLE_RELEASE_TIME = WRIST_SPIN_START + WRIST_SPIN_DURATION
HAND_CABLE_OFFSET = wp.vec3(0.0, 0.0, 0.14)
HAND_COLLISION_LINKS = ["/fr3_link7", "/fr3_leftfinger", "/fr3_rightfinger"]


@wp.kernel
def _set_arm_joint(joint_q: wp.array[wp.float32], target_angle: float):
    joint_q[ARM_RAISE_JOINT] = target_angle


@wp.kernel
def _set_wrist_spin_joint(joint_q: wp.array[wp.float32], target_angle: float):
    joint_q[WRIST_SPIN_JOINT] = target_angle


@wp.kernel
def _release_cable_anchor(
    body_mass: wp.array[wp.float32],
    body_inv_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    body_inv_inertia: wp.array[wp.mat33],
    body_qd: wp.array[wp.spatial_vector],
    anchor_body: int,
    anchor_mass: float,
    anchor_inv_mass: float,
    anchor_inertia: wp.mat33,
    anchor_inv_inertia: wp.mat33,
):
    body_mass[anchor_body] = anchor_mass
    body_inv_mass[anchor_body] = anchor_inv_mass
    body_inertia[anchor_body] = anchor_inertia
    body_inv_inertia[anchor_body] = anchor_inv_inertia
    body_qd[anchor_body] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _snapshot_hand_poses(
    hand_body_q: wp.array[wp.transform],
    hand_bodies: wp.array[int],
    prev_q: wp.array[wp.transform],
):
    tid = wp.tid()
    prev_q[tid] = hand_body_q[hand_bodies[tid]]


@wp.kernel
def _update_hand_mirrors(
    hand_body_q: wp.array[wp.transform],
    hand_bodies: wp.array[int],
    prev_q: wp.array[wp.transform],
    alpha: float,
    cable_body_q: wp.array[wp.transform],
    mirror_bodies: wp.array[int],
):
    # Move the kinematic hand copies inside the cable model from the previous
    # frame pose towards the new one, so they sweep smoothly through the substeps.
    tid = wp.tid()
    target = hand_body_q[hand_bodies[tid]]
    prev = prev_q[tid]
    pos = wp.lerp(wp.transform_get_translation(prev), wp.transform_get_translation(target), alpha)
    rot = wp.quat_slerp(wp.transform_get_rotation(prev), wp.transform_get_rotation(target), alpha)
    cable_body_q[mirror_bodies[tid]] = wp.transform(pos, rot)


@wp.kernel
def _update_cable_anchor(
    hand_body_q: wp.array[wp.transform],
    hand_body: int,
    cable_body_q: wp.array[wp.transform],
    cable_anchor_body: int,
):
    hand_xform = hand_body_q[hand_body]
    anchor_position = wp.transform_point(hand_xform, HAND_CABLE_OFFSET)
    cable_xform = cable_body_q[cable_anchor_body]
    cable_body_q[cable_anchor_body] = wp.transform(anchor_position, wp.transform_get_rotation(cable_xform))


class Example:
    """A static Panda with a soft, gravity-driven cable at the hand."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.sim_time = 0.0
        self.model, self.hand_body, self._panda_builder = self._build_model()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.initial_arm_angle = INITIAL_JOINTS[ARM_RAISE_JOINT]
        self.initial_wrist_angle = INITIAL_JOINTS[WRIST_SPIN_JOINT]
        self.raise_start_time = ARM_MOVE_DELAY
        self.raise_end_time = ARM_MOVE_DELAY + ARM_RAISE_DURATION
        self.spin_start_time = WRIST_SPIN_START
        self.cable_released = False
        self.cable_anchor_mass = 0.0
        self.cable_anchor_inv_mass = 0.0
        self.cable_anchor_inertia = wp.mat33(0.0)
        self.cable_anchor_inv_inertia = wp.mat33(0.0)
        self._build_soft_cable()
        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        self.viewer.set_camera(wp.vec3(1.4, -2.0, 1.15), pitch=-22.0, yaw=142.0)

    @staticmethod
    def _build_model():
        builder = newton.ModelBuilder()
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-0.5, -0.5, 0.05), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[: len(INITIAL_JOINTS)] = INITIAL_JOINTS
        builder.add_ground_plane()
        hand_body = next(i for i, label in enumerate(builder.body_label) if label.endswith("/fr3_link7"))
        return builder.finalize(), hand_body, builder

    def _build_soft_cable(self):
        # The cable uses its own VBD model: VBD supports Newton cable joints but
        # not the Panda's revolute joints. Its first segment is kinematically
        # carried by the hand until release; there is no force coupling.
        hand_xform = wp.transform(*self.state.body_q.numpy()[self.hand_body])
        cable_start = wp.transform_point(hand_xform, wp.vec3(0.0, 0.0, 0.14))
        segment_length = CABLE_LENGTH / CABLE_NUM_SEGMENTS
        cable_points = [
            cable_start + wp.vec3(0.0, 0.0, -segment_length * i)
            for i in range(CABLE_NUM_SEGMENTS + 1)
        ]

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1.0e2
        builder.default_shape_cfg.kd = 1.0e1
        builder.default_shape_cfg.mu = 1.0
        cable_bodies, _ = builder.add_rod(
            positions=cable_points,
            radius=CABLE_RADIUS,
            bend_stiffness=CABLE_BEND_STIFFNESS,
            bend_damping=CABLE_BEND_DAMPING,
            stretch_stiffness=CABLE_STRETCH_STIFFNESS,
            stretch_damping=0.0,
            label="soft_cable",
        )

        # Keep the first segment kinematic and move it with the hand until
        # release; at release it becomes a regular dynamic cable segment.
        anchor_body = cable_bodies[0]
        self.cable_anchor_body = anchor_body
        self.cable_anchor_mass = float(builder.body_mass[anchor_body])
        self.cable_anchor_inv_mass = float(builder.body_inv_mass[anchor_body])
        self.cable_anchor_inertia = builder.body_inertia[anchor_body]
        self.cable_anchor_inv_inertia = builder.body_inv_inertia[anchor_body]
        builder.body_mass[anchor_body] = 0.0
        builder.body_inv_mass[anchor_body] = 0.0
        builder.body_inertia[anchor_body] = wp.mat33(0.0)
        builder.body_inv_inertia[anchor_body] = wp.mat33(0.0)

        # Kinematic copies of the hand and fingers so the cable can collide with
        # them. VBD cannot simulate the Panda's revolute joints, so the real arm
        # stays in self.model and these copies are moved to its poses every substep.
        panda = self._panda_builder
        panda_q = self.state.body_q.numpy()
        mirror_shapes = []
        hand_bodies = []
        mirror_bodies = []
        for suffix in HAND_COLLISION_LINKS:
            panda_body = next(i for i, label in enumerate(panda.body_label) if label.endswith(suffix))
            mirror = builder.add_body(
                xform=wp.transform(*panda_q[panda_body]),
                is_kinematic=True,
                label=f"mirror_{suffix}",
            )
            for shape in range(panda.shape_count):
                if panda.shape_body[shape] != panda_body:
                    continue
                if not panda.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES):
                    continue
                shape_type = int(panda.shape_type[shape])
                if shape_type == int(newton.GeoType.MESH):
                    # Convex hulls are much cheaper than triangle meshes on CPU.
                    shape_type = int(newton.GeoType.CONVEX_MESH)
                mirror_shapes.append(
                    builder.add_shape(
                        mirror,
                        shape_type,
                        xform=panda.shape_transform[shape],
                        scale=panda.shape_scale[shape],
                        src=panda.shape_source[shape],
                    )
                )
            builder.body_mass[mirror] = 0.0
            builder.body_inv_mass[mirror] = 0.0
            builder.body_inertia[mirror] = wp.mat33(0.0)
            builder.body_inv_inertia[mirror] = wp.mat33(0.0)
            hand_bodies.append(panda_body)
            mirror_bodies.append(mirror)

        # The hand copies are driven kinematically and must not collide with each other.
        for i, shape_a in enumerate(mirror_shapes):
            for shape_b in mirror_shapes[i + 1 :]:
                builder.add_shape_collision_filter_pair(shape_a, shape_b)

        # The cable starts at the hand, so its first segments must ignore it.
        for segment_body in cable_bodies[:2]:
            for cable_shape in builder.body_shapes[segment_body]:
                for mirror_shape in mirror_shapes:
                    builder.add_shape_collision_filter_pair(cable_shape, mirror_shape)

        builder.add_ground_plane()
        builder.color()

        self.hand_bodies = wp.array(hand_bodies, dtype=int, device=self.viewer.device)
        self.mirror_bodies = wp.array(mirror_bodies, dtype=int, device=self.viewer.device)
        self.hand_prev_q = wp.array(panda_q[hand_bodies], dtype=wp.transform, device=self.viewer.device)

        self.cable_model = builder.finalize()
        self.cable_solver = newton.solvers.SolverVBD(self.cable_model, iterations=SIM_ITERATIONS)
        self.cable_state_0 = self.cable_model.state()
        self.cable_state_1 = self.cable_model.state()
        self.cable_control = self.cable_model.control()
        self.cable_contacts = self.cable_model.contacts()
        self.cable_capsule_xforms = wp.empty(
            CABLE_NUM_SEGMENTS,
            dtype=wp.transform,
            device=self.viewer.device,
        )
        self.cable_capsule_scales = wp.full(
            CABLE_NUM_SEGMENTS,
            wp.vec3(CABLE_RADIUS, CABLE_RADIUS, 0.5 * segment_length),
            dtype=wp.vec3,
            device=self.viewer.device,
        )
        self.cable_capsule_colors = wp.full(
            CABLE_NUM_SEGMENTS,
            wp.vec3(1.0, 0.25, 0.05),
            dtype=wp.vec3,
            device=self.viewer.device,
        )

    def step(self):
        # Remember last frame's hand poses; the mirrors sweep from them to the new ones.
        wp.launch(
            _snapshot_hand_poses,
            dim=len(self.hand_bodies),
            inputs=[self.state.body_q, self.hand_bodies, self.hand_prev_q],
            device=self.model.device,
        )

        if self.sim_time >= self.raise_start_time:
            progress = min(
                (self.sim_time - self.raise_start_time) / ARM_RAISE_DURATION,
                1.0,
            )
            target_angle = self.initial_arm_angle + ARM_RAISE_ANGLE * progress
            wp.launch(
                _set_arm_joint,
                dim=1,
                inputs=[self.model.joint_q, target_angle],
                device=self.model.device,
            )

        if self.sim_time >= self.spin_start_time:
            spin_progress = min(
                (self.sim_time - self.spin_start_time) / WRIST_SPIN_DURATION,
                1.0,
            )
            wrist_angle = self.initial_wrist_angle + WRIST_SPIN_ANGLE * spin_progress
            wp.launch(
                _set_wrist_spin_joint,
                dim=1,
                inputs=[self.model.joint_q, wrist_angle],
                device=self.model.device,
            )

        if self.sim_time >= self.raise_start_time:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
            if not self.cable_released and self.sim_time < CABLE_RELEASE_TIME:
                wp.launch(
                    _update_cable_anchor,
                    dim=1,
                    inputs=[
                        self.state.body_q,
                        self.hand_body,
                        self.cable_state_0.body_q,
                        self.cable_anchor_body,
                    ],
                    device=self.cable_model.device,
                )
            elif not self.cable_released:
                # Place the anchor at the final hand pose before making it dynamic.
                wp.launch(
                    _update_cable_anchor,
                    dim=1,
                    inputs=[
                        self.state.body_q,
                        self.hand_body,
                        self.cable_state_0.body_q,
                        self.cable_anchor_body,
                    ],
                    device=self.cable_model.device,
                )
                wp.launch(
                    _release_cable_anchor,
                    dim=1,
                    inputs=[
                        self.cable_model.body_mass,
                        self.cable_model.body_inv_mass,
                        self.cable_model.body_inertia,
                        self.cable_model.body_inv_inertia,
                        self.cable_state_0.body_qd,
                        self.cable_anchor_body,
                        self.cable_anchor_mass,
                        self.cable_anchor_inv_mass,
                        self.cable_anchor_inertia,
                        self.cable_anchor_inv_inertia,
                    ],
                    device=self.cable_model.device,
                )
                self.cable_released = True

        for substep in range(SIM_SUBSTEPS):
            wp.launch(
                _update_hand_mirrors,
                dim=len(self.hand_bodies),
                inputs=[
                    self.state.body_q,
                    self.hand_bodies,
                    self.hand_prev_q,
                    (substep + 1) / SIM_SUBSTEPS,
                    self.cable_state_0.body_q,
                    self.mirror_bodies,
                ],
                device=self.cable_model.device,
            )
            self.cable_state_0.clear_forces()
            self.cable_model.collide(self.cable_state_0, self.cable_contacts)
            self.cable_solver.step(
                self.cable_state_0,
                self.cable_state_1,
                self.cable_control,
                self.cable_contacts,
                SIM_DT,
            )
            self.cable_state_0, self.cable_state_1 = self.cable_state_1, self.cable_state_0
        self.sim_time += FRAME_DT

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.cable_contacts, self.cable_state_0)
        self.cable_capsule_xforms = self.cable_state_0.body_q[:CABLE_NUM_SEGMENTS]
        self.viewer.log_capsules(
            "/soft_cable",
            "/geometry/capsule",
            self.cable_capsule_xforms,
            self.cable_capsule_scales,
            self.cable_capsule_colors,
            None,
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=10_000)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer)
    newton.examples.run(example, args)
