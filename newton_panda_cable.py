"""CPU-capable Newton Panda with a spring-chain cable attached at the hand.

The cable is represented by particles and damped axial springs so the Panda's
Featherstone solver can advance both the robot and cable without CUDA.
"""

import copy

import warp as wp

import newton
import newton.examples

FPS = 60
SIM_SUBSTEPS = 10
SIM_DT = 1.0 / FPS / SIM_SUBSTEPS
NUM_CABLE_PARTICLES = 25
CABLE_LENGTH = 0.30
CABLE_RADIUS = 0.012
CABLE_PARTICLE_MASS = 0.02
CABLE_SPRING_KE = 250.0
CABLE_SPRING_KD = 2.0
CABLE_BEND_KE = 40.0
CABLE_BEND_KD = 1.0


@wp.kernel
def update_anchor_particle(
    body_q: wp.array(dtype=wp.transform),
    hand_body: int,
    local_anchor: wp.vec3,
    anchor_index: int,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    hand_xform = body_q[hand_body]
    particle_q[anchor_index] = wp.transform_point(hand_xform, local_anchor)
    particle_qd[anchor_index] = wp.vec3(0.0)


class Example:
    def __init__(self, viewer):
        self.viewer = viewer
        self.sim_time = 0.0
        self.cable_indices = []
        builder = newton.ModelBuilder(gravity=-9.81)

        # Load the Panda arm and hand used by Newton's robot example.
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = 2.5e3
        builder.default_shape_cfg.kd = 100.0
        builder.default_shape_cfg.mu = 0.8

        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-0.5, -0.5, 0.05), wp.quat_identity()),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )

        # Start at the Panda example's pickup pose with the gripper open.
        initial_joints = [
            -0.0036802115,
            0.02390172,
            0.0036804110,
            -2.3683236,
            -0.000129,
            2.3922248,
            0.785492,
            0.04,
            0.04,
        ]
        builder.joint_q[: len(initial_joints)] = initial_joints
        builder.joint_target_pos[: len(initial_joints)] = initial_joints
        for dof in range(builder.joint_dof_count):
            builder.joint_target_ke[dof] = 650.0
            builder.joint_target_kd[dof] = 100.0

        # Use convex hulls for Panda collision geometry; mesh SDF contacts need CUDA.
        mesh_colliders = [
            i
            for i, shape_type in enumerate(builder.shape_type)
            if shape_type == newton.GeoType.MESH
            and builder.shape_flags[i] & newton.ShapeFlags.COLLIDE_SHAPES
        ]
        builder.approximate_meshes(
            method="convex_hull",
            shape_indices=mesh_colliders,
            keep_visual_shapes=True,
        )

        # This compact CPU demo resolves particle contacts against the ground,
        # not robot links. Preserve Panda visuals while disabling mesh contacts.
        for i in range(builder.shape_count):
            builder.shape_flags[i] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
            builder.shape_flags[i] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)

        self.hand_body = next(
            i for i, label in enumerate(builder.body_label) if label.endswith("/fr3_link7")
        )
        self.hand_local_anchor = wp.vec3(0.0, 0.0, 0.14)

        # Evaluate the hand pose once so the cable starts attached without an
        # initial spring impulse. The anchor is updated from the live hand pose.
        preview_model = copy.deepcopy(builder).finalize()
        preview_state = preview_model.state()
        newton.eval_fk(
            preview_model,
            preview_model.joint_q,
            preview_model.joint_qd,
            preview_state,
        )
        hand_xform = wp.transform(*preview_state.body_q.numpy()[self.hand_body])
        cable_start = wp.transform_point(hand_xform, self.hand_local_anchor)
        del preview_state, preview_model

        cable_spacing = CABLE_LENGTH / (NUM_CABLE_PARTICLES - 1)
        for i in range(NUM_CABLE_PARTICLES):
            position = cable_start + wp.vec3(0.0, 0.0, -cable_spacing * i)
            mass = 0.0 if i == 0 else CABLE_PARTICLE_MASS
            particle_id = builder.add_particle(
                pos=position,
                vel=wp.vec3(0.0),
                mass=mass,
                radius=CABLE_RADIUS,
            )
            self.cable_indices.append(particle_id)

        # Axial springs preserve cable length; second-neighbor springs add
        # modest bending resistance while keeping the chain flexible.
        for i in range(NUM_CABLE_PARTICLES - 1):
            builder.add_spring(
                self.cable_indices[i],
                self.cable_indices[i + 1],
                ke=CABLE_SPRING_KE,
                kd=CABLE_SPRING_KD,
                control=0.0,
            )
        for i in range(NUM_CABLE_PARTICLES - 2):
            builder.add_spring(
                self.cable_indices[i],
                self.cable_indices[i + 2],
                ke=CABLE_BEND_KE,
                kd=CABLE_BEND_KD,
                control=0.0,
            )

        builder.default_particle_radius = CABLE_RADIUS
        builder.add_ground_plane()
        # Keep the floor available to particles but omit rigid-rigid contacts.
        builder.shape_flags[-1] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        builder.shape_flags[-1] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            rigid_contact_max=1,
            soft_contact_max=64,
            soft_contact_margin=CABLE_RADIUS,
        )
        self.solver = newton.solvers.SolverFeatherstone(self.model)
        self.cable_point_radii = wp.full(
            NUM_CABLE_PARTICLES,
            CABLE_RADIUS,
            dtype=wp.float32,
            device=self.model.device,
        )
        self.cable_point_colors = wp.full(
            NUM_CABLE_PARTICLES,
            wp.vec3(1.0, 0.25, 0.05),
            dtype=wp.vec3,
            device=self.model.device,
        )

        self.viewer.set_model(self.model)
        # Aim at the Panda and hanging cable rather than the scene origin.
        self.viewer.set_camera(wp.vec3(1.4, -2.0, 1.15), pitch=-22.0, yaw=142.0)
        self.viewer.picking_enabled = False

    def _update_cable_anchor(self):
        wp.launch(
            update_anchor_particle,
            dim=1,
            inputs=[
                self.state_0.body_q,
                self.hand_body,
                self.hand_local_anchor,
                self.cable_indices[0],
            ],
            outputs=[self.state_0.particle_q, self.state_0.particle_qd],
            device=self.model.device,
        )

    def step(self):
        for _ in range(SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self._update_cable_anchor()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                SIM_DT,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += SIM_DT

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        points = self.state_0.particle_q[self.cable_indices[0] : self.cable_indices[-1] + 1]
        self.viewer.log_points(
            "/cable/points",
            points,
            radii=self.cable_point_radii,
            colors=self.cable_point_colors,
        )
        self.viewer.log_lines(
            "/cable/segments",
            points[:-1],
            points[1:],
            colors=(1.0, 0.25, 0.05),
            width=2.0 * CABLE_RADIUS,
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=10_000)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer)
    newton.examples.run(example, args)
