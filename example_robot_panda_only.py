# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Display only the Franka Panda robot in the Newton viewer."""

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


class Example:
    """A static Panda model with no props, floor, or simulation."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.model = self._build_model()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
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
        return builder.finalize()

    def step(self):
        """Keep the robot at its spawn pose."""

    def render(self):
        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=10_000)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer)
    newton.examples.run(example, args)
