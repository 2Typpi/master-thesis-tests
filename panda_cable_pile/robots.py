# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Robot specifications: everything that differs between the supported arms.

``RobotSpec`` bundles asset loading, kinematic constants, gripper convention, IK mode and the
scene preset (pile size, cable length, controller heights) that fits the arm's workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import warp as wp
import newton

SEGMENT_LENGTH = 1.0 / 120.0  # [m] shared by every cable; 50% more segments while staying just above cable diameter


@dataclass(frozen=True)
class ScenePreset:
    """Scene layout and controller heights sized to a robot's workspace [m]."""

    base_pos: tuple = (0.0, 0.0, 0.0)
    pile_center: tuple = (0.0, 0.0)
    cable_length: float = 0.5
    chain_origin: tuple = (-0.03, 0.0)  # (x0, y0) of the interlocked A-B-C chain
    chain_R: float = 0.03  # eye radius
    chain_tail: float = 0.2
    chain_body_ab: float = 0.11
    chain_tail_c: float = 0.12
    free_offset: tuple = (0.11, -0.03)  # free cable position relative to the pile centre
    free_reach: float = 0.2  # how far the free cable's exposed end sticks out along +y
    random_spread: float = 0.07
    unlink_shift: float = 0.12
    pregrasp_height: float = 0.12
    grasp_height: float = 0.008
    pull_length: float = 0.25
    lift_height: float = 0.30
    retreat_rise: float = 0.15
    viewer_camera: tuple = (0.75, -0.85, 0.55)  # interactive viewer eye position

    @property
    def cable_segments(self) -> int:
        return int(round(self.cable_length / SEGMENT_LENGTH))


@dataclass(frozen=True)
class RobotSpec:
    name: str
    load: Callable  # (builder, base_xform) -> None
    home_q: tuple
    gripper_dofs: tuple  # indices into joint_q that receive the gripper command
    hand_body_suffix: str  # label suffix of the body the TCP offset and IK refer to
    tcp_offset: tuple
    mirror_links: tuple  # label suffixes of the bodies mirrored into the cable model
    finger_links: tuple  # subset of mirror_links whose cable contact forces are measured
    grip_open: float
    grip_closed: float
    grip_close_rate: float  # [command units / s]
    grip_reclose_rate: float
    grip_target_force: float  # [N] normal force per finger
    ik_mode: str  # 'pose6d': position + rotation target on the hand body
    hand_rotation: Callable  # (yaw) -> wp.quat
    yaw_for_cable_direction: Callable  # (direction_xy) -> yaw
    scene: ScenePreset = field(default_factory=ScenePreset)
    # Yaw-branch selection for grippers whose grasp is equivalent under a yaw shift of ``yaw_period``
    # (SO-101: either side of the cable works). The branch whose implied wrist roll is closest to the
    # current roll is used, which keeps the roll inside its limits and avoids flips mid-motion.
    yaw_period: float | None = None
    roll_dof: int | None = None  # joint_q index of the wrist roll
    roll_offset: float = 0.0  # roll = yaw - azimuth(base -> tcp) - roll_offset for a vertical tool
    roll_limits: tuple = (-np.pi, np.pi)
    ik_rot_weight: float = 1.0  # < 1 lets position win over tool orientation where the two conflict (5-DOF arm)


def _load_panda(builder, xform):
    asset = newton.utils.download_asset('franka_emika_panda')
    builder.add_urdf(str(asset / 'urdf' / 'fr3_franka_hand.urdf'), xform=xform, floating=False, enable_self_collisions=False, collapse_fixed_joints=True)


def _panda_hand_rotation(yaw_a: float) -> wp.quat:
    """Gripper pointing down; ``yaw_a`` is the rotation about the tool z."""
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi)
    return wp.mul(q, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw_a))


def _panda_yaw_for_cable_direction(direction_xy) -> float:
    """Hand angle that makes the fingers open perpendicular to a cable running along ``direction_xy``.

    Measured: with this hand_rotation the finger axis points at 3*pi/4 - yaw in the xy plane.
    """
    phi = float(np.arctan2(direction_xy[1], direction_xy[0]))
    return np.pi / 4 - phi


PANDA = RobotSpec(
    name='panda',
    load=_load_panda,
    home_q=(0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.785, 0.04, 0.04),
    gripper_dofs=(7, 8),
    hand_body_suffix='/fr3_link7',
    tcp_offset=(0.0, 0.0, 0.21),
    mirror_links=('/fr3_link7', '/fr3_leftfinger', '/fr3_rightfinger'),
    finger_links=('/fr3_leftfinger', '/fr3_rightfinger'),
    grip_open=0.04,
    grip_closed=0.002,
    grip_close_rate=0.1,
    grip_reclose_rate=0.005,
    grip_target_force=25.0,
    ik_mode='pose6d',
    hand_rotation=_panda_hand_rotation,
    yaw_for_cable_direction=_panda_yaw_for_cable_direction,
    scene=ScenePreset(base_pos=(-0.45, 0.0, 0.0)),
)

ASSET_DIR = Path(__file__).resolve().parent.parent / 'assets'


def _load_so101(builder, xform):
    # Menagerie MJCF. The wrist camera mount and its two collision boxes are dropped so they are never mirrored
    # (ignoring only the body is not enough: collapse_fixed_joints would still merge the boxes into the gripper).
    builder.add_mjcf(str(ASSET_DIR / 'so101' / 'so101.xml'), xform=xform, floating=False, enable_self_collisions=False, collapse_fixed_joints=True, ignore_names=['camera_mount', 'camera_box1', 'camera_box2'])


def _so101_hand_rotation(yaw_a: float) -> wp.quat:
    """Tool (gripper -z) pointing down; the jaw opens along the horizontal direction at azimuth ``yaw_a``."""
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw_a)


def _so101_yaw_for_cable_direction(direction_xy) -> float:
    """Jaw opening direction perpendicular to a cable running along ``direction_xy``."""
    phi = float(np.arctan2(direction_xy[1], direction_xy[0]))
    return phi + np.pi / 2


SO101 = RobotSpec(
    name='so101',
    load=_load_so101,
    home_q=(0.0, -0.44, 0.12, 1.87, 0.05, 0.5),  # tool ~vertical, TCP about (0, 0, 0.13) above the pile
    gripper_dofs=(5,),
    hand_body_suffix='/gripper',
    tcp_offset=(-0.003, 0.0, -0.092),  # between the jaw tips, ~9 mm above the fixed-jaw tip
    mirror_links=('/gripper', '/moving_jaw_so101_v1'),
    finger_links=('/gripper', '/moving_jaw_so101_v1'),
    grip_open=0.5,
    grip_closed=-0.17,
    grip_close_rate=1.0,
    grip_reclose_rate=0.05,
    grip_target_force=10.0,
    ik_mode='pose6d',
    hand_rotation=_so101_hand_rotation,
    yaw_for_cable_direction=_so101_yaw_for_cable_direction,
    scene=ScenePreset(
        base_pos=(-0.17, 0.0, 0.0),
        cable_length=0.3,
        chain_R=0.02,
        chain_tail=0.12,
        chain_body_ab=0.07,
        chain_tail_c=0.08,
        free_offset=(0.05, -0.02),
        free_reach=0.12,
        random_spread=0.04,
        unlink_shift=0.08,
        pregrasp_height=0.07,
        grasp_height=0.010,
        pull_length=0.10,
        lift_height=0.2,
        retreat_rise=0.03,
        viewer_camera=(0.4, -0.45, 0.3),
    ),
    yaw_period=np.pi,
    roll_dof=4,
    roll_offset=-0.046,  # mean of roll = yaw - azimuth - offset over the workspace (spread ~0.1 rad)
    roll_limits=(-2.7438, 2.8412),
    ik_rot_weight=0.3,
)

ROBOTS = {'so101': SO101, 'panda': PANDA}
DEFAULT_ROBOT = 'so101'
