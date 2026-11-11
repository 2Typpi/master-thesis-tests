# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels used by the cable-pile simulation."""

import warp as wp

def quat_to_vec4(q) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])

@wp.kernel
def _snapshot_hand_poses(hand_body_q: wp.array[wp.transform], hand_bodies: wp.array[int], prev_q: wp.array[wp.transform]):
    tid = wp.tid()
    prev_q[tid] = hand_body_q[hand_bodies[tid]]

@wp.kernel
def _update_hand_mirrors(hand_body_q: wp.array[wp.transform], hand_bodies: wp.array[int], prev_q: wp.array[wp.transform], alpha: float, cable_body_q: wp.array[wp.transform], mirror_bodies: wp.array[int]):
    tid = wp.tid()
    target = hand_body_q[hand_bodies[tid]]
    prev = prev_q[tid]
    pos = wp.lerp(wp.transform_get_translation(prev), wp.transform_get_translation(target), alpha)
    rot = wp.quat_slerp(wp.transform_get_rotation(prev), wp.transform_get_rotation(target), alpha)
    cable_body_q[mirror_bodies[tid]] = wp.transform(pos, rot)

@wp.kernel
def _drive_attached(hand_body_q: wp.array[wp.transform], hand_body: int, local_xform: wp.transform, cable_body_q: wp.array[wp.transform], cable_body_qd: wp.array[wp.spatial_vector], seg_body: int):
    cable_body_q[seg_body] = wp.transform_multiply(hand_body_q[hand_body], local_xform)
    cable_body_qd[seg_body] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

@wp.kernel
def _capsule_xforms(body_q: wp.array[wp.transform], first_body: int, half_height: float, out: wp.array[wp.transform]):
    tid = wp.tid()
    out[tid] = wp.transform_multiply(body_q[first_body + tid], wp.transform(wp.vec3(0.0, 0.0, half_height), wp.quat_identity()))
