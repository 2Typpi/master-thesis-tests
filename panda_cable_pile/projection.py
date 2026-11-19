# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pinhole camera maths matching Newton's ``SensorTiledCamera`` (numpy only, no Newton import).

Conventions, read from ``warp_raytrace/utils.py::compute_pinhole_camera_rays`` and verified by
``check_camera.py``:

* the field of view is the *vertical* one; pixels are square, so ``fx == fy``;
* the camera looks along its local ``-z``; ``+x`` is image right, ``+y`` is image up (OpenGL style);
* pixel ``(px, py)`` has its centre at ``(px + 0.5, py + 0.5)`` and ``py`` grows downwards;
* depth is the *ray-hit distance* from the camera centre, not the z-depth;
* camera poses are camera-to-world, quaternion order ``(x, y, z, w)``.
"""

import numpy as np


def intrinsics(width: int, height: int, fov_y: float) -> np.ndarray:
    """3x3 matrix ``K`` for index coordinates (pixel ``i`` covers ``[i, i+1)``, centre at ``i + 0.5``)."""
    f = 0.5 * height / np.tan(0.5 * fov_y)
    return np.array([[f, 0.0, 0.5 * width], [0.0, f, 0.5 * height], [0.0, 0.0, 1.0]])


def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix for a unit quaternion ``(x, y, z, w)``."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_matrix(position, quat_xyzw) -> np.ndarray:
    """4x4 camera-to-world matrix."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat_xyzw)
    T[:3, 3] = position
    return T


def project(points_world: np.ndarray, T_world_cam: np.ndarray, K: np.ndarray):
    """World points (N, 3) -> ``(px, py, ray_distance, z_depth)``; ``px, py`` are in index coordinates.

    ``z_depth`` is the distance along the optical axis; points behind the camera have ``z_depth <= 0``.
    """
    p = np.asarray(points_world, dtype=float).reshape(-1, 3)
    T_cam_world = np.linalg.inv(T_world_cam)
    pc = p @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
    z = -pc[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        px = K[0, 0] * pc[:, 0] / z + K[0, 2] - 0.5
        py = -K[1, 1] * pc[:, 1] / z + K[1, 2] - 0.5
    return px, py, np.linalg.norm(pc, axis=1), z


def pixel_rays(px, py, K: np.ndarray) -> np.ndarray:
    """Unit ray directions in camera space for index coordinates (any shape), as built by the Newton kernel."""
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    d = np.stack(((px + 0.5 - K[0, 2]) / K[0, 0], -(py + 0.5 - K[1, 2]) / K[1, 1], -np.ones_like(px)), axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def unproject(px, py, ray_distance, T_world_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels + ray-hit distance -> world points, shape ``px.shape + (3,)``."""
    d = pixel_rays(px, py, K) * np.asarray(ray_distance, dtype=float)[..., None]
    return d @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]


def depth_to_zdepth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert a ray-distance image (H, W) to z-depth (distance along the optical axis); 0 stays 0."""
    h, w = depth.shape
    py, px = np.mgrid[0:h, 0:w]
    return depth * -pixel_rays(px, py, K)[..., 2]
