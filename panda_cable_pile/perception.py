# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pixel (u, v) + depth image -> 3D grasp point in the world frame (numpy only).

The camera is static, so everything that depends only on the camera (position, rotation, intrinsics) is computed
once: ``PixelToWorld`` caches the unit world-space ray of every pixel. At run time a grasp point is then one
window read on the depth image and one multiply-add (``origin + ray * distance``).

Indexing: images are ``(H, W)``, so the depth at pixel ``(u, v)`` is ``depth[v, u]`` (``u`` = column, ``v`` = row).
Depth is the ray-hit distance from the camera centre in metres (see ``projection``), 0 = no hit.
"""

import numpy as np

from . import projection


class PixelToWorld:
    def __init__(self, K: np.ndarray, T_world_cam: np.ndarray, width: int, height: int):
        self.width, self.height = int(width), int(height)
        self.origin = np.asarray(T_world_cam[:3, 3], dtype=np.float64).copy()
        v, u = np.mgrid[0:self.height, 0:self.width]
        cam_rays = projection.pixel_rays(u, v, K)
        # (H, W, 3) unit directions in the world frame; float32 keeps a 1280x960 cache at ~15 MB
        self.rays = (cam_rays @ T_world_cam[:3, :3].T).astype(np.float32)

    def window_depth(self, depth: np.ndarray, u: int, v: int, window: int = 3) -> float:
        """Smallest valid ray distance in a ``window`` x ``window`` patch around ``(u, v)``; 0 if none is valid.

        The minimum (nearest surface) keeps a 1-pixel detection error at the border of an object from reading
        the table or a cable *behind* it.
        """
        r = window // 2
        patch = depth[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        return float(valid.min()) if valid.size else 0.0

    def point(self, u: int, v: int, ray_distance: float) -> np.ndarray:
        """World point hit by pixel ``(u, v)`` at ``ray_distance`` along its ray."""
        return self.origin + self.rays[v, u].astype(np.float64) * ray_distance

    def grasp_point(self, depth: np.ndarray, u: int, v: int, drop: float = 0.0, z_min: float = -np.inf, window: int = 3) -> np.ndarray:
        """Top-surface hit point under ``(u, v)``, lowered by ``drop`` and clamped to ``z >= z_min``.

        ``drop`` makes the gripper close around the body of the object instead of pinching its top; ``z_min`` keeps
        the fingers above the table.
        """
        u, v = int(round(u)), int(round(v))
        if not (0 <= u < self.width and 0 <= v < self.height):
            raise ValueError(f'pixel ({u}, {v}) is outside the {self.width}x{self.height} image')
        distance = self.window_depth(depth, u, v, window)
        if distance <= 0.0:
            raise ValueError(f'no valid depth around pixel ({u}, {v})')
        hit = self.point(u, v, distance)
        hit[2] = max(hit[2] - drop, z_min)
        return hit
