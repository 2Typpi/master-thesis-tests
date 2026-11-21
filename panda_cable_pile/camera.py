# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""RGB-D sensor with ground-truth outputs (instance / segment maps, intrinsics, pose)."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image, ImageDraw

import newton
from newton.sensors import SensorTiledCamera

from . import projection
from .perception import PixelToWorld
from .constants import (
    CAMERA_POSITION, CAMERA_QUAT_XYZW, CAMERA_FOV_Y_DEG, CAMERA_SIZE,
    CAMERA_LIGHT_DIRECTION, CAMERA_TABLE_COLOR,
)

NO_HIT_SHAPE_ID = 0xFFFFFFFF


@dataclass
class Capture:
    """One rendered view. All images are (H, W)."""

    rgb: np.ndarray  # uint8 (H, W, 3)
    depth: np.ndarray  # float32 ray-hit distance [m]; 0 = no hit
    instance: np.ndarray  # int16 cable index into ``Panda.cables``; -1 = background / non-cable
    segment: np.ndarray  # int16 segment index along that cable; -1 where instance == -1
    normal: np.ndarray  # float32 (H, W, 3) world-space surface normal; 0 where no hit
    K: np.ndarray  # 3x3 intrinsics (index coordinates)
    T_world_cam: np.ndarray  # 4x4 camera-to-world
    fov_y_deg: float
    sim_time: float
    at_rest: bool | None  # whether the pile had come to rest when the picture was taken (None = unknown)
    max_cable_speed: tuple[float, float]


class RGBDCameraMixin:
    def _setup_rgbd_camera(self):
        opts = getattr(self, 'camera_options', {}) or {}
        self.rgbd_saved = False
        self.rgbd_width, self.rgbd_height = opts.get('camera_size', CAMERA_SIZE)
        self.rgbd_fov_y_deg = float(opts.get('camera_fov', CAMERA_FOV_Y_DEG))
        position = (CAMERA_POSITION[0], CAMERA_POSITION[1], float(opts.get('camera_height', CAMERA_POSITION[2])))
        self.rgbd_light = opts.get('camera_light', 'ambient')  # 'ambient' (old look) | 'lit' | 'lit-shadows'
        self.rgbd_table = bool(opts.get('camera_table', False))
        assert self.rgbd_light in ('ambient', 'lit', 'lit-shadows'), self.rgbd_light

        # The ground plane is hidden in the cable model by scene.py; show it as a table on request. This must
        # happen before the sensor is built, because the sensor fixes its list of visible shapes at construction.
        shape_type = self.cable_model.shape_type.numpy()
        plane_ids = [int(i) for i in np.flatnonzero(shape_type == int(newton.GeoType.PLANE))]
        if self.rgbd_table:
            flags = self.cable_model.shape_flags.numpy()
            flags[plane_ids] |= int(newton.ShapeFlags.VISIBLE)
            self.cable_model.shape_flags.assign(flags)

        self.rgbd_sensor = SensorTiledCamera(model=self.cable_model)
        if self.rgbd_light != 'ambient':
            self.rgbd_sensor.render_context.utils.create_default_light(self.rgbd_light == 'lit-shadows', wp.vec3f(*CAMERA_LIGHT_DIRECTION))

        # Colours: cables show their *attribute* colour (neutral / uniform / role, see Panda.render_mode).
        n_shapes = self.cable_model.shape_count
        sensor_colors = np.ones((n_shapes, 4), dtype=np.float32)
        shape_bodies = self.cable_model.shape_body.numpy()
        self.rgbd_shape_cable = np.full(n_shapes, -1, dtype=np.int16)
        self.rgbd_shape_segment = np.full(n_shapes, -1, dtype=np.int16)
        body_to_cable = {int(b): (k, i) for k, cable in enumerate(self.cables) for i, b in enumerate(cable['bodies'])}
        for shape_id, body_id in enumerate(shape_bodies):
            hit = body_to_cable.get(int(body_id))
            if hit is None:
                continue
            k, i = hit
            self.rgbd_shape_cable[shape_id] = k
            self.rgbd_shape_segment[shape_id] = i
            sensor_colors[shape_id, :3] = np.asarray(self.cables[k]['attrs']['rgb'], dtype=np.float32)
        for plane in plane_ids:
            sensor_colors[plane, :3] = CAMERA_TABLE_COLOR
        self.rgbd_sensor.render_context.shape_colors = wp.array(sensor_colors, dtype=wp.vec4f, device=self.device)

        self.rgbd_K = projection.intrinsics(self.rgbd_width, self.rgbd_height, math.radians(self.rgbd_fov_y_deg))
        self.rgbd_T_world_cam = projection.pose_matrix(position, CAMERA_QUAT_XYZW)
        self.rgbd_rays = self.rgbd_sensor.compute_pinhole_camera_rays(self.rgbd_width, self.rgbd_height, math.radians(self.rgbd_fov_y_deg))
        self.rgbd_colors = self.rgbd_sensor.create_color_image_output(self.rgbd_width, self.rgbd_height)
        self.rgbd_depth = self.rgbd_sensor.create_depth_image_output(self.rgbd_width, self.rgbd_height)
        self.rgbd_shape_index = self.rgbd_sensor.create_shape_index_image_output(self.rgbd_width, self.rgbd_height)
        self.rgbd_normals = self.rgbd_sensor.create_normal_image_output(self.rgbd_width, self.rgbd_height)
        self.rgbd_camera_transforms = wp.array([[wp.transformf(wp.vec3f(*position), wp.quatf(*CAMERA_QUAT_XYZW))]], dtype=wp.transformf, device=self.device)

    def set_camera_pose(self, position, quat_xyzw=CAMERA_QUAT_XYZW):
        """Move the camera (camera-to-world, quaternion ``(x, y, z, w)``); keeps K, rays and output buffers."""
        position = tuple(float(v) for v in position)
        quat_xyzw = tuple(float(v) for v in quat_xyzw)
        self.rgbd_T_world_cam = projection.pose_matrix(position, quat_xyzw)
        self.rgbd_pixel_map = None  # cached world rays belong to the old pose
        self.rgbd_camera_transforms = wp.array([[wp.transformf(wp.vec3f(*position), wp.quatf(*quat_xyzw))]], dtype=wp.transformf, device=self.device)

    def pixel_map(self) -> PixelToWorld:
        """Per-pixel world rays of the (static) camera; built once, rebuilt only after ``set_camera_pose``."""
        if getattr(self, 'rgbd_pixel_map', None) is None:
            self.rgbd_pixel_map = PixelToWorld(self.rgbd_K, self.rgbd_T_world_cam, self.rgbd_width, self.rgbd_height)
        return self.rgbd_pixel_map

    def world_to_pixel(self, point_world) -> tuple[int, int]:
        """Nearest pixel ``(u, v)`` of a world point (stand-in for a detector while testing)."""
        px, py, _, _ = projection.project(point_world, self.rgbd_T_world_cam, self.rgbd_K)
        return int(round(float(px[0]))), int(round(float(py[0])))

    def grasp_point_from_pixel(self, u: int, v: int, drop: float, z_min: float, depth: np.ndarray | None = None) -> np.ndarray:
        """3D grasp point for a detected pixel, read from the live depth channel (renders a frame unless ``depth`` is given)."""
        if depth is None:
            depth = self.capture().depth
        return self.pixel_map().grasp_point(depth, u, v, drop=drop, z_min=z_min)

    def capture(self) -> Capture:
        """Render the current cable state; returns images plus everything needed to project to / from 3D."""
        self.rgbd_sensor.update(
            self.cable_state_0, self.rgbd_camera_transforms, self.rgbd_rays,
            color_image=self.rgbd_colors, depth_image=self.rgbd_depth,
            shape_index_image=self.rgbd_shape_index, normal_image=self.rgbd_normals,
        )
        packed = self.rgbd_colors.numpy()[0, 0]
        rgb = np.stack((packed & 255, packed >> 8 & 255, packed >> 16 & 255), axis=-1).astype(np.uint8)
        depth = self.rgbd_depth.numpy()[0, 0].astype(np.float32)
        depth[~np.isfinite(depth) | (depth < 0.0)] = 0.0
        shape = self.rgbd_shape_index.numpy()[0, 0]
        hit = shape != NO_HIT_SHAPE_ID
        safe = np.where(hit, shape, 0).astype(np.int64)
        instance = np.where(hit, self.rgbd_shape_cable[safe], -1).astype(np.int16)
        segment = np.where(hit, self.rgbd_shape_segment[safe], -1).astype(np.int16)
        lin, ang = self.max_cable_speed()
        at_rest = getattr(self, 'settled_at_rest', None)
        return Capture(
            rgb=rgb, depth=depth, instance=instance, segment=segment,
            normal=self.rgbd_normals.numpy()[0, 0].astype(np.float32),
            K=self.rgbd_K.copy(), T_world_cam=self.rgbd_T_world_cam.copy(), fov_y_deg=self.rgbd_fov_y_deg,
            sim_time=float(self.sim_time), at_rest=at_rest, max_cable_speed=(lin, ang),
        )

    def save_grasp_marker(self, cap: Capture, u: int, v: int, point_world, out_dir, prefix: str = 'settled_') -> Path:
        """Write ``<prefix>rgb_grasp.png``: the camera RGB with a ring + crosshair at pixel ``(u, v)`` and a caption."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(cap.rgb).convert('RGB')
        draw = ImageDraw.Draw(img)
        r, gap, tick = 16, 5, 30
        # dark outline under a bright colour, so the marker reads on both black background and white cables
        for width, colour in ((5, (0, 0, 0)), (2, (255, 0, 255))):
            draw.ellipse([u - r, v - r, u + r, v + r], outline=colour, width=width)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                draw.line([u + dx * gap, v + dy * gap, u + dx * tick, v + dy * tick], fill=colour, width=width)
        x, y, z = (float(c) for c in point_world)
        text = f'pixel ({u}, {v})  ->  world ({x:.3f}, {y:.3f}, {z:.3f}) m'
        tx, ty = min(u + r + 6, max(img.width - 420, 0)), min(v + r + 6, img.height - 16)
        draw.rectangle([tx - 2, ty - 2, tx + 8 * len(text) * 0.75 + 2, ty + 12], fill=(0, 0, 0))
        draw.text((tx, ty), text, fill=(255, 0, 255))
        path = out / f'{prefix}rgb_grasp.png'
        img.save(path)
        return path

    def cable_table(self) -> list[dict]:
        """JSON-serialisable per-cable record (name, attributes, segment count)."""
        return [{'index': k, 'name': c['name'], 'n_segments': len(c['bodies']), 'attrs': dict(c['attrs'])} for k, c in enumerate(self.cables)]

    def save_capture(self, cap: Capture, out_dir, prefix: str = '') -> Path:
        """Write rgb.png, depth.npy, depth.png, instance.png (+ .npy), segment.npy and meta.json into ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cap.rgb).save(out / f'{prefix}rgb.png')
        np.save(out / f'{prefix}depth.npy', cap.depth)
        np.save(out / f'{prefix}instance.npy', cap.instance)
        np.save(out / f'{prefix}segment.npy', cap.segment)
        valid = cap.depth > 0.0
        depth_gray = np.zeros(cap.depth.shape, dtype=np.uint8)
        preview_scale = {'method': 'ray_distance_percentile', 'range_percentile': [3, 97]}
        if self.rgbd_table:
            # Ray distance includes the extra distance to off-axis pixels, even on a perfectly flat table.
            # Flatten the plane to one neutral gray and map *only cable pixels* by world height above z=0.
            # This makes the preview show cable relief instead of the table's radial ray-distance gradient.
            table = valid & (cap.instance < 0)
            cables = valid & (cap.instance >= 0)
            depth_gray[table] = 24
            if cables.any():
                ys, xs = np.nonzero(cables)
                xyz = projection.unproject(xs, ys, cap.depth[ys, xs], cap.T_world_cam, cap.K)
                height = xyz[:, 2]  # sim table is the horizontal plane z=0
                lo, hi = np.percentile(height, [5, 95])
                if hi <= lo:
                    hi = lo + 1e-6
                # Keep a dark-to-bright span for the cables; height above the table, not camera ray length,
                # determines brightness. The table remains a constant dark gray reference.
                depth_gray[ys, xs] = (50.0 + 185.0 * np.clip((height - lo) / (hi - lo), 0.0, 1.0)).astype(np.uint8)
                preview_scale = {
                    'method': 'table_flat_cable_world_height', 'cable_height_percentile': [5, 95],
                    'cable_height_range_m': [float(lo), float(hi)], 'table_gray': 24,
                    'cable_gray_range': [50, 235],
                }
        elif valid.any():
            near, far = np.percentile(cap.depth[valid], [3, 97])
            if far <= near:
                far = near + 1e-06
            scaled = np.clip((cap.depth[valid] - near) / (far - near), 0.0, 1.0)
            depth_gray[valid] = (255 * (1.0 - scaled)).astype(np.uint8)
        Image.fromarray(depth_gray).save(out / f'{prefix}depth.png')
        palette = np.zeros((len(self.cables) + 1, 3), dtype=np.uint8)
        rng = np.random.default_rng(7)
        palette[1:] = rng.integers(60, 255, size=(len(self.cables), 3))
        Image.fromarray(palette[cap.instance.astype(np.int64) + 1]).save(out / f'{prefix}instance.png')
        meta = {
            'seed': int(self.seed), 'render_mode': self.render_mode, 'size': [int(cap.rgb.shape[1]), int(cap.rgb.shape[0])],
            'K': cap.K.tolist(), 'T_world_cam': cap.T_world_cam.tolist(), 'fov_y_deg': cap.fov_y_deg,
            'depth_semantics': 'ray-hit distance [m] from camera centre; 0 = no hit',
            'depth_preview': preview_scale,
            'sim_time': cap.sim_time, 'at_rest': cap.at_rest, 'max_cable_speed': list(cap.max_cable_speed),
            'camera': {'light': self.rgbd_light, 'table': self.rgbd_table}, 'cables': self.cable_table(),
        }
        (out / f'{prefix}meta.json').write_text(json.dumps(meta, indent=2))
        return out

    def save_settled_rgbd(self, out_dir=None):
        """Capture and write the settled view (thin wrapper kept for ``Panda.step``)."""
        out_dir = out_dir or getattr(self.args, 'rgbd_dir', None) or 'rgbd_out'
        self.save_capture(self.capture(), out_dir, prefix='settled_')
        self.rgbd_saved = True
