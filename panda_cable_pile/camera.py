# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""RGB-D sensor setup and settled-scene image export."""

import math
import numpy as np
import warp as wp
from PIL import Image
from newton.sensors import SensorTiledCamera

class RGBDCameraMixin:
    def _setup_rgbd_camera(self):
        self.rgbd_saved = False
        self.rgbd_width = 1280
        self.rgbd_height = 960
        self.rgbd_sensor = SensorTiledCamera(model=self.cable_model)
        sensor_colors = np.ones((self.cable_model.shape_count, 4), dtype=np.float32)
        shape_bodies = self.cable_model.shape_body.numpy()
        for cable in self.cables:
            cable_bodies = {int(body) for body in cable['bodies']}
            rgb = np.asarray(cable['colour'], dtype=np.float32)
            for shape_id, body_id in enumerate(shape_bodies):
                if int(body_id) in cable_bodies:
                    sensor_colors[shape_id, :3] = rgb
        self.rgbd_sensor.render_context.shape_colors = wp.array(sensor_colors, dtype=wp.vec4f, device=self.device)
        self.rgbd_rays = self.rgbd_sensor.compute_pinhole_camera_rays(self.rgbd_width, self.rgbd_height, math.radians(55.0))
        self.rgbd_colors = self.rgbd_sensor.create_color_image_output(self.rgbd_width, self.rgbd_height)
        self.rgbd_depth = self.rgbd_sensor.create_depth_image_output(self.rgbd_width, self.rgbd_height)
        self.rgbd_camera_transforms = wp.array([[wp.transformf(wp.vec3f(0.0, 0.0, 0.8), wp.quatf(0.0, 0.0, 0.0, 1.0))]], dtype=wp.transformf, device=self.device)

    def save_settled_rgbd(self):
        self.rgbd_sensor.update(self.cable_state_0, self.rgbd_camera_transforms, self.rgbd_rays, color_image=self.rgbd_colors, depth_image=self.rgbd_depth)
        packed = self.rgbd_colors.numpy()[0, 0]
        rgb = np.stack((packed & 255, packed >> 8 & 255, packed >> 16 & 255), axis=-1).astype(np.uint8)
        Image.fromarray(rgb).save('settled_color.png')
        depth = self.rgbd_depth.numpy()[0, 0]
        valid = np.isfinite(depth) & (depth > 0.0)
        depth_gray = np.zeros(depth.shape, dtype=np.uint8)
        if valid.any():
            near, far = np.percentile(depth[valid], [3, 97])
            if far <= near:
                far = near + 1e-06
            scaled = np.clip((depth[valid] - near) / (far - near), 0.0, 1.0)
            depth_gray[valid] = (255 * (1.0 - scaled)).astype(np.uint8)
        Image.fromarray(depth_gray).save('settled_depth.png')
        np.save('settled_depth.npy', depth.astype(np.float32))
        self.rgbd_saved = True
