# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared scene and simulation parameters."""

import numpy as np

# Robot-specific values (home pose, TCP offset, gripper convention, pile layout) live in robots.py.
# The Panda values are re-exported here for backward compatibility.
from .robots import PANDA

HOME_Q = list(PANDA.home_q)

PANDA_BASE = PANDA.scene.base_pos

PILE_CENTER = np.array(PANDA.scene.pile_center)

HAND_COLLISION_LINKS = list(PANDA.mirror_links)

TCP_OFFSET = PANDA.tcp_offset

FINGER_OPEN = PANDA.grip_open

FRAME_DT = 1.0 / 60.0

SIM_SUBSTEPS = 10

SIM_ITERATIONS = 10

SIM_DT = FRAME_DT / SIM_SUBSTEPS

CABLE_RADIUS = 0.004

CABLE_LENGTH = 0.5

CABLE_SEGMENTS = 60

KNOT_SEGMENTS = 60

YOUNGS_MODULUS = 100000000.0

CABLE_MODULI = {'cable_A': 100000000.0, 'cable_B': 60000000.0, 'cable_C': 150000000.0, 'cable_free': 80000000.0}

BEND_SCALE = {'cable_A': 0.25, 'cable_B': 0.8, 'cable_C': 0.8, 'cable_free': 0.1}

CABLE_MU = 3.0

CABLE_KE = 1000000.0

CABLE_KD = 0.1

FRICTION_EPS = 0.001

STRETCH_DAMPING = 0.0001

BEND_DAMPING = 0.05

CONTACT_GAP = 0.004

HAND_MU = 3.0

HAND_KE = 10000.0
GRIP_TARGET_NORMAL_FORCE = 25.0  # N normal force per finger

COLOR_KNOT = (0.9, 0.15, 0.1)

COLOR_LINKED = (0.95, 0.6, 0.1)

COLOR_FREE = (0.1, 0.7, 0.2)

COLOR_OTHER = (0.35, 0.5, 0.85)

# --- RGB-D camera defaults (overridable through Panda(args) / _setup_rgbd_camera) ---
CAMERA_POSITION = (0.0, 0.0, 0.8)
CAMERA_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)  # identity: looks straight down (-z)
CAMERA_FOV_Y_DEG = 55.0  # VERTICAL field of view (Newton convention)
CAMERA_SIZE = (1280, 960)  # (width, height)
CAMERA_LIGHT_DIRECTION = (-0.57735026, 0.57735026, -0.57735026)
CAMERA_TABLE_COLOR = (0.55, 0.5, 0.42)

# --- Visual attributes of cables (what a VLM can see), independent of the simulator role ---
# 'neutral': colour drawn per cable from CABLE_PALETTE with a seeded RNG, uncorrelated with the role.
# 'uniform': every cable has the same colour (topology-only prompts).
# 'role':    today's role-coded colours (control mode; leaks the role to a VLM).
RENDER_MODES = ('neutral', 'uniform', 'role')
CABLE_PALETTE = {
    'red': (0.85, 0.10, 0.10),
    'blue': (0.15, 0.25, 0.85),
    'green': (0.10, 0.65, 0.15),
    'black': (0.08, 0.08, 0.08),
    'white': (0.95, 0.95, 0.95),
}
UNIFORM_COLOR_NAME = 'white'

# Simulator-internal role per cable label (never shown to a model; used for ground truth only).
ROLE_BY_LABEL = {'cable_A': 'knot', 'cable_B': 'linked', 'cable_C': 'linked', 'cable_free': 'free'}
ROLE_DEFAULT = 'other'
ROLE_COLOR_NAMES = {'knot': 'red', 'linked': 'orange', 'free': 'green', 'other': 'blue'}
