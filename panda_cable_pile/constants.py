# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared scene and simulation parameters."""

import numpy as np
import warp as wp

HOME_Q = [0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.785, 0.04, 0.04]

PANDA_BASE = (-0.45, 0.0, 0.0)

PILE_CENTER = np.array([0.0, 0.0])

HAND_COLLISION_LINKS = ['/fr3_link7', '/fr3_leftfinger', '/fr3_rightfinger']

TCP_OFFSET = (0.0, 0.0, 0.21)

FINGER_OPEN = 0.04

FRAME_DT = 1.0 / 60.0

SIM_SUBSTEPS = 10

SIM_ITERATIONS = 10

SIM_DT = FRAME_DT / SIM_SUBSTEPS

CABLE_RADIUS = 0.004

CABLE_LENGTH = 0.5

CABLE_SEGMENTS = 40

KNOT_SEGMENTS = 60

YOUNGS_MODULUS = 100000000.0

CABLE_MODULI = {'cable_A': 100000000.0, 'cable_B': 60000000.0, 'cable_C': 150000000.0, 'cable_free': 80000000.0}

BEND_SCALE = {'cable_A': 0.25, 'cable_B': 0.8, 'cable_C': 0.8, 'cable_free': 0.1}

CABLE_MU = 5.0

CABLE_KE = 1000000.0

CABLE_KD = 0.1

FRICTION_EPS = 0.001

STRETCH_DAMPING = 0.0001

BEND_DAMPING = 0.05

CONTACT_GAP = 0.004

HAND_MU = 3.0

HAND_KE = 10000.0

COLOR_KNOT = (0.9, 0.15, 0.1)

COLOR_LINKED = (0.95, 0.6, 0.1)

COLOR_FREE = (0.1, 0.7, 0.2)

COLOR_OTHER = (0.35, 0.5, 0.85)
