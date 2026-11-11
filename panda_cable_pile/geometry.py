# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable centerline generation and geometry helpers."""

import numpy as np
import warp as wp
import newton

from .constants import CABLE_RADIUS, CABLE_LENGTH, CABLE_SEGMENTS

def _resample(points: np.ndarray, n_segments: int) -> np.ndarray:
    """Resample a polyline into ``n_segments`` equal-length segments."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, s[-1], n_segments + 1)
    return np.stack([np.interp(targets, s, points[:, k]) for k in range(3)], axis=1)

def _resample_by_length(points, seg_target):
    points = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(round(s[-1] / seg_target)))
    t = np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(t, s, points[:, k]) for k in range(3)], axis=1)

def eye_helix(S, t, c, R, rise, turns=1.0, n=400):
    """Helix of ~one turn, starting at S heading along t; ring centre at S + R*c; axis n = t x c.

    Returns points from S outward and the axis vector (rise runs along +axis)."""
    S, t, c = (np.asarray(v, float) for v in (S, t, c))
    axis = np.cross(t, c)
    centre = S + R * c
    phi = np.linspace(0.0, turns * 2 * np.pi, n)[:, None]
    pts = centre - R * c * np.cos(phi) + R * t * np.sin(phi) + axis * rise * phi / (2 * np.pi)
    return pts

def line(a, b, n=60):
    a, b = (np.asarray(a, float), np.asarray(b, float))
    return a + (b - a) * np.linspace(0, 1, n)[:, None]

def chain_points(x0, y0, r=CABLE_RADIUS, R=0.03, tail=0.2, body_ab=0.11, tail_c=0.12, seg=0.0125):
    """Three interlocked cables A-B-C along +y. A's tail tip is its LAST point (graspable, pulled -y).

    A: vertical eye (yz plane) at y0; B: horizontal eye hooked through A's, a straight body, then a
    vertical eye; C: horizontal eye hooked through B's second eye, body to +y.
    Heights step up by ~3r+ at every hook so the wires start out of contact.
    """
    rise = 3.0 * r
    zA, zB, zC = (r, 4.0 * r, 8.0 * r)
    SA = np.array([x0, y0, zA])
    eyeA = eye_helix(SA, t=[0, 1, 0], c=[0, 0, 1], R=R, rise=rise)
    A = np.vstack([eyeA[::-1], line(SA, SA + [0, -tail, 0])])
    SB1 = np.array([x0 + R, y0 + R, zB])
    eyeB1 = eye_helix(SB1, t=[0, -1, 0], c=[-1, 0, 0], R=R, rise=rise)
    if np.cross([0, -1, 0], [-1, 0, 0])[2] < 0:
        eyeB1 = eye_helix(SB1, t=[0, -1, 0], c=[-1, 0, 0], R=R, rise=-rise)
    ys2 = y0 + R + body_ab
    SB2 = np.array([x0 + R, ys2, zB])
    eyeB2 = eye_helix(SB2, t=[0, 1, 0], c=[0, 0, 1], R=R, rise=rise)
    B = np.vstack([eyeB1[::-1], line(SB1, SB2), eyeB2])
    SC = np.array([x0 + 2 * R, ys2 + R, zC])
    eyeC = eye_helix(SC, t=[0, -1, 0], c=[-1, 0, 0], R=R, rise=rise if np.cross([0, -1, 0], [-1, 0, 0])[2] > 0 else -rise)
    C = np.vstack([eyeC[::-1], line(SC, SC + [0, tail_c, 0])])
    out = {'A': _resample_by_length(A, seg), 'B': _resample_by_length(B, seg), 'C': _resample_by_length(C, seg)}
    eye_len = 2 * np.pi * R
    n_eye = int(round(eye_len / seg))
    return (out, n_eye)

def free_cable_points(center: np.ndarray, length: float=CABLE_LENGTH) -> np.ndarray:
    """Straight cable along y with its +y end exposed (about 0.2 m past the centre)."""
    ys = np.linspace(center[1] + 0.2 - length, center[1] + 0.2, CABLE_SEGMENTS + 1)
    pts = np.stack([np.full_like(ys, center[0]), ys, np.full_like(ys, CABLE_RADIUS + 0.001)], axis=1)
    return pts

def random_cable_points(rng: np.random.Generator, center: np.ndarray, z0: float) -> np.ndarray:
    """Gently curved cable dropped from ``z0`` with a random pose."""
    start = np.array([center[0], center[1]]) + rng.uniform(-0.07, 0.07, 2)
    heading = rng.uniform(0, 2 * np.pi)
    step = CABLE_LENGTH / CABLE_SEGMENTS
    pts = [np.array([start[0], start[1], z0])]
    for _ in range(CABLE_SEGMENTS):
        heading += rng.normal(0.0, 0.12)
        pts.append(pts[-1] + step * np.array([np.cos(heading), np.sin(heading), 0.0]))
    pts = np.array(pts)
    pts[:, :2] -= pts[:, :2].mean(axis=0) - np.array(center)
    return pts

def _zero_mass(builder: newton.ModelBuilder, body: int) -> None:
    builder.body_mass[body] = 0.0
    builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33(0.0)
    builder.body_inv_inertia[body] = wp.mat33(0.0)
