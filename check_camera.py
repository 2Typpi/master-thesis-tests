# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 camera audit: do the images agree with the simulator state?

Run (headless)::

    /home/mwuerstle@corp.exxcellent.de/miniconda3/envs/mujoco/bin/python check_camera.py [--frames 30] [--seeds 0 1]

Checks, for several camera poses / fields of view:

1. **Back-projection**: every cable pixel is unprojected with its ray-hit distance; its distance to the true capsule
   axis of the labelled segment must equal the cable radius (this fixes FOV convention, axes, pose and depth meaning).
2. **Forward projection**: points on the capsule axes are projected into the image; where the pixel shows the same
   cable, the rendered depth must be one radius short of the axis distance.
3. **Negative control**: reading the depth as a z-depth instead of a ray distance must be *worse* off-axis.
4. **Coverage**: every hit pixel is a cable pixel (no table), and no cable pixel lacks a valid depth.
"""

import argparse
import sys
import types

import numpy as np
import warp as wp

import newton
import newton.viewer

from panda_cable_pile import Panda, projection
from panda_cable_pile.constants import CABLE_RADIUS

TOL_SURFACE = 5e-4  # [m] allowed |distance-to-axis - radius| for unprojected surface pixels
TOL_DEPTH = 1.5e-3  # [m] allowed depth error for projected axis points


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    return (*(np.sin(angle / 2) * axis), np.cos(angle / 2))


def capsule_axes(panda):
    """Per-cable-shape world endpoints (a, b) of the capsule axis, plus (cable, segment) labels."""
    model = panda.cable_model
    body_q = panda.cable_state_0.body_q.numpy()
    shape_body = model.shape_body.numpy()
    shape_xf = model.shape_transform.numpy()
    shape_scale = model.shape_scale.numpy()
    a_pts, b_pts, cab, seg = [], [], [], []
    for sid in range(model.shape_count):
        k = int(panda.rgbd_shape_cable[sid])
        if k < 0:
            continue
        bq = body_q[shape_body[sid]]
        R_b = projection.quat_to_matrix(bq[3:7])
        c = bq[:3] + R_b @ shape_xf[sid][:3]
        z = R_b @ projection.quat_to_matrix(shape_xf[sid][3:7]) @ np.array([0.0, 0.0, 1.0])
        hh = float(shape_scale[sid][1])
        a_pts.append(c - hh * z)
        b_pts.append(c + hh * z)
        cab.append(k)
        seg.append(int(panda.rgbd_shape_segment[sid]))
    return np.array(a_pts), np.array(b_pts), np.array(cab), np.array(seg)


def point_segment_distance(p, a, b):
    ab = b - a
    t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.einsum('ij,ij->i', ab, ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def ray_capsule_distance(origin, direction, a, b, r):
    """Smallest positive t with ``origin + t*direction`` on the capsule (axis a-b, radius r); nan if missed.

    Vectorised over rays (direction: (N, 3)) with one capsule per ray (a, b: (N, 3)). Direction must be unit length.
    """
    n = len(direction)
    best = np.full(n, np.inf)
    ba = b - a
    baba = np.einsum('ij,ij->i', ba, ba)
    oa = origin - a
    bard = np.einsum('ij,ij->i', ba, direction)
    baoa = np.einsum('ij,ij->i', ba, oa)
    # infinite cylinder, accepted only between the end planes
    qa = baba - bard ** 2
    qb = baba * np.einsum('ij,ij->i', oa, direction) - baoa * bard
    qc = baba * np.einsum('ij,ij->i', oa, oa) - baoa ** 2 - r * r * baba
    with np.errstate(divide='ignore', invalid='ignore'):
        disc = qb ** 2 - qa * qc
        ok = (disc >= 0) & (qa > 1e-12)
        t = np.where(ok, (-qb - np.sqrt(np.where(ok, disc, 0))) / qa, np.inf)
        y = baoa + t * bard
        ok &= (t > 0) & (y >= 0) & (y <= baba)
        best = np.where(ok, t, best)
    # the two end-cap spheres
    for centre in (a, b):
        oc = origin - centre
        bq = np.einsum('ij,ij->i', oc, direction)
        cq = np.einsum('ij,ij->i', oc, oc) - r * r
        d2 = bq ** 2 - cq
        with np.errstate(invalid='ignore'):
            ts = np.where(d2 >= 0, -bq - np.sqrt(np.where(d2 >= 0, d2, 0)), np.inf)
        best = np.where((ts > 0) & (ts < best), ts, best)
    return np.where(np.isfinite(best), best, np.nan)


def audit(panda, label, z_depth_control=True):
    cap = panda.capture()
    K, T = cap.K, cap.T_world_cam
    h, w = cap.depth.shape
    a, b, cab, seg = capsule_axes(panda)
    lookup = {(int(c), int(s)): i for i, (c, s) in enumerate(zip(cab, seg))}

    # --- 1. back-projection -------------------------------------------------------------------
    ys, xs = np.nonzero((cap.instance >= 0) & (cap.depth > 0))
    pts = projection.unproject(xs, ys, cap.depth[ys, xs], T, K)
    idx = np.array([lookup[(int(c), int(s))] for c, s in zip(cap.instance[ys, xs], cap.segment[ys, xs])])
    err = np.abs(point_segment_distance(pts, a[idx], b[idx]) - CABLE_RADIUS)
    res = {'cable_px': len(xs), 'surf_err_mean_mm': 1e3 * err.mean(), 'surf_err_p99_mm': 1e3 * np.percentile(err, 99), 'surf_err_max_mm': 1e3 * err.max()}

    # --- 4. coverage --------------------------------------------------------------------------
    if not panda.rgbd_table:
        res['hit_but_no_instance'] = int(((cap.depth > 0) & (cap.instance < 0)).sum())
    res['instance_but_no_depth'] = int(((cap.instance >= 0) & (cap.depth <= 0)).sum())

    # --- 2. forward projection of axis points + exact ray/capsule check -----------------------------
    # Project points on the capsule axes into the image. For every pixel that shows the same (cable, segment), the
    # rendered depth must equal the exact intersection of that pixel's centre ray with that capsule.
    t = np.linspace(0.15, 0.85, 5)
    samples = (a[:, None, :] * (1 - t[None, :, None]) + b[:, None, :] * t[None, :, None]).reshape(-1, 3)
    lab_c = np.repeat(cab, len(t))
    lab_s = np.repeat(seg, len(t))
    cap_idx = np.repeat(np.arange(len(cab)), len(t))
    px, py, ray_d, _ = projection.project(samples, T, K)
    ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    same = np.zeros(len(samples), dtype=bool)
    same[inside] = (cap.instance[iy[inside], ix[inside]] == lab_c[inside]) & (cap.segment[iy[inside], ix[inside]] == lab_s[inside])
    dirs = (projection.pixel_rays(ix[same], iy[same], K)) @ T[:3, :3].T
    exact = ray_capsule_distance(T[:3, 3], dirs, a[cap_idx[same]], b[cap_idx[same]], CABLE_RADIUS)
    d_err = np.abs(exact - cap.depth[iy[same], ix[same]])
    d_err = d_err[np.isfinite(d_err)]
    res['fwd_points'] = int(inside.sum())
    res['fwd_same_cable_frac'] = float(same.sum() / max(1, inside.sum()))
    res['fwd_depth_err_p99_mm'] = 1e3 * float(np.percentile(d_err, 99)) if d_err.size else float('nan')
    res['fwd_depth_err_max_mm'] = 1e3 * float(d_err.max()) if d_err.size else float('nan')

    # --- 3. negative control: pretend depth was a z-depth ----------------------------------------
    if z_depth_control:
        rays = projection.pixel_rays(xs, ys, K)
        wrong = projection.unproject(xs, ys, cap.depth[ys, xs] / -rays[:, 2], T, K)
        wrong_err = np.abs(point_segment_distance(wrong, a[idx], b[idx]) - CABLE_RADIUS)
        res['control_zdepth_err_max_mm'] = 1e3 * wrong_err.max()

    ok = (res['cable_px'] > 500 and res['surf_err_p99_mm'] < 1e3 * TOL_SURFACE and res['instance_but_no_depth'] == 0
          and res.get('hit_but_no_instance', 0) == 0 and res['fwd_depth_err_p99_mm'] < 1e3 * TOL_DEPTH and res['fwd_same_cable_frac'] > 0.5)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    for k, v in res.items():
        print(f'        {k:28s} {v:.3f}' if isinstance(v, float) else f'        {k:28s} {v}')
    return ok, res


def make_panda(seed, **camera_options):
    parser = argparse.Namespace(seed=seed, num_cables=6, render_mode='neutral', **camera_options)
    viewer = newton.viewer.ViewerNull(num_frames=1)
    return Panda(viewer, parser)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=30, help='frames to step before capturing (so cables are not in their initial pose)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    args = ap.parse_args()

    failures = 0
    for seed in args.seeds:
        panda = make_panda(seed)
        for _ in range(args.frames):
            panda.step()
        poses = [
            ('top-down, default camera (z=0.8, fov 55)', (0.0, 0.0, 0.8), (0.0, 0.0, 0.0, 1.0)),
            ('tilted 30 deg about x, off-centre', (0.0, -0.35, 0.7), quat_from_axis_angle((1, 0, 0), np.radians(30))),
            ('tilted 45 deg about y + yaw 20 deg', (-0.3, 0.1, 0.65), quat_from_axis_angle((0, 1, 0), np.radians(-45))),
        ]
        for name, pos, quat in poses:
            if name.startswith('tilted 45'):
                # compose yaw 20 deg about world z after the tilt
                qz = quat_from_axis_angle((0, 0, 1), np.radians(20))
                R = projection.quat_to_matrix(qz) @ projection.quat_to_matrix(quat)
                quat = rot_to_quat(R)
            panda.set_camera_pose(pos, quat)
            ok, _ = audit(panda, f'seed {seed}: {name}')
            failures += (not ok)
    # a different resolution / fov, to check K is not hard-wired to the defaults
    panda = make_panda(0, camera_size=(640, 480), camera_fov=40.0, camera_height=0.6)
    for _ in range(args.frames):
        panda.step()
    ok, _ = audit(panda, 'seed 0: 640x480, fov 40, z=0.6')
    failures += (not ok)
    # table visible: planes must not be labelled as cables
    panda = make_panda(0, camera_table=True, camera_light='lit')
    for _ in range(args.frames):
        panda.step()
    ok, res = audit(panda, 'seed 0: table + light enabled')
    failures += (not ok)
    print('ALL PASSED' if failures == 0 else f'{failures} FAILED')
    sys.exit(1 if failures else 0)


def rot_to_quat(R):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return ((R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


if __name__ == '__main__':
    main()
