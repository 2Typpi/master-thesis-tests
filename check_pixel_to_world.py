# SPDX-License-Identifier: Apache-2.0
"""Single-frame sanity check of pixel + depth -> 3D grasp point against simulator ground truth.

Run: /home/mwuerstle@corp.exxcellent.de/miniconda3/envs/mujoco/bin/python check_pixel_to_world.py [--robot NAME] [--seed N]
"""
import argparse
import sys

import numpy as np

from check_camera import make_panda, quat_from_axis_angle
from panda_cable_pile.constants import CABLE_RADIUS

TOL = 2e-3  # [m] top-down: error left is mostly pixel quantisation on a 4 mm cable
TOL_OBLIQUE = 5e-3  # [m] tilted view: the ray hits the cable's side, up to one radius off the axis; a z-only drop cannot fix that


def check(panda, label, tol=TOL):
    cap = panda.capture()
    pm = panda.pixel_map()
    body_q = panda.cable_state_0.body_q.numpy()
    errs, xy_errs = [], []
    occluded = 0
    for cable in panda.cables:
        for i in range(2, len(cable['bodies']) - 2):
            axis_pt = body_q[cable['bodies'][i], :3].astype(float)
            u, v = panda.world_to_pixel(axis_pt)
            if not (1 <= u < pm.width - 1 and 1 <= v < pm.height - 1):
                continue
            # Skip points that another cable (or a far part of the same one) lies on top of: the camera cannot see them.
            k = panda.cables.index(cable)
            inst, seg = cap.instance[v - 1:v + 2, u - 1:u + 2], cap.segment[v - 1:v + 2, u - 1:u + 2]
            if not ((inst == k).all() and (np.abs(seg - i) <= 1).all()):
                occluded += 1
                continue
            # Compare against the capsule axis: with drop = radius the output must land on it (up to quantisation).
            p = pm.grasp_point(cap.depth, u, v, drop=CABLE_RADIUS)
            errs.append(np.linalg.norm(p - axis_pt))
            xy_errs.append(np.linalg.norm(p[:2] - axis_pt[:2]))
    errs = np.array(errs)
    ok = len(errs) > 20 and np.percentile(errs, 90) < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: n={len(errs)} (skipped {occluded} occluded) err p50={1e3*np.median(errs):.2f} mm "
          f"p90={1e3*np.percentile(errs, 90):.2f} mm max={1e3*errs.max():.2f} mm (xy p90 {1e3*np.percentile(xy_errs, 90):.2f} mm)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frames', type=int, default=30)
    args = ap.parse_args()
    panda = make_panda(args.seed, robot=args.robot)
    for _ in range(args.frames):
        panda.step()
    ok = check(panda, 'top-down camera')
    panda.set_camera_pose((0.0, -0.35, 0.7), quat_from_axis_angle((1, 0, 0), np.radians(30)))
    ok &= check(panda, 'tilted camera (cache rebuilt after set_camera_pose)', TOL_OBLIQUE)
    # edge handling: no depth -> explicit error; out of image -> explicit error
    pm = panda.pixel_map()
    for args_, name in (((0, 0), 'corner without cable'), ((-5, 3), 'outside image')):
        try:
            pm.grasp_point(panda.capture().depth, *args_)
            if name == 'outside image':
                ok = False
                print('FAIL: out-of-image pixel did not raise')
        except ValueError as e:
            print(f'[PASS] {name}: ValueError: {e}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
