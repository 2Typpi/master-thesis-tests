# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sanity checks for every robot spec in ``panda_cable_pile.robots``.

Run (headless)::

    /home/mwuerstle@corp.exxcellent.de/miniconda3/envs/mujoco/bin/python check_robots.py [--robot so101 panda]

For each robot: the model loads, the mirror/finger links exist and carry collision shapes (so the hand mirrors
are not empty), and the scripted waypoints of both pulls (pregrasp, grasp, end of pull) are reachable by IK.
"""

import argparse
import sys

import numpy as np
import warp as wp

import newton
import newton.viewer

from panda_cable_pile import Panda
from panda_cable_pile.robots import ROBOTS

# Expected number of collide-flagged shapes on the mirrored bodies (a loader change that drops them would
# leave the hand invisible to the cables). Panda counts are not asserted, only that they are non-zero.
EXPECTED_SHAPES = {'so101': {'/gripper': 12, '/moving_jaw_so101_v1': 8}}
IK_TOL = 5e-3  # [m] at pregrasp / grasp / end of pull


def collide_shape_counts(sim, suffixes):
    b = sim._panda_builder
    counts = {}
    for suffix in suffixes:
        body = next(i for i, l in enumerate(b.body_label) if l.endswith(suffix))
        counts[suffix] = sum(1 for s in range(b.shape_count) if b.shape_body[s] == body and b.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES))
    return counts


def check(name, seed=0):
    ok = True
    sim = Panda(newton.viewer.ViewerNull(num_frames=1), argparse.Namespace(seed=seed, num_cables=6, robot=name))
    r = sim.robot
    print(f'[{name}] {sim.model.joint_dof_count} DOF, {sim.model.body_count} bodies')

    counts = collide_shape_counts(sim, r.mirror_links)
    for suffix, n in counts.items():
        want = EXPECTED_SHAPES.get(name, {}).get(suffix)
        good = n > 0 and (want is None or n == want)
        ok &= good
        print(f'    mirror {suffix:24s} {n:3d} collide shapes' + ('' if want is None else f' (expected {want})') + ('' if good else '   <-- FAIL'))
    missing = set(r.finger_links) - set(r.mirror_links)
    if missing:
        ok = False
        print(f'    finger_links not in mirror_links: {missing}   <-- FAIL')

    base = np.array(r.scene.base_pos)
    for pull in ('knot', 'free'):
        sim.pull = pull
        sim.reset(seed)
        sim._plan_grasp()
        T = sim.target
        pt, d, yaw = T['point'], T['dir'], T['yaw']
        waypoints = {
            'pregrasp': pt + [0, 0, sim.PREGRASP_HEIGHT - sim.GRASP_HEIGHT],
            'grasp': pt,
            'end of pull': pt + d * sim.PULL_LENGTH + [0, 0, 0.01],
        }
        for label, p in waypoints.items():
            sim.joint_q_ik.assign(np.array([r.home_q], dtype=np.float32))
            for _ in range(15):
                sim.solve_ik(p, yaw, iters=24)
            sim.set_arm(r.grip_open)
            err = float(np.linalg.norm(sim.tcp_position() - p))
            good = err < IK_TOL
            ok &= good
            print(f'    pull={pull:5s} {label:11s} r={np.hypot(*(p[:2] - base[:2])):.3f} m  IK err {err * 1000:6.2f} mm' + ('' if good else '   <-- FAIL'))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', nargs='+', default=sorted(ROBOTS), choices=sorted(ROBOTS))
    args = ap.parse_args()
    results = {name: check(name) for name in args.robot}
    if all(results.values()):
        print('ALL PASSED')
        return 0
    print('FAILED: ' + ', '.join(n for n, ok in results.items() if not ok))
    return 1


if __name__ == '__main__':
    sys.exit(main())
