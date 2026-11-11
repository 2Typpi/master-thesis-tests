# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Franka Panda pulling a cable from a pile in Newton.

Run headless, for example::
    python example_robot_panda_cable_pile.py --viewer null --num-frames 900 --pull knot

Implementation is split into the :mod:`panda_cable_pile` package by responsibility.
"""

import newton.examples

from panda_cable_pile import Panda


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=900)
    parser.add_argument("--pull", choices=["knot", "free"], default="knot")
    parser.add_argument("--grasp", choices=["friction", "sticky"], default="friction")
    parser.add_argument("--num-cables", type=int, default=6, help="total, including A, B, C and the free cable")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-time", type=float, default=2.0, help="cap [s] on waiting for the pile to rest")
    parser.add_argument("--unlink-control", action="store_true", help="control run: B and C are not hooked into A")
    parser.add_argument("--filter-mirrors", action="store_true", help="control run: hand cannot touch cables")
    viewer, args = newton.examples.init(parser)
    example = Panda(viewer, args)
    newton.examples.run(example, args)
