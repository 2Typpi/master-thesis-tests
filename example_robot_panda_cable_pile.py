# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Franka Panda next to a pile of cables (Newton).

Three cables (A, B, C) are hooked into each other like chain links and none is
pinned, so pulling A's tail drags B and C out of the pile. Another cable lies
partly under the pile and slides out on its own when pulled. The arm is scripted (IK state machine) and
grasps the cable tail with a friction pinch (``--grasp friction``) or, for
debugging only, a rigid attachment (``--grasp sticky``).

Architecture (see plans/plan_cable_pile.md): two unconnected models.
``self.model`` is the Panda (no solver, ``joint_q`` + ``eval_fk``).
``self.cable_model`` is stepped by ``SolverVBD`` and contains kinematic
mirrors of the hand and fingers. Coupling is one-way: the arm deflects the
cables, the cables never push back.

Run headless:
    python example_robot_panda_cable_pile.py --viewer null --num-frames 900 --pull knot
"""

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

# ---------------------------------------------------------------- constants
HOME_Q = [0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.785, 0.04, 0.04]
PANDA_BASE = (-0.45, 0.0, 0.0)
PILE_CENTER = np.array([0.0, 0.0])
HAND_COLLISION_LINKS = ["/fr3_link7", "/fr3_leftfinger", "/fr3_rightfinger"]
# Tool centre point (between the finger tips), in the frame of fr3_link7.
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
YOUNGS_MODULUS = 1.0e8
CABLE_MODULI = {
    "cable_A": 1.0e8,
    "cable_B": 6.0e7,
    "cable_C": 1.5e8,
    "cable_free": 8.0e7,
}
BEND_SCALE = {
    "cable_A": 0.25,
    "cable_B": 0.8,
    "cable_C": 0.8,
    "cable_free": 0.1,
}
CABLE_MU = 5.0
CABLE_KE = 1.0e6  # AVBD contact penalty is capped at ke; 1e2 pinned it at k_start and never stiffened
CABLE_KD = 0.1
FRICTION_EPS = 1.0e-3
STRETCH_DAMPING = 1.0e-4
BEND_DAMPING = 5.0e-2
CONTACT_GAP = 0.004
HAND_MU = 3.0  # rubber-like finger pads
HAND_KE = 1.0e4  # stiff pads: pinch force = ke * penetration

COLOR_KNOT = (0.9, 0.15, 0.1)  # cable A (the one the arm pulls)
COLOR_LINKED = (0.95, 0.6, 0.1)  # cables B, C hooked to A
COLOR_FREE = (0.1, 0.7, 0.2)
COLOR_OTHER = (0.35, 0.5, 0.85)


def quat_to_vec4(q) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


# ------------------------------------------------------------------ kernels
@wp.kernel
def _snapshot_hand_poses(
    hand_body_q: wp.array[wp.transform],
    hand_bodies: wp.array[int],
    prev_q: wp.array[wp.transform],
):
    tid = wp.tid()
    prev_q[tid] = hand_body_q[hand_bodies[tid]]


@wp.kernel
def _update_hand_mirrors(
    hand_body_q: wp.array[wp.transform],
    hand_bodies: wp.array[int],
    prev_q: wp.array[wp.transform],
    alpha: float,
    cable_body_q: wp.array[wp.transform],
    mirror_bodies: wp.array[int],
):
    # Sweep the kinematic hand copies from last frame's pose to the new one.
    tid = wp.tid()
    target = hand_body_q[hand_bodies[tid]]
    prev = prev_q[tid]
    pos = wp.lerp(wp.transform_get_translation(prev), wp.transform_get_translation(target), alpha)
    rot = wp.quat_slerp(wp.transform_get_rotation(prev), wp.transform_get_rotation(target), alpha)
    cable_body_q[mirror_bodies[tid]] = wp.transform(pos, rot)


@wp.kernel
def _drive_attached(
    hand_body_q: wp.array[wp.transform],
    hand_body: int,
    local_xform: wp.transform,
    cable_body_q: wp.array[wp.transform],
    cable_body_qd: wp.array[wp.spatial_vector],
    seg_body: int,
):
    # --grasp sticky: rigidly carry one segment with the hand.
    cable_body_q[seg_body] = wp.transform_multiply(hand_body_q[hand_body], local_xform)
    cable_body_qd[seg_body] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _capsule_xforms(
    body_q: wp.array[wp.transform],
    first_body: int,
    half_height: float,
    out: wp.array[wp.transform],
):
    # Body origin is the segment start; the capsule is centred half_height along +Z.
    tid = wp.tid()
    out[tid] = wp.transform_multiply(
        body_q[first_body + tid], wp.transform(wp.vec3(0.0, 0.0, half_height), wp.quat_identity())
    )


# --------------------------------------------------------- cable geometry
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
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + (b - a) * np.linspace(0, 1, n)[:, None]

def chain_points(x0, y0, r=CABLE_RADIUS, R=0.03, tail=0.20, body_ab=0.11, tail_c=0.12, seg=0.0125):
    """Three interlocked cables A-B-C along +y. A's tail tip is its LAST point (graspable, pulled -y).

    A: vertical eye (yz plane) at y0; B: horizontal eye hooked through A's, a straight body, then a
    vertical eye; C: horizontal eye hooked through B's second eye, body to +y.
    Heights step up by ~3r+ at every hook so the wires start out of contact.
    """
    rise = 3.0 * r
    zA, zB, zC = r, 4.0 * r, 8.0 * r
    # --- A: reversed(eye entered from the tail) + straight tail going -y
    SA = np.array([x0, y0, zA])
    eyeA = eye_helix(SA, t=[0, 1, 0], c=[0, 0, 1], R=R, rise=rise)
    A = np.vstack([eyeA[::-1], line(SA, SA + [0, -tail, 0])])
    # --- B: horizontal eye hooked through A's eye (centre one radius ahead in y)
    SB1 = np.array([x0 + R, y0 + R, zB])
    eyeB1 = eye_helix(SB1, t=[0, -1, 0], c=[-1, 0, 0], R=R, rise=rise)   # exits east, goes south first
    # rise must lift (+z): axis = t x c
    if np.cross([0, -1, 0], [-1, 0, 0])[2] < 0:
        eyeB1 = eye_helix(SB1, t=[0, -1, 0], c=[-1, 0, 0], R=R, rise=-rise)
    ys2 = y0 + R + body_ab
    SB2 = np.array([x0 + R, ys2, zB])
    eyeB2 = eye_helix(SB2, t=[0, 1, 0], c=[0, 0, 1], R=R, rise=rise)
    B = np.vstack([eyeB1[::-1], line(SB1, SB2), eyeB2])
    # --- C: horizontal eye hooked through B's second eye, body to +y
    SC = np.array([x0 + 2 * R, ys2 + R, zC])
    eyeC = eye_helix(SC, t=[0, -1, 0], c=[-1, 0, 0], R=R, rise=rise if np.cross([0,-1,0],[-1,0,0])[2] > 0 else -rise)
    C = np.vstack([eyeC[::-1], line(SC, SC + [0, tail_c, 0])])
    out = {"A": _resample_by_length(A, seg), "B": _resample_by_length(B, seg), "C": _resample_by_length(C, seg)}
    # number of bodies in each eye (for linked-ness checks)
    eye_len = 2 * np.pi * R
    n_eye = int(round(eye_len / seg))
    return out, n_eye



def free_cable_points(center: np.ndarray, length: float = CABLE_LENGTH) -> np.ndarray:
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


class Example:
    """Panda + cable pile. Scene build / reset / physics are separate from the controller."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args
        self.pull = getattr(args, "pull", "knot")
        self.grasp = getattr(args, "grasp", "friction")
        self.num_cables = max(4, getattr(args, "num_cables", 6))
        self.seed = getattr(args, "seed", 0)
        self.settle_time = getattr(args, "settle_time", 2.0)
        self.filter_mirrors = getattr(args, "filter_mirrors", False)
        self.unlink_control = getattr(args, "unlink_control", False)
        self.device = viewer.device if hasattr(viewer, "device") else wp.get_device()

        self._build_panda()
        self._setup_ik()
        self.reset(self.seed)

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        self.viewer.set_camera(wp.vec3(0.75, -0.85, 0.55), pitch=-28.0, yaw=135.0)

    # ------------------------------------------------------------- panda
    def _build_panda(self):
        builder = newton.ModelBuilder()
        asset = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(PANDA_BASE, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )
        builder.joint_q[: len(HOME_Q)] = HOME_Q
        builder.add_ground_plane()
        self._panda_builder = builder
        self.hand_body = next(i for i, l in enumerate(builder.body_label) if l.endswith("/fr3_link7"))
        self.model = builder.finalize()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def _setup_ik(self):
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.hand_body,
            link_offset=wp.vec3(*TCP_OFFSET),
            target_positions=wp.array([wp.vec3(0.0, 0.0, 0.5)], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.hand_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(1.0, 0.0, 0.0, 0.0)], dtype=wp.vec4),
        )
        self.limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
        )
        self.joint_q_ik = wp.array(np.array([HOME_Q], dtype=np.float32), dtype=wp.float32)
        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    @staticmethod
    def hand_rotation(yaw_a: float) -> wp.quat:
        """Gripper pointing down; ``yaw_a`` is the rotation about the tool z."""
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi)
        return wp.mul(q, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw_a))

    @staticmethod
    def yaw_for_cable_direction(direction_xy) -> float:
        """Hand angle that makes the fingers open perpendicular to a cable running along ``direction_xy``.

        Measured: with this hand_rotation the finger axis points at 3*pi/4 - yaw in the xy plane.
        """
        phi = float(np.arctan2(direction_xy[1], direction_xy[0]))
        return np.pi / 4 - phi

    def solve_ik(self, tcp: np.ndarray, yaw_a: float, iters: int | None = None):
        self.pos_obj.set_target_positions(wp.array([wp.vec3(*tcp)], dtype=wp.vec3))
        self.rot_obj.set_target_rotations(wp.array([quat_to_vec4(self.hand_rotation(yaw_a))], dtype=wp.vec4))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=iters or self.ik_iters)

    def tcp_position(self) -> np.ndarray:
        xf = wp.transform(*self.state.body_q.numpy()[self.hand_body])
        return np.array(wp.transform_point(xf, wp.vec3(*TCP_OFFSET)))

    def set_arm(self, finger_width: float):
        """Write IK solution + finger width into the Panda model and update FK."""
        q = self.joint_q_ik.numpy()[0].copy()
        q[7] = q[8] = finger_width
        self.model.joint_q.assign(q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    # ------------------------------------------------------ scene / reset
    def reset(self, seed: int | None = None):
        """Rebuild the pile for ``seed`` and restart the controller."""
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)
        self.joint_q_ik.assign(np.array([HOME_Q], dtype=np.float32))
        self.set_arm(FINGER_OPEN)
        self._build_cables(rng)
        self._init_controller()

    def _add_cable(self, builder, pts, label, youngs_modulus=YOUNGS_MODULUS):
        stretch, bend = newton.utils.create_cable_stiffness_from_elastic_moduli(youngs_modulus, CABLE_RADIUS)
        bend *= BEND_SCALE.get(label, 1.0)
        bodies, _ = builder.add_rod(
            positions=[wp.vec3(*p) for p in pts],
            radius=CABLE_RADIUS,
            bend_stiffness=bend,
            bend_damping=BEND_DAMPING,
            stretch_stiffness=stretch,
            stretch_damping=STRETCH_DAMPING,
            label=label,
        )
        seg = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        return bodies, seg

    def _build_cables(self, rng):
        center = PILE_CENTER
        chain, _ = chain_points(-0.03, 0.0)
        if self.unlink_control:  # control run: B and C are moved away so they are NOT hooked into A
            chain["B"] = chain["B"] + np.array([0.12, 0.0, 0.0])
            chain["C"] = chain["C"] + np.array([0.12, 0.0, 0.0])
        knot_pts = chain["A"]
        free_pts = free_cable_points(center + np.array([0.11, -0.03]))
        # Authored cables. A, B, C are hooked into each other and none is pinned, so pulling
        # A's tail drags B and C out of the pile.
        specs = [
            (chain["A"], "cable_A", COLOR_KNOT),
            (chain["B"], "cable_B", COLOR_LINKED),
            (chain["C"], "cable_C", COLOR_LINKED),
            (free_pts, "cable_free", COLOR_FREE),
        ]
        # Grasp points: just behind the exposed tip of each authored cable.
        self.grasp_info = {
            "knot": self._grasp_from_tail(knot_pts, -1),
            "free": self._grasp_from_tail(free_pts, -1),
        }
        keep_clear = [self.grasp_info[k]["point"][:2] for k in self.grasp_info]
        for k in range(max(0, self.num_cables - 4)):
            for _ in range(50):
                pts = random_cable_points(rng, center, z0=0.09 + 0.014 * k)
                if all(np.min(np.linalg.norm(pts[:, :2] - c, axis=1)) > 0.06 for c in keep_clear):
                    break
            specs.append((pts, f"cable_{k}", COLOR_OTHER))

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = CABLE_KE
        builder.default_shape_cfg.kd = CABLE_KD
        builder.default_shape_cfg.mu = CABLE_MU
        # Default gap is 0.1 m (25 cable radii): thousands of far-away candidate contacts.
        builder.rigid_gap = CONTACT_GAP

        self.cables = []  # dicts: first, n, seg, colour, name
        for pts, label, colour in specs:
            modulus = CABLE_MODULI.get(label, YOUNGS_MODULUS)
            bodies, seg = self._add_cable(builder, pts, label, modulus)
            self.cables.append({"name": label, "bodies": bodies, "seg": seg, "colour": colour, "pts0": pts})

        self._add_hand_mirrors(builder)
        builder.add_ground_plane()
        builder.color()

        self.cable_model = builder.finalize()
        self.cable_solver = newton.solvers.SolverVBD(
            self.cable_model, iterations=SIM_ITERATIONS, friction_epsilon=FRICTION_EPS
        )
        self.cable_state_0 = self.cable_model.state()
        self.cable_state_1 = self.cable_model.state()
        self.cable_control = self.cable_model.control()
        self.cable_contacts = self.cable_model.contacts()

        inv_mass = self.cable_model.body_inv_mass.numpy()
        self.dynamic_cable_bodies = [b for c in self.cables for b in c["bodies"] if inv_mass[b] > 0.0]

        self.capsule_buffers = []
        for c in self.cables:
            n = len(c["bodies"])
            self.capsule_buffers.append(
                (
                    wp.empty(n, dtype=wp.transform),
                    wp.full(n, wp.vec3(CABLE_RADIUS, CABLE_RADIUS, 0.5 * c["seg"]), dtype=wp.vec3),
                    wp.full(n, wp.vec3(*c["colour"]), dtype=wp.vec3),
                )
            )
        self.sim_time = 0.0
        self.frame = 0

    @staticmethod
    def _grasp_from_tail(pts: np.ndarray, end_index: int) -> dict:
        """Grasp ~3 segments behind the tip; pull direction continues along the tail."""
        i_tip = len(pts) - 1 if end_index == -1 else 0
        step = -1 if end_index == -1 else 1
        i_grasp = i_tip + 3 * step
        i_ref = i_tip + 6 * step
        point = pts[i_grasp].copy()
        direction = pts[i_grasp] - pts[i_ref]
        direction[2] = 0.0
        direction /= np.linalg.norm(direction)
        return {"point": point, "dir": direction, "index": i_grasp}

    def _add_hand_mirrors(self, builder):
        panda = self._panda_builder
        panda_q = self.state.body_q.numpy()
        mirror_shapes, hand_bodies, mirror_bodies = [], [], []
        hand_cfg = builder.default_shape_cfg.copy()
        hand_cfg.mu = HAND_MU
        hand_cfg.ke = HAND_KE
        for suffix in HAND_COLLISION_LINKS:
            pb = next(i for i, l in enumerate(panda.body_label) if l.endswith(suffix))
            mirror = builder.add_body(xform=wp.transform(*panda_q[pb]), is_kinematic=True, label=f"mirror_{suffix}")
            for sh in range(panda.shape_count):
                if panda.shape_body[sh] != pb or not panda.shape_flags[sh] & int(newton.ShapeFlags.COLLIDE_SHAPES):
                    continue
                st = int(panda.shape_type[sh])
                if st == int(newton.GeoType.MESH):
                    st = int(newton.GeoType.CONVEX_MESH)
                mirror_shapes.append(
                    builder.add_shape(
                        mirror,
                        st,
                        xform=panda.shape_transform[sh],
                        scale=panda.shape_scale[sh],
                        src=panda.shape_source[sh],
                        cfg=hand_cfg,
                    )
                )
            _zero_mass(builder, mirror)
            hand_bodies.append(pb)
            mirror_bodies.append(mirror)
        for i, a in enumerate(mirror_shapes):
            for b in mirror_shapes[i + 1 :]:
                builder.add_shape_collision_filter_pair(a, b)
        if self.filter_mirrors:  # control run for the contact check
            for c in self.cables:
                for body in c["bodies"]:
                    for cs in builder.body_shapes[body]:
                        for ms in mirror_shapes:
                            builder.add_shape_collision_filter_pair(cs, ms)
        self.hand_bodies = wp.array(hand_bodies, dtype=int)
        self.mirror_bodies = wp.array(mirror_bodies, dtype=int)
        self.hand_prev_q = wp.array(panda_q[hand_bodies], dtype=wp.transform)
        self.mirror_shapes = mirror_shapes

    # --------------------------------------------------------- controller
    # (name, duration [s]); the scripted controller can be replaced by a policy
    # that calls ``apply_command`` + ``advance_physics`` itself.
    PHASES = [
        ("settle", None),
        ("pregrasp", 1.5),
        ("descend", 1.0),
        ("close", 0.8),
        ("hold", 0.3),
        ("pull", 2.5),
        ("lift", 1.0),
        ("release", 0.5),
        ("retreat", 0.6),
    ]
    MIN_SETTLE = 0.5
    REST_LIN = 0.005  # m/s
    REST_ANG = 0.05  # rad/s
    REST_HOLD = 0.3  # s the pile must stay below both thresholds
    PREGRASP_HEIGHT = 0.12
    GRASP_HEIGHT = 0.012
    PULL_LENGTH = 0.25
    FINGER_CLOSED = 0.002  # per-finger target; the cable (r=4 mm) stops the fingers earlier
    LIFT_HEIGHT = 0.15

    def _init_controller(self):
        self.phase_idx = 0
        self.phase_time = 0.0
        self.settle_duration = self.settle_time
        self.rest_time = 0.0
        self.settled_at = None
        self.target = None
        self.attached = None
        self.ik_residuals = {}
        self.cmd = (self.tcp_position(), self.yaw_for_cable_direction((1.0, 0.0)), FINGER_OPEN)
        self.metrics_before = None
        self.metrics_after = None
        self.links_after = None
        self.links_before = None

    def max_cable_speed(self) -> tuple[float, float]:
        """(max linear [m/s], max angular [rad/s]) body speed over all non-kinematic cable bodies."""
        qd = self.cable_state_0.body_qd.numpy()
        idx = self.dynamic_cable_bodies
        return float(np.abs(qd[idx, :3]).max()), float(np.abs(qd[idx, 3:]).max())

    def cable_by_name(self, name):
        return next(c for c in self.cables if c["name"] == name)

    def cable_starts(self, cable) -> np.ndarray:
        q = self.cable_state_0.body_q.numpy()
        return q[cable["bodies"], :3]

    def _plan_grasp(self):
        cable = self.cable_by_name("cable_A" if self.pull == "knot" else "cable_free")
        starts = self.cable_starts(cable)
        n = len(starts)
        point = starts[n - 4].copy()
        direction = starts[n - 4] - starts[n - 7]
        direction[2] = 0.0
        direction /= np.linalg.norm(direction)
        point[2] = self.GRASP_HEIGHT
        self.target = {"cable": cable, "point": point, "dir": direction, "seg_index": n - 4}
        self.target["yaw"] = self.yaw_for_cable_direction(direction)
        self.metrics_before = self.pile_metrics()
        self.links_before = self.link_distances()

    def controller_command(self):
        """Scripted state machine -> (tcp [3], hand yaw, finger width)."""
        name, duration = self.PHASES[self.phase_idx]
        if name == "settle":
            return self.cmd
        T = self.target
        u = min(self.phase_time / duration, 1.0)
        s = u * u * (3.0 - 2.0 * u)  # smooth step
        above = T["point"] + np.array([0.0, 0.0, self.PREGRASP_HEIGHT - self.GRASP_HEIGHT])
        pulled = T["point"] + T["dir"] * self.PULL_LENGTH + np.array([0.0, 0.0, 0.01])
        lifted = pulled + np.array([0.0, 0.0, self.LIFT_HEIGHT])
        yaw = T["yaw"]
        if name == "pregrasp":
            tcp = self.home_tcp * (1 - s) + above * s
            yaw = self.home_yaw * (1 - s) + yaw * s
            return tcp, yaw, FINGER_OPEN
        if name == "descend":
            return above * (1 - s) + T["point"] * s, yaw, FINGER_OPEN
        if name == "close":
            return T["point"], yaw, FINGER_OPEN + (self.FINGER_CLOSED - FINGER_OPEN) * s
        if name == "hold":
            return T["point"], yaw, self.FINGER_CLOSED
        if name == "pull":
            return T["point"] * (1 - u) + pulled * u, yaw, self.FINGER_CLOSED
        if name == "lift":
            return pulled * (1 - s) + lifted * s, yaw, self.FINGER_CLOSED
        if name == "release":
            return lifted, yaw, self.FINGER_CLOSED + (FINGER_OPEN - self.FINGER_CLOSED) * s
        return lifted + np.array([0.0, 0.0, 0.15 * s]), yaw, FINGER_OPEN  # retreat

    def _advance_phase(self):
        name, duration = self.PHASES[self.phase_idx]
        if name == "settle":
            # Leave settle once the pile has stayed at rest for REST_HOLD seconds (or the cap is hit).
            lin, ang = self.max_cable_speed()
            if lin < self.REST_LIN and ang < self.REST_ANG:
                self.rest_time += FRAME_DT
            else:
                self.rest_time = 0.0
            at_rest = self.rest_time >= self.REST_HOLD and self.phase_time >= self.MIN_SETTLE
            if not at_rest:
                if self.phase_time < self.settle_duration:
                    return
                print(
                    f"WARNING: pile not at rest after {self.settle_duration:.1f}s "
                    f"(max v={lin:.4f} m/s, w={ang:.3f} rad/s); starting the arm anyway"
                )
            self.settled_at = self.sim_time
        elif self.phase_time + 1e-9 < duration or self.phase_idx == len(self.PHASES) - 1:
            return
        # Leaving ``name``: record reachability / sticky attachment / metrics.
        if name in ("pregrasp", "descend"):
            self.ik_residuals[name] = float(np.linalg.norm(self.tcp_position() - self.cmd[0]))
        if name == "hold" and self.grasp == "sticky":
            self._attach_segment()
        if name == "pull":
            self.metrics_after = self.pile_metrics()
            self.links_after = self.link_distances()
            self.held_distance = self._held_distance()
        if name == "release":
            self.attached = None
        self.phase_idx += 1
        self.phase_time = 0.0
        new = self.PHASES[self.phase_idx][0]
        if new == "pregrasp":
            self._plan_grasp()
            self.home_tcp = self.tcp_position()
            self.home_yaw = self.cmd[1]
        elif new == "descend":
            # Re-plan from the current cable pose; the arm is still above, so it only shifts a little.
            before = (self.metrics_before, self.links_before)
            self._plan_grasp()
            self.metrics_before, self.links_before = before

    def _attach_segment(self):
        seg_body = self.target["cable"]["bodies"][self.target["seg_index"]]
        hand_q = wp.transform(*self.state.body_q.numpy()[self.hand_body])
        seg_q = wp.transform(*self.cable_state_0.body_q.numpy()[seg_body])
        self.attached = (seg_body, wp.transform_multiply(wp.transform_inverse(hand_q), seg_q))

    def _held_distance(self) -> float:
        """Distance between the grasped segment and the TCP (small => still held)."""
        T = self.target
        start = self.cable_starts(T["cable"])[T["seg_index"]]
        return float(np.linalg.norm(start - self.tcp_position()))

    # ------------------------------------------------------------ physics
    def apply_command(self, tcp, yaw, width):
        """Arm side of the step: IK to the commanded TCP pose, fingers to ``width``."""
        # Remember the pre-move hand poses; the mirrors sweep from them to the new ones.
        wp.launch(
            _snapshot_hand_poses,
            dim=len(self.hand_bodies),
            inputs=[self.state.body_q, self.hand_bodies, self.hand_prev_q],
        )
        self.cmd = (np.asarray(tcp, dtype=float), yaw, width)
        self.solve_ik(self.cmd[0], yaw)
        self.set_arm(width)

    def advance_physics(self):
        """Cable side of the step: sweep the mirrors and run the VBD substeps."""
        for sub in range(SIM_SUBSTEPS):
            wp.launch(
                _update_hand_mirrors,
                dim=len(self.hand_bodies),
                inputs=[
                    self.state.body_q,
                    self.hand_bodies,
                    self.hand_prev_q,
                    (sub + 1) / SIM_SUBSTEPS,
                    self.cable_state_0.body_q,
                    self.mirror_bodies,
                ],
            )
            if self.attached is not None:
                seg_body, local = self.attached
                wp.launch(
                    _drive_attached,
                    dim=1,
                    inputs=[
                        self.state.body_q,
                        self.hand_body,
                        local,
                        self.cable_state_0.body_q,
                        self.cable_state_0.body_qd,
                        seg_body,
                    ],
                )
            self.cable_state_0.clear_forces()
            self.cable_model.collide(self.cable_state_0, self.cable_contacts)
            self.cable_solver.step(
                self.cable_state_0, self.cable_state_1, self.cable_control, self.cable_contacts, SIM_DT
            )
            self.cable_state_0, self.cable_state_1 = self.cable_state_1, self.cable_state_0

    def step(self):
        if self.PHASES[self.phase_idx][0] == "settle":
            self.apply_command(*self.cmd)  # arm parked at home
        else:
            self.apply_command(*self.controller_command())
        self.advance_physics()
        self.sim_time += FRAME_DT
        self.frame += 1
        self.phase_time += FRAME_DT
        self._advance_phase()

    # ------------------------------------------------------------ metrics
    def cable_metrics(self, cable) -> dict:
        """End-to-end distance / length of one cable."""
        p = self.cable_starts(cable)
        length = cable["seg"] * (len(p) - 1)
        e2e = float(np.linalg.norm(p[-1] - p[0]))
        return {"ratio": e2e / length, "e2e": e2e, "length": length}

    def centroids(self) -> dict:
        """Centroid [x, y, z] of every cable, by name."""
        return {c["name"]: self.cable_starts(c).mean(axis=0) for c in self.cables}

    def pile_metrics(self) -> dict:
        """Per-cable centroid positions, used to see which cables a pull dragged along."""
        return {"centroids": self.centroids(), "A": self.cable_metrics(self.cable_by_name("cable_A"))}

    def link_distances(self) -> dict:
        """Closest approach between the centrelines of A-B and B-C.

        Cables that are hooked into each other stay within about one eye radius of each other
        while they are linked; once a link has opened the distance grows without bound.
        """
        pts = {n: self.cable_starts(self.cable_by_name(n)) for n in ("cable_A", "cable_B", "cable_C")}

        def closest(a, b):
            return float(np.linalg.norm(a[:, None] - b[None], axis=2).min())

        return {"A-B": closest(pts["cable_A"], pts["cable_B"]), "B-C": closest(pts["cable_B"], pts["cable_C"])}

    # ------------------------------------------------------------- render
    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        for k, c in enumerate(self.cables):
            xf, sc, col = self.capsule_buffers[k]
            wp.launch(
                _capsule_xforms,
                dim=len(c["bodies"]),
                inputs=[self.cable_state_0.body_q, c["bodies"][0], 0.5 * c["seg"], xf],
            )
            self.viewer.log_capsules(f"/cables/{c['name']}", "/geometry/capsule", xf, sc, col, None)
        self.viewer.end_frame()

    # --------------------------------------------------------------- test
    def test_final(self):
        q = self.cable_state_0.body_q.numpy()
        qd = self.cable_state_0.body_qd.numpy()
        assert np.isfinite(q).all() and np.isfinite(qd).all(), "non-finite cable state"
        for c in self.cables:
            p = q[c["bodies"], :3]
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            assert (np.abs(seg - c["seg"]) < 0.1 * c["seg"]).all(), f"{c['name']}: segment length off by >10%"
            assert p[:, 2].min() > -0.02, f"{c['name']} went through the ground"


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
    example = Example(viewer, args)
    newton.examples.run(example, args)
