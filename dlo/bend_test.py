"""Standalone bending test for the DER cable, independent of the Panda model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .controller import DloRope
from .rope_xml import RopeConfig, generate_rope_xml
from .sim import step


@dataclass
class BendSimulation:
    model: mujoco.MjModel
    data: mujoco.MjData
    rope: DloRope
    start_mocap_id: int
    end_mocap_id: int
    start_anchor: np.ndarray
    end_anchor: np.ndarray
    start_weld_id: int
    end_weld_id: int
    start_visual_geom_id: int
    end_visual_geom_id: int
    released: set[str]


@dataclass
class BendMetrics:
    arc_length: float
    chord: float
    sag: float
    max_lateral_deflection: float
    max_curvature: float
    bending_energy: float
    twist: float
    tip_height: float
    tip_offset: float
    max_anchor_error: float
    max_speed: float


def _scene_xml(config: RopeConfig, gravity: bool, timestep: float, iterations: int) -> str:
    rope = ET.fromstring(generate_rope_xml(config))
    root = ET.Element("mujoco", {"model": "der_bend_test"})
    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local", "autolimits": "true"})
    ET.SubElement(root, "option", {
        "timestep": f"{timestep:.9g}",
        "gravity": "0 0 -9.81" if gravity else "0 0 0",
        "solver": "Newton",
        "iterations": str(iterations),
        "tolerance": "1e-10",
        "jacobian": "sparse",
        "cone": "elliptic",
        "integrator": "implicit",
    })
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "geom", {
        "name": "floor",
        "type": "plane",
        "pos": "0 0 0",
        "size": "3 3 0.05",
        "rgba": "0.25 0.3 0.35 1",
        "contype": "1",
        "conaffinity": "4",
    })
    ET.SubElement(worldbody, "light", {"pos": "0 0 3", "dir": "0 0 -1", "directional": "true"})

    for child in rope.find("worldbody"):
        worldbody.append(child)
    step_length = config.length / config.segments
    start = np.asarray(config.initial_position, dtype=np.float64)
    end = start + np.array([(config.segments - 1) * step_length, 0.0, 0.0])
    quat = " ".join(f"{value:.9g}" for value in config.initial_quaternion)
    for name, position in (("bend_anchor_start", start), ("bend_anchor_end", end)):
        anchor = ET.SubElement(worldbody, "body", {
            "name": name,
            "mocap": "true",
            "pos": " ".join(f"{value:.9g}" for value in position),
            "quat": quat,
        })
        ET.SubElement(anchor, "geom", {
            "name": f"{name}_visual",
            "type": "sphere",
            "size": f"{config.radius * 2:g}",
            "rgba": "0.15 0.65 0.9 0.7",
            "contype": "0",
            "conaffinity": "0",
        })

    contact = ET.SubElement(root, "contact")
    rope_contact = rope.find("contact")
    if rope_contact is not None:
        for child in rope_contact:
            contact.append(child)
    equality = ET.SubElement(root, "equality")
    for weld_name, body_name, anchor_name in (
        ("weld_start", "B_first", "bend_anchor_start"),
        ("weld_end", "B_last", "bend_anchor_end"),
    ):
        ET.SubElement(equality, "weld", {
            "name": weld_name,
            "body1": body_name,
            "body2": anchor_name,
            "solref": "0.001 1",
        })

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"azimuth": "90", "elevation": "-20"})
    ET.SubElement(root, "statistic", {"extent": "2.5", "meansize": "0.05"})
    return ET.tostring(root, encoding="unicode")


def build_simulation(
    *,
    alpha_bar: float = 1.345,
    beta_bar: float = 0.789,
    gravity: bool = True,
    timestep: float = 0.0015,
    iterations: int = 30,
    config: RopeConfig | None = None,
) -> BendSimulation:
    """Create a horizontally suspended rope with both end frames welded to mocap anchors."""
    if alpha_bar < 0 or beta_bar < 0:
        raise ValueError("stiffness values must be nonnegative")
    if config is None:
        config = RopeConfig(
            initial_position=(-0.975, 0.0, 1.0),
            collision_type=4,
            collision_affinity=1,
            excluded_bodies=(),
        )
    model = mujoco.MjModel.from_xml_string(_scene_xml(config, gravity, timestep, iterations))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    start_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bend_anchor_start")
    end_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bend_anchor_end")
    if start_body_id < 0 or end_body_id < 0:
        raise RuntimeError("failed to build mocap anchor bodies")
    start_mocap_id = int(model.body_mocapid[start_body_id])
    end_mocap_id = int(model.body_mocapid[end_body_id])
    if start_mocap_id < 0 or end_mocap_id < 0:
        raise RuntimeError("bend anchors must be mocap bodies")
    start_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "weld_start")
    end_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "weld_end")
    start_visual_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bend_anchor_start_visual")
    end_visual_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bend_anchor_end_visual")
    if min(start_weld_id, end_weld_id, start_visual_geom_id, end_visual_geom_id) < 0:
        raise RuntimeError("failed to resolve named bend anchors, welds, or marker geoms")

    rope = DloRope(
        model,
        data,
        config.segments,
        overall_rot=0.0,
        alpha_bar=alpha_bar,
        beta_bar=beta_bar,
    )
    return BendSimulation(
        model=model,
        data=data,
        rope=rope,
        start_mocap_id=start_mocap_id,
        end_mocap_id=end_mocap_id,
        start_anchor=data.mocap_pos[start_mocap_id].copy(),
        end_anchor=data.mocap_pos[end_mocap_id].copy(),
        start_weld_id=start_weld_id,
        end_weld_id=end_weld_id,
        start_visual_geom_id=start_visual_geom_id,
        end_visual_geom_id=end_visual_geom_id,
        released=set(),
    )


def release_anchor(sim: BendSimulation, which: str = "end") -> None:
    """Deactivate one cable-end weld so the selected endpoint becomes free."""
    if which not in {"start", "end"}:
        raise ValueError("which must be 'start' or 'end'")
    if which in sim.released:
        return

    weld_id = sim.start_weld_id if which == "start" else sim.end_weld_id
    marker_geom_id = sim.start_visual_geom_id if which == "start" else sim.end_visual_geom_id
    sim.data.eq_active[weld_id] = False
    sim.model.geom_rgba[marker_geom_id, 3] = 0.0
    sim.released.add(which)


def measure(sim: BendSimulation) -> BendMetrics:
    mujoco.mj_forward(sim.model, sim.data)
    nodes = sim.data.site_xpos[sim.rope.site_ids]
    edges = np.diff(nodes, axis=0)
    edge_lengths = np.linalg.norm(edges, axis=1)
    tangents = edges / np.maximum(edge_lengths[:, None], np.finfo(np.float64).tiny)
    turning_angles = np.arccos(np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0))
    local_lengths = 0.5 * (edge_lengths[:-1] + edge_lengths[1:])
    curvature = turning_angles / np.maximum(local_lengths, np.finfo(np.float64).tiny)

    endpoint_z = 0.5 * (nodes[0, 2] + nodes[-1, 2])
    anchor_errors = []
    anchor_specs = (
        ("start", "B_first", sim.start_mocap_id),
        ("end", "B_last", sim.end_mocap_id),
    )
    for which, body_name, mocap_id in anchor_specs:
        if which in sim.released:
            continue
        body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        target = sim.data.mocap_pos[mocap_id]
        anchor_errors.append(float(np.linalg.norm(sim.data.xpos[body_id] - target)))
    remaining_anchor_id = sim.start_mocap_id if "end" in sim.released else sim.end_mocap_id
    free_tip = nodes[-1] if "end" in sim.released else nodes[0]
    tip_offset = float(np.linalg.norm(free_tip[:2] - sim.data.mocap_pos[remaining_anchor_id, :2]))

    return BendMetrics(
        arc_length=float(np.sum(edge_lengths)),
        chord=float(np.linalg.norm(nodes[-1] - nodes[0])),
        sag=float(np.max(np.abs(nodes[:, 2] - endpoint_z))),
        max_lateral_deflection=float(np.max(np.abs(nodes[:, 1] - 0.5 * (nodes[0, 1] + nodes[-1, 1])))),
        max_curvature=float(np.max(curvature)) if len(curvature) else 0.0,
        bending_energy=float(sim.rope.dlo_math.calculateEnergy()),
        tip_height=float(free_tip[2]),
        tip_offset=tip_offset,
        twist=float(sim.rope.theta_n),
        max_anchor_error=max(anchor_errors, default=0.0),
        max_speed=float(np.max(np.linalg.norm(sim.data.cvel[sim.rope.body_ids, 3:], axis=1))),
    )


def _advance(
    sim: BendSimulation,
    steps: int,
    target_anchors: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    start_from = sim.data.mocap_pos[sim.start_mocap_id].copy()
    end_from = sim.data.mocap_pos[sim.end_mocap_id].copy()
    for index in range(steps):
        if target_anchors is not None:
            fraction = (index + 1) / steps
            smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            start_to, end_to = target_anchors
            sim.data.mocap_pos[sim.start_mocap_id] = start_from + smooth_fraction * (start_to - start_from)
            sim.data.mocap_pos[sim.end_mocap_id] = end_from + smooth_fraction * (end_to - end_from)
        step(sim.model, sim.data, sim.rope)


def _compression_target(sim: BendSimulation, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= fraction < 0.5:
        raise ValueError("compression must be in [0, 0.5)")
    span = sim.end_anchor - sim.start_anchor
    offset = 0.5 * fraction * span
    return sim.start_anchor + offset, sim.end_anchor - offset


def run_protocol(
    sim: BendSimulation,
    *,
    settle_seconds: float = 3.0,
    compress_seconds: float = 15.0,
    hold_seconds: float = 2.0,
    drop_seconds: float = 0.0,
    drop_anchor: str = "end",
    compression: float = 0.2,
    on_sample=None,
) -> dict[str, BendMetrics]:
    """Settle under gravity, smoothly shorten the end span, then hold."""
    dt = float(sim.model.opt.timestep)
    initial = measure(sim)
    if on_sample:
        on_sample("initial", sim, initial)
    _advance(sim, round(settle_seconds / dt))
    settled = measure(sim)
    if on_sample:
        on_sample("settled", sim, settled)

    target = _compression_target(sim, compression)
    _advance(sim, max(1, round(compress_seconds / dt)), target)
    compressed = measure(sim)
    if on_sample:
        on_sample("compressed", sim, compressed)

    _advance(sim, round(hold_seconds / dt))
    held = measure(sim)
    if on_sample:
        on_sample("held", sim, held)

    reports = {"initial": initial, "settled": settled, "compressed": compressed, "held": held}
    if drop_seconds > 0:
        release_anchor(sim, drop_anchor)
        dropped = measure(sim)
        if on_sample:
            on_sample("released", sim, dropped)
        sample_steps = max(1, round(0.5 / dt))
        remaining_steps = round(drop_seconds / dt)
        elapsed_steps = 0
        while remaining_steps > 0:
            chunk = min(sample_steps, remaining_steps)
            _advance(sim, chunk)
            elapsed_steps += chunk
            remaining_steps -= chunk
            dropped = measure(sim)
            reports[f"drop +{elapsed_steps * dt:.1f}s"] = dropped
            if on_sample:
                on_sample(f"drop +{elapsed_steps * dt:.1f}s", sim, dropped)
        reports["dropped"] = dropped
        if on_sample:
            on_sample("dropped", sim, dropped)
    return reports


def print_report(reports: dict[str, BendMetrics], alpha_bar: float) -> None:
    print(f"alpha_bar={alpha_bar:g} (dimensionless DER stiffness; absolute EI mapping not assumed)")
    print("stage       arc length   chord      sag       lateral    max curvature   bend energy    twist       tip z     tip offset   anchor error   max speed")
    for stage, metrics in reports.items():
        print(
            f"{stage:10s} {metrics.arc_length:10.5f} {metrics.chord:9.5f} {metrics.sag:9.5f} "
            f"{metrics.max_lateral_deflection:9.5f} {metrics.max_curvature:14.5g} {metrics.bending_energy:13.5g} {metrics.twist:10.5g} "
            f"{metrics.tip_height:9.5f} {metrics.tip_offset:11.5f} {metrics.max_anchor_error:13.5g} {metrics.max_speed:11.5g}"
        )


def _stiffness_sweep(args) -> list[tuple[float, BendMetrics]]:
    results = []
    for alpha in (0.0, 0.3, 1.345, 5.0):
        sim = build_simulation(
            alpha_bar=alpha,
            beta_bar=args.beta,
            gravity=not args.no_gravity,
            timestep=args.timestep,
            iterations=args.iterations,
        )
        _advance(sim, round(args.settle / sim.model.opt.timestep))
        results.append((alpha, measure(sim)))
    return results


def _plot_shapes(shapes: list[tuple[str, np.ndarray]], path: str) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 4.5))
    for label, nodes in shapes:
        axis.plot(nodes[:, 0], nodes[:, 2], marker=".", markersize=2, label=label)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("z (m)")
    axis.set_title("DER cable bending under self-weight and end compression")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    print(f"Saved shape plot to {path}")


def _run_checks(args) -> None:
    stiffness_results = _stiffness_sweep(args)
    print("settle stiffness sweep:")
    for alpha, metrics in stiffness_results:
        print(
            f"alpha={alpha:5.3f} sag={metrics.sag:.6f} m, lateral={metrics.max_lateral_deflection:.6f} m, "
            f"arc={metrics.arc_length:.6f} m, max-curvature={metrics.max_curvature:.5g} 1/m, energy={metrics.bending_energy:.5g}"
        )

    curvatures = [metrics.max_curvature for _, metrics in stiffness_results]
    energies = [metrics.bending_energy for _, metrics in stiffness_results]
    assert all(math.isfinite(value) for value in curvatures + energies), "non-finite stiffness metric"
    assert curvatures[0] > curvatures[2], "zero-bending control should curve more than the nominal-stiffness rope"
    assert energies[0] <= energies[1] <= energies[2] <= energies[3], "bending energy should increase with alpha_bar"
    assert all(abs(metrics.arc_length - 2.0) < 0.03 for _, metrics in stiffness_results), "rope arc length drifted"

    sim = build_simulation(
        alpha_bar=args.alpha,
        beta_bar=args.beta,
        gravity=not args.no_gravity,
        timestep=args.timestep,
        iterations=args.iterations,
    )
    reports = run_protocol(
        sim,
        settle_seconds=args.settle,
        compress_seconds=args.compress_seconds,
        hold_seconds=args.hold,
        compression=args.compress,
    )
    print("compression protocol:")
    print_report(reports, args.alpha)
    compressed = reports["compressed"]
    held = reports["held"]
    assert abs(compressed.arc_length - 2.0) < 0.03, "rope stretched or lost length during compression"
    assert compressed.chord < reports["settled"].chord, "anchor compression did not shorten the cable chord"
    assert abs(compressed.twist) < 0.2, "planar bend test accumulated unexpected twist"
    assert compressed.bending_energy >= reports["settled"].bending_energy - 1e-8, "bending energy fell during compression"
    assert held.max_speed < max(0.05, reports["compressed"].max_speed * 2.0), "rope did not settle after compression"
    assert max(reports["compressed"].max_anchor_error, held.max_anchor_error) < 0.01, "weld anchors drifted"

    drop_sim = build_simulation(
        alpha_bar=args.alpha,
        beta_bar=args.beta,
        gravity=not args.no_gravity,
        timestep=args.timestep,
        iterations=args.iterations,
    )
    drop_reports = run_protocol(
        drop_sim,
        settle_seconds=args.settle,
        compress_seconds=args.compress_seconds,
        hold_seconds=args.hold,
        drop_seconds=args.drop,
        drop_anchor=args.drop_anchor,
        compression=args.compress,
    )
    if args.drop > 0:
        released = drop_reports["held"]
        dropped = drop_reports["dropped"]
        print("anchor-release check:")
        print_report({"held": released, "dropped": dropped}, args.alpha)
        assert np.isfinite(drop_sim.data.qpos).all() and np.isfinite(drop_sim.data.qvel).all(), "non-finite state after release"
        assert abs(dropped.arc_length - 2.0) < 0.1, "rope stretched or lost length after release"
        assert dropped.max_anchor_error < 0.01, "remaining anchor drifted after release"
        assert dropped.tip_height < released.tip_height - 0.05, "released tip did not fall"
        minimum_height = float(np.min(drop_sim.data.site_xpos[drop_sim.rope.site_ids, 2]))
        assert minimum_height >= -RopeConfig().radius - 1e-3, "rope penetrated the floor"


def _run_viewer(args) -> list[tuple[str, np.ndarray]]:
    import mujoco.viewer

    sim = build_simulation(
        alpha_bar=args.alpha,
        beta_bar=args.beta,
        gravity=not args.no_gravity,
        timestep=args.timestep,
        iterations=args.iterations,
    )
    dt = float(sim.model.opt.timestep)
    shapes = []
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        stages = [
            ("settle", round(args.settle / dt), None),
            ("compress", round(args.compress_seconds / dt), _compression_target(sim, args.compress)),
            ("hold", round(args.hold / dt), None),
        ]
        if args.drop > 0:
            stages.append(("drop", round(args.drop / dt), None))
        for name, count, target in stages:
            if name == "drop":
                release_anchor(sim, args.drop_anchor)
                release_metrics = measure(sim)
                print_report({f"{args.drop_anchor} anchor released": release_metrics}, args.alpha)
            started = time.perf_counter()
            for index in range(count):
                if target is not None:
                    fraction = (index + 1) / count
                    smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
                    start_to, end_to = target
                    sim.data.mocap_pos[sim.start_mocap_id] = sim.start_anchor + smooth_fraction * (start_to - sim.start_anchor)
                    sim.data.mocap_pos[sim.end_mocap_id] = sim.end_anchor + smooth_fraction * (end_to - sim.end_anchor)
                step(sim.model, sim.data, sim.rope)
                if index % 10 == 0:
                    viewer.sync()
                    elapsed = time.perf_counter() - started
                    remaining = (index + 1) * dt - elapsed
                    if remaining > 0:
                        time.sleep(min(remaining, 0.01))
                if name == "drop" and (index + 1) % max(1, round(0.5 / dt)) == 0:
                    viewer.sync()
                    drop_elapsed = (index + 1) * dt
                    metrics = measure(sim)
                    print_report({f"drop +{drop_elapsed:.1f}s": metrics}, args.alpha)
                    shapes.append((f"drop +{drop_elapsed:.1f}s", sim.data.site_xpos[sim.rope.site_ids].copy()))
            viewer.sync()
            metrics = measure(sim)
            print_report({name: metrics}, args.alpha)
            print(f"  anchor error={metrics.max_anchor_error:.6g} m; max speed={metrics.max_speed:.6g} m/s")
            shapes.append((name, sim.data.site_xpos[sim.rope.site_ids].copy()))
    return shapes


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run headless physics and stiffness checks")
    parser.add_argument("--plot", metavar="PATH", help="save the recorded x-z cable shapes as an image")
    parser.add_argument("--alpha", type=float, default=1.345, help="DER bending stiffness")
    parser.add_argument("--beta", type=float, default=0.789, help="DER twist stiffness")
    parser.add_argument("--compress", type=float, default=0.2, help="fractional end-span shortening, in [0, 0.5)")
    parser.add_argument("--settle", type=float, default=3.0, help="seconds to settle before compression")
    parser.add_argument("--compress-seconds", type=float, default=15.0, help="seconds over which to move the end anchor")
    parser.add_argument("--hold", type=float, default=2.0, help="seconds to hold after compression")
    parser.add_argument("--drop", type=float, default=6.0, help="seconds to simulate after releasing one anchor; 0 disables")
    parser.add_argument("--drop-anchor", choices=("end", "start"), default="end", help="which cable endpoint to release")
    parser.add_argument("--timestep", type=float, default=0.0015, help="MuJoCo integration timestep")
    parser.add_argument("--iterations", type=int, default=30, help="Newton solver iterations")
    parser.add_argument("--no-gravity", action="store_true", help="disable gravity for a pure compression demo")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.alpha < 0 or args.beta < 0:
        raise SystemExit("alpha and beta must be nonnegative")
    if not 0.0 <= args.compress < 0.5:
        raise SystemExit("--compress must be in [0, 0.5)")
    if args.drop < 0:
        raise SystemExit("--drop must be nonnegative")
    if args.check:
        _run_checks(args)
        if args.plot:
            sim = build_simulation(alpha_bar=args.alpha, beta_bar=args.beta, gravity=not args.no_gravity,
                                   timestep=args.timestep, iterations=args.iterations)
            shapes = []
            def record(stage: str, current: BendSimulation, metrics: BendMetrics) -> None:
                if stage in ("settled", "compressed", "held") or stage.startswith("drop +"):
                    print_report({stage: metrics}, args.alpha)
                    shapes.append((stage, current.data.site_xpos[current.rope.site_ids].copy()))
            run_protocol(sim, settle_seconds=args.settle, compress_seconds=args.compress_seconds,
                         hold_seconds=args.hold, drop_seconds=args.drop, drop_anchor=args.drop_anchor,
                         compression=args.compress, on_sample=record)
            _plot_shapes(shapes, args.plot)
    else:
        shapes = _run_viewer(args)
        if args.plot:
            _plot_shapes(shapes, args.plot)


if __name__ == "__main__":
    main()
