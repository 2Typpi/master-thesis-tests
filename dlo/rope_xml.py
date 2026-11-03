"""Generate a capsule-link rope MJCF with the naming expected by the DER controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class RopeConfig:
    length: float = 2.0
    segments: int = 40
    radius: float = 0.01
    damping: float = 1.0
    density: float = 1000.0
    initial_position: tuple[float, float, float] = (0.6, 0.35, 0.01)
    initial_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    color: tuple[float, float, float, float] = (0.8, 0.2, 0.1, 1.0)
    collision_type: int = 4
    collision_affinity: int = 7
    excluded_bodies: tuple[str, ...] = (
        "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
        "hand", "left_finger", "right_finger",
    )

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.segments < 3:
            raise ValueError("segments must be at least 3")
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.length / self.segments <= 2 * self.radius:
            raise ValueError(f"segment length must be greater than capsule diameter {self.length / self.segments} <= {2 * self.radius}")
        if self.damping < 0 or self.density <= 0:
            raise ValueError("damping must be nonnegative and density must be positive")
        if self.collision_type < 0 or self.collision_affinity < 0:
            raise ValueError("collision masks must be nonnegative")


def _numbers(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _link_body_name(index: int, segments: int) -> str:
    if index == 0:
        return "B_first"
    if index == segments - 1:
        return "B_last"
    return f"B_{index}"


def generate_rope_xml(config: RopeConfig = RopeConfig()) -> str:
    """Return MJCF for a free capsule chain; the root body's free joint supplies six dofs."""
    step = config.length / config.segments
    root = ET.Element("mujoco", {"model": "der_rope"})
    worldbody = ET.SubElement(root, "worldbody")
    rope = ET.SubElement(worldbody, "body", {
        "name": "stiffrope",
        "pos": _numbers(config.initial_position),
        "quat": _numbers(config.initial_quaternion),
    })
    ET.SubElement(rope, "freejoint", {"name": "freejoint_A"})
    ET.SubElement(rope, "site", {"name": "ft_rope", "pos": "0 0 0", "size": f"{config.radius:g}", "rgba": "1 0 0 0"})

    parent = rope
    for index in range(config.segments):
        body_attrs = {"name": _link_body_name(index, config.segments)}
        if index:
            body_attrs["pos"] = f"{step:.9g} 0 0"
        body = ET.SubElement(parent, "body", body_attrs)
        if index:
            joint_name = "J_last" if index == config.segments - 1 else f"J_{index}"
            ET.SubElement(body, "joint", {
                "name": joint_name,
                "type": "ball",
                "pos": "0 0 0",
                "stiffness": "0",
                "damping": f"{config.damping:g}",
                "actuatorfrclimited": "false",
            })
        if index == 0:
            ET.SubElement(body, "site", {"name": "S_first", "pos": "0 0 0", "size": f"{config.radius * 0.45:g}"})
        ET.SubElement(body, "site", {
            "name": f"S_{index}", "pos": "0 0 0", "size": f"{config.radius * 0.45:g}", "rgba": "0 0 0 0",
        })
        ET.SubElement(body, "geom", {
            "name": f"G{index}",
            "type": "capsule",
            "pos": f"{step / 2:.9g} 0 0",
            "quat": "0.707106781 0 -0.707106781 0",
            "size": f"{config.radius:.9g} {step / 2:.9g}",
            "rgba": _numbers(config.color),
            "density": f"{config.density:g}",
            "contype": str(config.collision_type),
            "conaffinity": str(config.collision_affinity),
            "solref": "0.001 1",
            "condim": "3",
        })
        parent = body

    ET.SubElement(parent, "site", {
        "name": "S_last", "pos": f"{step:.9g} 0 0", "size": f"{config.radius * 0.45:g}",
    })
    ET.SubElement(parent, "body", {"name": "B_last2", "pos": f"{step:.9g} 0 0"})

    contact = ET.SubElement(root, "contact")
    for index in range(config.segments - 1):
        ET.SubElement(contact, "exclude", {
            "body1": _link_body_name(index, config.segments),
            "body2": _link_body_name(index + 1, config.segments),
        })

    return ET.tostring(root, encoding="unicode")


def write_rope_xml(path: str | Path, config: RopeConfig = RopeConfig()) -> Path:
    """Write generated MJCF to a caller-selected path and return the resolved path."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(generate_rope_xml(config) + "\n", encoding="utf-8")
    return target
