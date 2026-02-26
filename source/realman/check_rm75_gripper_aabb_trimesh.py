"""AABB overlap check without Omniverse/USD.

Loads RM75-6FB-V-with-gripper URDF and meshes via trimesh, computes world-space
AABBs at nominal joint positions (arm joints 0, gripper joint 0.4 rad), and
prints signed distances from link_7 to each gripper link. Negative => overlap.

Requires: numpy, trimesh
Install if needed: `pip install numpy trimesh`
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

# Paths
ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "realman" / "rm_models" / "RM75" / "urdf" / "RM75-6FB-V" / "urdf" / "RM75-6FB-V-with-gripper.urdf"

# Joint defaults (arm zeros, gripper slightly open)
JOINT_POS = {
    "4C2_Joint1": 0.4,
}

ARM_LINK = "link_7"
GRIPPER_LINKS = [
    "4C2_baselink",
    "4C2_Link1",
    "4C2_Link2",
    "4C2_Link3",
    "4C2_Link4",
    "4C2_Link5",
    "4C2_Link6",
]

PACKAGE_PATHS = {
    "RM75-6FB-V": ROOT / "realman" / "rm_models" / "RM75" / "urdf" / "RM75-6FB-V",
    "EG2-4C2": ROOT / "realman" / "rm_models" / "thirdparty" / "Two-finger Electric Gripper.urdf",
}

def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def make_transform(translation, rotation_rpy) -> np.ndarray:
    tx, ty, tz = translation
    rr, rp, ry = rotation_rpy
    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(rr), -math.sin(rr)],
            [0, math.sin(rr), math.cos(rr)],
        ]
    )
    Ry = np.array(
        [
            [math.cos(rp), 0, math.sin(rp)],
            [0, 1, 0],
            [-math.sin(rp), 0, math.cos(rp)],
        ]
    )
    Rz = np.array(
        [
            [math.cos(ry), -math.sin(ry), 0],
            [math.sin(ry), math.cos(ry), 0],
            [0, 0, 1],
        ]
    )
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def parse_origin(elem):
    if elem is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xyz = tuple(map(float, elem.attrib.get("xyz", "0 0 0").split()))
    rpy = tuple(map(float, elem.attrib.get("rpy", "0 0 0").split()))
    return xyz, rpy


def resolve_mesh(filename: str) -> Path:
    if filename.startswith("package://"):
        rest = filename[len("package://") :]
        package, rel = rest.split("/", 1)
        base = PACKAGE_PATHS.get(package)
        if base is None:
            raise FileNotFoundError(f"Unknown package {package} in {filename}")
        return base / rel
    return Path(filename)


def load_urdf(urdf_path: Path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {l.attrib["name"]: l for l in root.findall("link")}
    joints = []
    for j in root.findall("joint"):
        joints.append(
            {
                "name": j.attrib["name"],
                "type": j.attrib["type"],
                "parent": j.find("parent").attrib["link"],
                "child": j.find("child").attrib["link"],
                "origin": parse_origin(j.find("origin")),
                "axis": tuple(map(float, j.find("axis").attrib.get("xyz", "0 0 1").split())) if j.find("axis") is not None else (0.0, 0.0, 1.0),
                "mimic": j.find("mimic").attrib if j.find("mimic") is not None else None,
            }
        )
    return links, joints


def topological_sort(joints):
    parents = {j["child"]: j for j in joints}
    order = []
    def dfs(link):
        for j in joints:
            if j["parent"] == link:
                dfs(j["child"])
                order.append(j)
    dfs("base_link")
    return order


def compute_transforms(links, joints):
    # Initialize poses
    T = {"base_link": np.eye(4)}
    # Build mapping child->joint for traversal
    # Simple forward pass assuming tree from base_link
    pending = list(joints)
    while pending:
        progressed = False
        for j in pending[:]:
            parent = j["parent"]
            child = j["child"]
            if parent not in T:
                continue
            xyz, rpy = j["origin"]
            T_joint = make_transform(xyz, rpy)
            angle = JOINT_POS.get(j["name"], 0.0)
            if j["mimic"] is not None:
                ref = j["mimic"].get("joint")
                mult = float(j["mimic"].get("multiplier", 1.0))
                offset = float(j["mimic"].get("offset", 0.0))
                angle = JOINT_POS.get(ref, 0.0) * mult + offset
            if j["type"] == "revolute" or j["type"] == "continuous":
                axis = np.array(j["axis"], dtype=float)
                R_axis = axis_angle_to_matrix(axis, angle)
                T_rot = np.eye(4)
                T_rot[:3, :3] = R_axis
                T_child_local = T_joint @ T_rot
            else:
                T_child_local = T_joint
            T[child] = T[parent] @ T_child_local
            pending.remove(j)
            progressed = True
        if not progressed:
            break
    return T


def link_mesh_aabb(link_elem, T_world):
    # use first visual mesh
    vis = link_elem.find("visual")
    if vis is None:
        return None
    geom = vis.find("geometry")
    if geom is None or geom.find("mesh") is None:
        return None
    mesh_file = geom.find("mesh").attrib.get("filename")
    if mesh_file is None:
        return None
    mesh_path = resolve_mesh(mesh_file)
    if not mesh_path.is_file():
        return None
    mesh = trimesh.load(mesh_path, force="mesh")
    origin = vis.find("origin")
    xyz, rpy = parse_origin(origin)
    T_vis = make_transform(xyz, rpy)
    T_total = T_world @ T_vis
    mesh.apply_transform(T_total)
    return np.array(mesh.bounds[0]), np.array(mesh.bounds[1])


def aabb_distance(a_min, a_max, b_min, b_max):
    sep = 0.0
    for i in range(3):
        if a_max[i] < b_min[i]:
            sep = max(sep, b_min[i] - a_max[i])
        elif b_max[i] < a_min[i]:
            sep = max(sep, a_min[i] - b_max[i])
    if sep == 0.0:
        overlaps = [min(a_max[i], b_max[i]) - max(a_min[i], b_min[i]) for i in range(3)]
        return -min(overlaps)
    return sep


def main():
    links, joints = load_urdf(URDF_PATH)
    T = compute_transforms(links, joints)

    arm_link = ARM_LINK
    if arm_link not in T:
        print(f"Missing link transform: {arm_link}")
        return
    arm_aabb = link_mesh_aabb(links[arm_link], T[arm_link])
    if arm_aabb is None:
        print(f"No mesh for {arm_link}")
        return

    print("AABB distances vs link_7 (negative => overlap):")
    for g in GRIPPER_LINKS:
        if g not in T:
            print(f"  {g}: MISSING transform")
            continue
        g_aabb = link_mesh_aabb(links[g], T[g])
        if g_aabb is None:
            print(f"  {g}: missing mesh")
            continue
        dist = aabb_distance(arm_aabb[0], arm_aabb[1], g_aabb[0], g_aabb[1])
        print(f"  {g}: {dist:.5f} m")


if __name__ == "__main__":
    main()
