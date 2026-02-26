"""AABB overlap check for RM75-6FB-V-with-gripper.

Spawns the merged URDF, builds world-space AABBs, and reports min distances
between wrist link_7 and all gripper links. Negative distance => penetration/overlap.
"""

from __future__ import annotations

from itertools import product

from pxr import UsdGeom

from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from realman.robot_rm75 import RM75_6FB_V_CFG


GRIPPER_LINKS = [
    "4C2_baselink",
    "4C2_Link1",
    "4C2_Link2",
    "4C2_Link3",
    "4C2_Link4",
    "4C2_Link5",
    "4C2_Link6",
]
ARM_LINK = "link_7"


def aabb_distance(a_min, a_max, b_min, b_max):
    """Return signed min distance between two AABBs (negative => overlap)."""
    sep = 0.0
    for i in range(3):
        if a_max[i] < b_min[i]:
            sep = max(sep, b_min[i] - a_max[i])
        elif b_max[i] < a_min[i]:
            sep = max(sep, a_min[i] - b_max[i])
    # If sep stayed 0, boxes overlap or touch; penetration depth estimate: min axis overlap
    if sep == 0.0:
        overlaps = [min(a_max[i], b_max[i]) - max(a_min[i], b_min[i]) for i in range(3)]
        return -min(overlaps)
    return sep


def main():
    app = AppLauncher(headless=True).app
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    robot_cfg = RM75_6FB_V_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=robot_cfg)

    sim.reset()
    stage = sim._stage  # noqa: SLF001

    # Build bbox cache in world space
    bbox_cache = UsdGeom.BBoxCache(sim_utils.constants.DEFAULT_TIME_CODE, includedPurposes=["default", "render", "proxy"], useExtentsHint=True)

    def prim_aabb(path: str):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None
        box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        return box.GetMin(), box.GetMax()

    arm_path = f"/World/Robot/{ARM_LINK}"
    arm_aabb = prim_aabb(arm_path)
    if arm_aabb is None:
        print(f"Arm link not found: {arm_path}")
        app.close()
        return

    print("AABB distances vs link_7 (negative => overlap):")
    for g in GRIPPER_LINKS:
        g_path = f"/World/Robot/{g}"
        g_aabb = prim_aabb(g_path)
        if g_aabb is None:
            print(f"  {g}: MISSING")
            continue
        dist = aabb_distance(arm_aabb[0], arm_aabb[1], g_aabb[0], g_aabb[1])
        print(f"  {g}: {dist:.5f} m")

    app.close()


if __name__ == "__main__":
    main()
