#!/usr/bin/env python3
"""Add a camera to the robot USD file."""

from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import omni.isaac.core.utils.stage as stage_utils
from pxr import Gf, Sdf, Usd, UsdGeom
import carb

# Load the robot USD
urdf_path = "/home/maverick/Humanoid/IsaacLab_New/source/realman/rm_models/RM75/urdf/RM75-6FB-V/urdf/RM75-6FB-V-robotiq.urdf"
output_usd = "/home/maverick/Humanoid/IsaacLab_New/source/realman/rm_models/RM75/urdf/RM75-6FB-V/urdf/RM75-6FB-V-robotiq_with_camera.usd"

print(f"Converting URDF to USD...")
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.importer.urdf import _urdf

# Import URDF
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = True

_, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=import_config,
)

print(f"Robot imported at: {prim_path}")

# Get stage
stage = omni.usd.get_context().get_stage()
robot_prim = stage.GetPrimAtPath(prim_path)

# Find link_7 (end-effector)
link7_path = f"{prim_path}/link_7"
link7_prim = stage.GetPrimAtPath(link7_path)

if not link7_prim.IsValid():
    print(f"ERROR: Could not find link_7 at {link7_path}")
    simulation_app.close()
    exit(1)

print(f"Found link_7 at: {link7_path}")

# Create camera under link_7
camera_path = f"{link7_path}/wrist_camera"
camera = UsdGeom.Camera.Define(stage, camera_path)

# Set camera properties
camera.GetHorizontalApertureAttr().Set(20.955)
camera.GetVerticalApertureAttr().Set(15.2908)
camera.GetFocalLengthAttr().Set(24.0)
camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 2.0))

# Position camera: 5cm forward from end-effector, looking down
camera_xform = UsdGeom.Xformable(camera)
camera_xform.ClearXformOpOrder()

# Translate forward
translate_op = camera_xform.AddTranslateOp()
translate_op.Set(Gf.Vec3d(0.05, 0.0, 0.0))

# Rotate to look forward/down (adjust as needed)
rotate_op = camera_xform.AddRotateXYZOp()
rotate_op.Set(Gf.Vec3d(0, 0, 0))  # Start with no rotation, can adjust in Isaac Sim

print(f"✓ Camera created at: {camera_path}")
print(f"  Position: (0.05, 0, 0) relative to link_7")
print(f"  You can now adjust the camera in Isaac Sim GUI")

# Save USD
layer = stage.GetRootLayer()
layer.Export(output_usd)

print(f"\n✓ USD saved to: {output_usd}")
print(f"\nNext steps:")
print(f"1. Open {output_usd} in Isaac Sim")
print(f"2. Select the camera: {camera_path}")
print(f"3. Adjust position/rotation using the transform gizmo")
print(f"4. Use 'View > Camera' to see what the camera sees")
print(f"5. Save the file when happy with camera placement")

simulation_app.close()
