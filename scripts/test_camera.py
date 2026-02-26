#!/usr/bin/env python3
"""Test script to verify camera is capturing images correctly."""

import torch
import matplotlib.pyplot as plt
import numpy as np

from isaaclab.app import AppLauncher

# Launch Isaac Sim with cameras enabled
app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks

# Import realman tasks to register the environment
import sys
sys.path.insert(0, "/home/maverick/Humanoid/IsaacLab_New/source")
import realman  # noqa: F401
sys.path.pop(0)

# Create environment with proper configuration
from realman.pick_place_env_cfg import RealmanPickPlaceEnvCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

# Test different camera offsets - third-person views from base_link
offsets_to_test = [
    {"pos": (0, 0.0, 0.01), 
     "rot": (0.6408,0.2988,-0.2988,0.6408),  # 90° pitch up (rotate around Y)
     "name": "front_mid"},
]

for i, offset_cfg in enumerate(offsets_to_test):
    print(f"\n{'='*60}")
    print(f"Testing offset {i+1}/{len(offsets_to_test)}: {offset_cfg['name']}")
    print(f"  pos={offset_cfg['pos']}, rot={offset_cfg['rot']}")
    print(f"{'='*60}")
    
    # Modify config
    env_cfg = RealmanPickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    
    # Disable command arrow visualization
    env_cfg.commands.object_pose.debug_vis = False
    
    # Update camera offset
    env_cfg.scene.wrist_cam.offset = CameraCfg.OffsetCfg(
        pos=offset_cfg['pos'],
        rot=offset_cfg['rot'],
        convention="world"
    )
    
    # Create environment directly
    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(cfg=env_cfg)
    obs, _ = env.reset()
    
    # Take a few steps to ensure camera is rendering
    for _ in range(300):
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, _, _, _, _ = env.step(action)
    
    # Get camera data
    camera_data = env.unwrapped.scene.sensors["wrist_cam"].data.output["rgb"]

    # Convert to numpy and save
    img = camera_data[0].cpu().numpy()
    print(f"  Image range: [{img.min()}, {img.max()}], mean: {img.mean():.1f}")
    
    # Normalize for better visualization
    img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Save both original and normalized images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(img.astype(np.uint8))
    ax1.set_title(f"Original\npos={offset_cfg['pos']}")
    ax1.axis('off')
    
    ax2.imshow(img_normalized)
    ax2.set_title(f"Normalized\npos={offset_cfg['pos']}")
    ax2.axis('off')
    
    output_path = f"/home/maverick/Humanoid/IsaacLab_New/camera_test_{offset_cfg['name']}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")
    
    # Cleanup
    env.close()

print(f"\n{'='*60}")
print("✓ All camera tests complete! Check the saved images.")
print(f"{'='*60}")
simulation_app.close()
