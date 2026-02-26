#!/usr/bin/env python3
"""Callback to log camera images to tensorboard during training."""

import os
import torch
import torchvision
from typing import Any


class CameraImageLogger:
    """Logs camera images to tensorboard periodically."""
    
    def __init__(self, env, runner, log_interval: int = 500, num_images: int = 4):
        """
        Args:
            env: The IsaacLab environment
            runner: RSL-RL OnPolicyRunner (has writer attribute)
            log_interval: Log images every N training steps
            num_images: Number of environment images to log (grid)
        """
        self.env = env
        self.runner = runner
        self.log_interval = log_interval
        self.num_images = min(num_images, env.num_envs)
        self.step_count = 0
        
    def __call__(self, iteration: int) -> None:
        """Called after each training iteration."""
        self.step_count += 1
        
        # Skip first few iterations to let rendering initialize
        if iteration < 10:
            return
        
        if self.step_count % self.log_interval == 0:
            # Get camera data from environment
            try:
                camera_sensor = self.env.scene.sensors["wrist_cam"]
                camera_data = camera_sensor.data.output.get("rgb")
                
                if camera_data is not None:
                    # Take first N environments
                    images = camera_data[:self.num_images]  # [N, H, W, 3]
                    
                    # Convert to [N, 3, H, W] and normalize to [0, 1]
                    images = images.permute(0, 3, 1, 2).float()
                    if images.max() > 1.0:
                        images = images / 255.0
                    
                    # Create image grid
                    grid = torchvision.utils.make_grid(images, nrow=2, normalize=False)
                    
                    # Log to tensorboard (access writer at runtime)
                    if hasattr(self.runner, 'writer') and self.runner.writer is not None:
                        self.runner.writer.add_image(
                            "camera/wrist_cam_rgb",
                            grid,
                            global_step=iteration,
                        )
                        print(f"[CameraLogger] Logged {self.num_images} camera images at iteration {iteration}")
                    else:
                        print(f"[CameraLogger] Warning: Writer not available at iteration {iteration}")
            except Exception as e:
                print(f"[CameraLogger] Warning: Could not log camera images: {e}")
