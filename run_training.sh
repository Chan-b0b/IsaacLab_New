#!/bin/bash

# Activate conda environment
source ~/miniconda/bin/activate env_isaaclab

# Set environment variables for headless GPU rendering
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=0

# Disable Vulkan validation layers to avoid driver errors
export VK_LOADER_DEBUG=error
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

# Run training
cd /home/IsaacLab_New
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --num_envs 128 --device cuda:0
