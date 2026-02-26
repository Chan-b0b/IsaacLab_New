"""Quick visual check: spawn RM75-6FB-V with gripper only and view overlap."""

from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from realman.robot_rm75 import RM75_6FB_V_CFG


# Launch non-headless so you can inspect visually.
app_launcher = AppLauncher(headless=False)
app = app_launcher.app

# Basic simulation context (no ground, no extras)
sim_cfg = sim_utils.SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([1.0, 0.0, 0.5], [0.0, 0.0, 0.3])

# Spawn the robot at origin
robot_cfg = RM75_6FB_V_CFG.replace(prim_path="/World/Robot")
robot = Articulation(cfg=robot_cfg)

# Reset and step a few frames to let assets load
sim.reset()
sim.play()
for _ in range(10):
    sim.step()

# Keep app running until you close the window
while app.is_running():
    sim.step()

app.close()
