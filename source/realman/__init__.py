import gymnasium as gym

from . import agents


gym.register(
    id="Realman-RM75-PickPlace-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "realman.pick_place_env_cfg:RealmanPickPlaceEnvCfg",
        "play_env_cfg_entry_point": "realman.pick_place_env_cfg:RealmanPickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RealmanPickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Realman-RM75-PickPlace-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "realman.pick_place_env_cfg:RealmanPickPlaceEnvCfg",
        "play_env_cfg_entry_point": "realman.pick_place_env_cfg:RealmanPickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RealmanPickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)
