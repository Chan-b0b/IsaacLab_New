from __future__ import annotations

from typing import Callable, Tuple, Any

import time
import torch
import statistics
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner


ExpertProvider = Callable[[TensorDict, int, int], Tuple[torch.Tensor, torch.Tensor]]
# Signature: (obs, step_global_counter, tot_timesteps) -> (indices, actions)


class ExpertOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner subclass that can inject expert actions during rollout.

    Provide an `expert_provider` callable to `learn()` which returns a tuple
    `(indices, actions)` to overwrite the agent actions for those env indices
    for the current step. `indices` can be a 1D LongTensor of environment indices
    or a boolean mask of shape `(num_envs,)`. `actions` must match the action
    shape for the selected indices.
    """

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
        expert_provider: ExpertProvider | None = None,
        expert_reset: Callable[[torch.Tensor], None] | None = None,
    ) -> None:
        # Mostly copy parent's learn logic but allow action injection per step
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        from collections import deque

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # Track rewards and episode lengths for steps where expert was NOT used
        nonexpert_rewbuffer = deque(maxlen=100)
        nonexpert_lenbuffer = deque(maxlen=100)
        cur_nonexpert_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_nonexpert_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if getattr(self.alg, "rnd", None):
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            # Rollout timing
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for step_idx in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)

                    # Expert injection: ask provider for indices and expert actions
                    # default: no env uses expert this step
                    expert_mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=actions.device)
                    if expert_provider is not None:
                        try:
                            expert_idx, expert_actions = expert_provider(obs, step_idx, self.tot_timesteps)
                            if expert_idx is not None:
                                # normalize index format to boolean mask on actions.device
                                if isinstance(expert_idx, torch.BoolTensor) or (
                                    isinstance(expert_idx, torch.Tensor) and expert_idx.dtype == torch.bool
                                ):
                                    mask = expert_idx.to(device=actions.device)
                                    expert_mask = mask
                                    if mask.any():
                                        actions[mask] = expert_actions.to(actions.device)
                                else:
                                    idx = expert_idx.long().to(device=actions.device)
                                    if idx.numel() > 0:
                                        expert_mask[idx] = True
                                        actions[idx] = expert_actions.to(actions.device)
                        except Exception:
                            # swallow provider errors to avoid breaking rollout
                            expert_mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=actions.device)

                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    # Reset expert state machine for finished envs if callback provided
                    if expert_reset is not None:
                        try:
                            finished = dones.nonzero(as_tuple=False).squeeze(-1)
                            if finished.numel() > 0:
                                expert_reset(finished)
                        except Exception:
                            pass

                    intrinsic_rewards = getattr(self.alg, "intrinsic_rewards", None)

                    # Logging bookkeeping (minimal, follow base behavior)
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                        # accumulate full rewards (existing behaviour)
                        if getattr(self.alg, "rnd", None):
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards

                        # accumulate non-expert-only rewards: those envs where expert was NOT used
                        # ensure expert_mask is on the runner device for indexing
                        try:
                            mask_runner = expert_mask.to(self.device)
                        except Exception:
                            mask_runner = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
                        nonexpert_inds = (~mask_runner).nonzero(as_tuple=False).squeeze(-1)
                        if nonexpert_inds.numel() > 0:
                            cur_nonexpert_reward_sum[nonexpert_inds] += rewards[nonexpert_inds]
                            cur_nonexpert_episode_length[nonexpert_inds] += 1
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        # finalize and log non-expert episodic returns/lengths for finished envs
                        if new_ids.numel() > 0:
                            nonexpert_vals = cur_nonexpert_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                            nonexpert_lens = cur_nonexpert_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                            nonexpert_rewbuffer.extend(nonexpert_vals)
                            nonexpert_lenbuffer.extend(nonexpert_lens)
                            cur_nonexpert_reward_sum[new_ids] = 0
                            cur_nonexpert_episode_length[new_ids] = 0
                        if getattr(self.alg, "rnd", None):
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                # Compute returns
                self.alg.compute_returns(obs)

            # compute collection time
            collection_time = time.time() - start

            # Update policy (measure learn time)
            start_learn = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - start_learn

            # Log and save using parent's mechanisms
            self.current_learning_iteration = it
            if self.log_dir is not None and not getattr(self, "disable_logs", False):
                locs = {
                    "collection_time": collection_time,
                    "learn_time": learn_time,
                    "it": it,
                    "start_it": start_iter,
                    "tot_iter": tot_iter,
                    "num_learning_iterations": num_learning_iterations,
                    "loss_dict": loss_dict,
                    "ep_infos": ep_infos,
                    "rewbuffer": rewbuffer,
                    "lenbuffer": lenbuffer,
                    "erewbuffer": erewbuffer if getattr(self, "alg", None) and getattr(self.alg, "rnd", None) else [],
                    "irewbuffer": irewbuffer if getattr(self, "alg", None) and getattr(self.alg, "rnd", None) else [],
                    # Non-expert aggregated episode stats
                    "nonexpert_rewbuffer": nonexpert_rewbuffer,
                    "nonexpert_lenbuffer": nonexpert_lenbuffer,
                    "start_iter": start_iter,
                }
                self.log(locs)
                if it % self.save_interval == 0:
                    self.save(self.log_dir + "/last.pt")

            ep_infos.clear()

        if self.log_dir is not None and not getattr(self, "disable_logs", False):
            self.save(self.log_dir + "/last.pt")

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Log episode information
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # Handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # Log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # Log noise std
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # Log performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Log training
        if len(locs["rewbuffer"]) > 0:
            # Separate logging for intrinsic and extrinsic rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # Everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            # Print losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
            # Print rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(locs["erewbuffer"]):.2f}\n"""
                    f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(locs["irewbuffer"]):.2f}\n"""
                )
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            # Print episode information
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""

            if "nonexpert_rewbuffer" in locs and len(locs.get("nonexpert_rewbuffer", [])) > 0:
                self.writer.add_scalar(
                    "Train/nonexpert_mean_reward", statistics.mean(locs["nonexpert_rewbuffer"]), locs["it"]
                )

                if self.logger_type != "wandb":
                    self.writer.add_scalar(
                        "Train/nonexpert_mean_reward/time", statistics.mean(locs["nonexpert_rewbuffer"]), self.tot_time
                    )

        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
                        # Non-expert stats (episodes where no expert action was used)
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        self.tot_time
                        / (locs["it"] - locs["start_iter"] + 1)
                        * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
                    ),
                )
            }\n"""
        )
        # Print non-expert episode stats when available
        if "nonexpert_rewbuffer" in locs and len(locs.get("nonexpert_rewbuffer", [])) > 0:
            log_string += (
                f"""{"Mean non-expert reward:":>{pad}} {statistics.mean(locs["nonexpert_rewbuffer"]):.2f}\n"""
            )
            log_string += (
                f"""{"Mean non-expert episode length:":>{pad}} {statistics.mean(locs["nonexpert_lenbuffer"]):.2f}\n"""
            )
        print(log_string)