#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Steerable Diffusion Policy

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

from collections import deque
from collections.abc import Callable
from typing import List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from core.env_adapters import BaseEnvAdapter
from core.fkd_class import FKD, PotentialType
from utils.logging_utils import SteerLogger
log = SteerLogger("DiffusionPolicySteer")

class DiffusionPolicySteer(DiffusionPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://huggingface.co/papers/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    name = "diffusion_steer"

    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
        self._adapter = None
        self._postprocessor = None

    def post_init(
        self,
        adapter: BaseEnvAdapter,
        postprocessor: Callable,
        sample_batch_size: int = 1,
        policy_config: dict = None,
    ) -> None:
        self._adapter = adapter
        self._postprocessor = postprocessor
        self._sample_batch_size = sample_batch_size
        if policy_config is not None:
            action_horizon = policy_config.get('action_horizon', 8)
        else:
            action_horizon = 8
        self.config.n_action_steps = action_horizon
        self._action_chunk_horizon = action_horizon
        
        # Initialize reward tracking (for compatibility with PI05PolicySteer)
        self._last_normalized_reward = 0.0
        self._last_scale = 0.0
        self._stage_init_reward = None
    
    def reset(self):
        """Reset policy state for new episode."""
        # Call parent reset to initialize _queues
        super().reset()
        self._cached_action_chunk = None
        self._stage_init_reward = None
        self._last_normalized_reward = 0.0
        self._last_scale = 0.0
    
    # Alias for compatibility
    reset_episode = reset
    
    def reset_stage(self):
        """Reset stage-specific state (call when stage changes)."""
        self._stage_init_reward = None
    
    def get_normalized_reward(self) -> float:
        """Get last normalized reward (0=start, 1=reached target)."""
        return self._last_normalized_reward
    
    def get_last_scale(self) -> float:
        """Get last guidance scale used."""
        return self._last_scale
        # self.._kfd = FKD(
        #         potential_type=PotentialType.MAX,
        #         lmbda=10.0,
        #         num_particles=batch_size,
        #         adaptive_resampling=False,
        #         resample_frequency=1,
        #         resampling_t_start=int(start_step),
        #         resampling_t_end=int(timesteps[-1].item()),
        #         timesteps=timesteps,
        #         reward_fn=_reward_fn,
        #         reward_min_value=-(float('inf')),
        #         device=self.device,
        #     )

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        generate_new_chunk: bool = False,
        use_guidance: bool = False,
        keypoints: Optional[np.ndarray] = None,
        guidance_fns: Optional[List[Callable]] = None,  # Changed from guidance_fn to guidance_fns
        guide_scale: float = 1.0,
        sigmoid_k: float = 12.0,
        sigmoid_x0: float = 0.7,
        start_ratio: Optional[float] = None,
        use_diversity: bool = True,
        diversity_scale: float = 1.0,
        MCMC_steps: int = 4,
        verbose: bool = False,
        use_fkd: bool = False,
        fkd_config: Optional[dict] = None,
        global_step: int = 0,
        current_stage: int = 1,
    ) -> Tensor:

        if batch is None:
            raise ValueError("batch cannot be None")
            
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features and len(self.config.image_features) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if generate_new_chunk:
            if use_guidance:
                # Calculate start_step from start_ratio if provided
                try:
                    num_inference_steps = self.diffusion.num_inference_steps
                    # Ensure num_inference_steps is a valid integer
                    if not isinstance(num_inference_steps, (int, float)):
                        log.warning(f"num_inference_steps is {type(num_inference_steps)}: {num_inference_steps}, using default 100")
                        num_inference_steps = 100
                except Exception as e:
                    log.warning(f"Failed to get num_inference_steps: {e}, using default 100")
                    num_inference_steps = 100
                
                # Ensure start_ratio is valid
                if start_ratio is not None and isinstance(start_ratio, (int, float)):
                    start_step = int(float(num_inference_steps) * float(start_ratio))
                else:
                    start_step = int(float(num_inference_steps) * 0.7)  # Default to 70%
                
                action_chunk = self._predict_action_chunk_guided (
                    batch=batch,
                    keypoints=keypoints,
                    guidance_fn=guidance_fns,  # Use guidance_fns (the parameter name)
                    guide_scale=guide_scale,
                    start_step=start_step,
                    use_diversity=use_diversity,
                    diversity_scale=diversity_scale,
                    verbose=verbose,
                    use_fkd=use_fkd,
                    fkd_config=fkd_config,
                    global_step=global_step,
                    current_stage=current_stage,
                    sigmoid_k=sigmoid_k,
                    sigmoid_x0=sigmoid_x0,
                )
            else:
                action_chunk = self.predict_action_chunk(batch)
            # Cache the action chunk for subsequent calls
            self._cached_action_chunk = action_chunk
        else:
            # Return cached action chunk
            action_chunk = self._cached_action_chunk

        return self._postprocessor(action_chunk)

    def _predict_action_chunk_guided(
        self,
        batch: dict[str, Tensor],
        keypoints: Optional[np.ndarray] = None,
        guidance_fn: Optional[Callable] = None,
        guide_scale: float = 1.0,
        start_step: Optional[int] = None,
        use_diversity: bool = True,
        diversity_scale: float = 1.0,
        verbose: bool = False,
        use_fkd: bool = False,
        fkd_config: Optional[dict] = None,
        global_step: int = 0,
        current_stage: int = 1,
        sigmoid_k: float = 12.0,
        sigmoid_x0: float = 0.7,
    ) -> Tensor:

        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}

        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self.diffusion._prepare_global_conditioning(batch)

        actions = self._guided_conditional_sample(
            batch_size=batch_size,
            global_cond=global_cond,
            keypoints=keypoints,
            guidance_fn=guidance_fn,
            guide_scale=guide_scale,
            start_step=start_step,
            use_diversity=use_diversity,
            diversity_scale=diversity_scale,
            verbose=verbose,
            use_fkd=use_fkd,
            fkd_config=fkd_config,
            global_step=global_step,
            current_stage=current_stage,
            sigmoid_k=sigmoid_k,
            sigmoid_x0=sigmoid_x0,
        )

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    def _guided_conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor,
        generator: Optional[torch.Generator] = None,
        keypoints: Optional[np.ndarray] = None,
        guidance_fn: Optional[Callable] = None,
        guide_scale: float = 1.0,
        start_step: Optional[int] = None,
        use_diversity: bool = True,
        diversity_scale: float = 1.0,
        verbose: bool = False,
        use_fkd: bool = False,
        fkd_config: Optional[dict] = None,
        global_step: int = 0,
        current_stage: int = 1,
        sigmoid_k: float = 12.0,
        sigmoid_x0: float = 0.7,
    ) -> Tensor:
        """
        Core logic of guided conditional sampling.

        Sampling process:
        1. t > start_step: Use RBF diversity guidance (make trajectories diverse)
        2. t <= start_step: Use keypoint gradient guidance (guide trajectories towards the target)
        3. If use_fkd: Apply FKD particle resampling at specified intervals
        """
        device = get_device_from_parameters(self.diffusion)
        dtype = get_dtype_from_parameters(self.diffusion)

        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.diffusion.noise_scheduler.set_timesteps(self.diffusion.num_inference_steps)
        timesteps = self.diffusion.noise_scheduler.timesteps

        if start_step is None:
            start_step = int(timesteps[len(timesteps) // 3].item())

        use_keypoint_guidance = (
            guidance_fn is not None
            and keypoints is not None
        )

        keypoints_tensor = None
        if keypoints is not None:
            keypoints_tensor = torch.tensor(keypoints, device=device, dtype=dtype)

        # Initialize FKD if enabled
        fkd = None
        if use_fkd and fkd_config is not None and batch_size > 1:
            # Create reward function for FKD using guidance_fn
            def fkd_reward_fn(x0_preds: Tensor) -> Tensor:
                """Compute reward for FKD resampling using the guidance function."""
                trajectories_3d = self._sample_to_trajectory_3d(x0_preds)
                # Slice to match gradient guidance: skip first point, use n_action_steps, only xyz
                trajectories_3d = trajectories_3d[:, 1:self.config.n_action_steps, :3]
                rewards = []
                for b in range(trajectories_3d.shape[0]):
                    # Keep batch dim: guidance expects (B, T, 3) not (T, 3)
                    traj = trajectories_3d[b:b+1]  # (1, T, 3)
                    if guidance_fn is not None:
                        try:
                            if isinstance(guidance_fn, list):
                                # Order: (keypoints, trajectory) - same as _compute_keypoint_gradient
                                reward = sum(fn(keypoints_tensor, traj) for fn in guidance_fn)
                            else:
                                reward = guidance_fn(keypoints_tensor, traj)
                            # Handle tensor rewards
                            if hasattr(reward, 'item'):
                                reward = reward.item()
                            rewards.append(float(reward))
                        except Exception as e:
                            log.warning(f"[FKD] Reward computation failed for batch {b}: {e}")
                            rewards.append(0.0)
                    else:
                        rewards.append(0.0)
                return torch.tensor(rewards, device=device, dtype=dtype)

            fkd = FKD(
                potential_type=fkd_config.get('potential_type', 'max'),
                lmbda=fkd_config.get('lmbda', 10.0),
                num_particles=batch_size,
                adaptive_resampling=fkd_config.get('adaptive_resampling', True),
                resample_frequency=fkd_config.get('resample_frequency', 5),
                resampling_t_start=int(start_step),
                resampling_t_end=int(timesteps[-1].item()),
                timesteps=timesteps,
                reward_fn=fkd_reward_fn,
                reward_min_value=float('-inf'),
                device=device,
            )
            if verbose:
                log.info(f"[FKD] Initialized with lmbda={fkd_config.get('lmbda', 10.0)}, "
                        f"potential={fkd_config.get('potential_type', 'max')}, "
                        f"resample_freq={fkd_config.get('resample_frequency', 5)}")

        # Reward tracking for debugging
        reward_history = []

        for i, t in enumerate(timesteps):

            model_output = self.diffusion.unet(
                sample,
                torch.full((batch_size,), t, dtype=torch.long, device=device),
                global_cond=global_cond,
            )

            if use_diversity and t > start_step and batch_size > 1:
                div_grad = self._compute_diversity_gradient(sample, verbose=(verbose and i == 0))
                if div_grad is not None:
                    model_output[:, :, :3] += diversity_scale * div_grad[:, :, :3]

            elif use_keypoint_guidance and t <= start_step:

                kp_grad, reward_value = self._compute_keypoint_gradient(
                    sample, keypoints_tensor, guidance_fn, verbose=(verbose and i == int(len(timesteps) * 0.8))
                )
                
                if kp_grad is not None:
                    # Reward-based adaptive scaling (same as PI05PolicySteer)
                    if self._stage_init_reward is not None and self._stage_init_reward < -1e-6:
                        normalized_reward = 1 - (reward_value / self._stage_init_reward)
                        normalized_reward = max(0.0, min(1.2, normalized_reward))
                    else:
                        normalized_reward = 0.0  # No baseline yet, treat as start
                    
                    self._last_normalized_reward = normalized_reward
                    reward_history.append((i, reward_value, normalized_reward))
                    
                    # Sigmoid-based guidance strength
                    guidance_strength = 1.0 / (1.0 + np.exp(sigmoid_k * (normalized_reward - sigmoid_x0)))
                    
                    alpha_t = self.diffusion.noise_scheduler.alphas_cumprod[t]
                    scale = guide_scale * guidance_strength * (1 - alpha_t).sqrt()
                    self._last_scale = scale
                    
                    if verbose and i == int(len(timesteps) * 0.8):
                        init_str = f"{self._stage_init_reward:.6f}" if self._stage_init_reward is not None else "None"
                        log.info(
                            f"[Step {global_step}] Stage {current_stage} | "
                            f"reward={reward_value:.6f}, init={init_str}, "
                            f"norm_r={normalized_reward:.3f}, sig_strength={guidance_strength:.3f}, scale={scale:.2f}"
                        )
                    
                    model_output[:, :self.config.n_action_steps, :3] -= scale * kp_grad[:, :self.config.n_action_steps, :3]

            sample = self.diffusion.noise_scheduler.step(
                model_output, t, sample, generator=generator
            ).prev_sample

            # FKD resampling step
            if fkd is not None and t <= start_step:
                # Use the current sample as x0 prediction for FKD
                # In practice, we use the denoised sample estimate
                sample, _ = fkd.resample(
                    sampling_idx=int(t.item()),
                    latents=sample,
                    x0_preds=sample,  # Use current sample as x0 estimate
                )
                if verbose and fkd.reached_terminal:
                    log.info(f"[FKD] Terminal reached at t={t}, particles sorted by reward")

        # Use the final reward of the first chunk as baseline for normalization
        if reward_history and self._stage_init_reward is None:
            final_reward = reward_history[-1][1]  # Last step's reward
            self._stage_init_reward = final_reward
            log.info(f"[Step {global_step}] Stage {current_stage} init_reward={final_reward:.6f} (from first chunk's final step)")

        return sample

    def _sample_to_trajectory_3d(self, sample: Tensor) -> Tensor:

        batch_size = sample.shape[0]
        device, dtype = sample.device, sample.dtype

        actions = self._postprocessor(sample).to(device, dtype)

        if batch_size == 1:
            action_seq = actions.squeeze(0)[:self.config.n_action_steps, :]
            traj = self._adapter.delta_actions_to_ee_trajectory(action_seq).to(device, dtype)
            return traj.unsqueeze(0)
        else:
            trajs = []
            for b in range(batch_size):
                action_seq = actions[b, :self.config.n_action_steps, :]
                traj = self._adapter.delta_actions_to_ee_trajectory(action_seq).to(device, dtype)
                trajs.append(traj)
            return torch.stack(trajs, dim=0)

    def _compute_keypoint_gradient(
        self,
        sample: Tensor,
        keypoints_tensor: Tensor,
        guidance_fn: Union[Callable, List[Callable]],
        verbose: bool = False,
    ) -> tuple[Optional[Tensor], float]:
        """
        Compute gradient for keypoint guidance with statistics for adaptive scaling.
        
        Returns:
            normalized_grad: Normalized gradient (or None)
            reward_value: Raw reward value from guidance function
        """
        if not guidance_fn or self._adapter is None:
            return None, 0.0

        try:
            with torch.enable_grad():
                sample_grad = sample.detach().requires_grad_(True)

                trajectories_3d = self._sample_to_trajectory_3d(sample_grad)
                trajectories_3d = trajectories_3d[:, :self.config.n_action_steps, :3]

                # Keep batch dimension for consistency - guidance functions expect (B, T, 3)
                traj_input = trajectories_3d

                # guidance_fn can be a list of functions or a single function
                if isinstance(guidance_fn, list):
                    reward = sum(fn(keypoints_tensor, traj_input) for fn in guidance_fn)
                else:
                    reward = guidance_fn(keypoints_tensor, traj_input)

                # Handle case where reward is already a float/int
                if isinstance(reward, (int, float)):
                    return None, float(reward)
                
                if reward is None:
                    return None, 0.0
                    
                if not hasattr(reward, 'requires_grad') or not reward.requires_grad:
                    # Return the value but no gradient
                    reward_value = reward.item() if hasattr(reward, 'item') else float(reward)
                    return None, reward_value

                # Sum to scalar if batched
                if hasattr(reward, 'dim') and reward.dim() > 0:
                    reward = reward.sum()

                reward_value = reward.item() if hasattr(reward, 'item') else float(reward)

                gradient = torch.autograd.grad(
                    reward, sample_grad, create_graph=False, retain_graph=False
                )[0]

                grad_norm = torch.norm(gradient).item()
                normalized_grad = gradient / (grad_norm + 1e-8) if grad_norm > 1e-8 else gradient

                if verbose:
                    log.info(f"Reward: {reward_value:.4f}")

                return normalized_grad, reward_value
        except Exception as e:
            if verbose or not hasattr(self, '_guidance_error_logged'):
                import traceback
                log.warning(f"Guidance function error: {e}")
                log.debug(f"Traceback: {traceback.format_exc()}")
                self._guidance_error_logged = True
            return None, 0.0


    def _compute_diversity_gradient(
        self,
        sample: Tensor,
        verbose: bool = False,
    ) -> Optional[Tensor]:
        """
        Compute RBF diversity gradient (make trajectories disperse).
        """
        batch_size = sample.shape[0]
        if batch_size < 2 or self._adapter is None:
            return None

        # Need to enable grad since we're inside a no_grad context from select_action
        with torch.enable_grad():
            sample_grad = sample.detach().requires_grad_(True)

            trajectories_3d = self._sample_to_trajectory_3d(sample_grad)
            trajectories_pos = trajectories_3d[:, 1:, :3]  # (B, T, 3)

            traj_flat = trajectories_pos.reshape(batch_size, -1)  # (B, D)

            traj_i = traj_flat.unsqueeze(1)  # (B, 1, D)
            traj_j = traj_flat.unsqueeze(0)  # (1, B, D)
            squared_dist = torch.sum((traj_i - traj_j) ** 2, dim=2)  # (B, B)

            mask = ~torch.eye(batch_size, dtype=torch.bool, device=sample.device)

            eps = 1e-6
            dist = torch.sqrt(squared_dist + eps)
            inv_dist = (1.0 / (dist + eps)) * mask.float()
            total_potential = torch.sum(inv_dist)

            gradient = torch.autograd.grad(
                total_potential, sample_grad, create_graph=False, retain_graph=False
            )[0]

            if verbose:
                log.info(f"Diversity gradient norm: {torch.norm(gradient).item():.6f}")

        return gradient
