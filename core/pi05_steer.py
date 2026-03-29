#!/usr/bin/env python

"""
Steerable PI05 Policy - PI05 with gradient-based trajectory guidance

Inherits from PI05Policy and only overrides the core sampling functions
to add steering capabilities.
"""

from collections import deque
from collections.abc import Callable
from typing import List, Optional, Union
import numpy as np
import torch
from torch import Tensor

from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from core.env_adapters import BaseEnvAdapter
from core.fkd_class import FKD
from utils.logging_utils import SteerLogger

log = SteerLogger("PI05PolicySteer")


class PI05PolicySteer(PI05Policy):
    """
    PI05 Policy with gradient-based trajectory steering.
    """
    
    name = "pi05_steer"
    
    def __init__(self, config):
        super().__init__(config)
        self._adapter = None
        self._postprocessor = None
        self._original_action_dim = self.config.output_features[ACTION].shape[0]
        self._cached_action_chunk = None
        # Reward-based adaptive guidance
        self._stage_init_reward = None
        self._last_normalized_reward = 0.0
        self._last_scale = 0.0
    
    def post_init(
        self,
        adapter: BaseEnvAdapter,
        postprocessor: Callable,
        sample_batch_size: int,
        policy_config: dict,
    ) -> None:
        """Initialize steering components after model loading."""
        self._adapter = adapter
        self._postprocessor = postprocessor
        self._sample_batch_size = sample_batch_size
        self._inference_steps = policy_config['num_inference_steps']
        self._action_chunk_horizon = policy_config['action_chunk_horizon']
    
    def reset(self):
        """Reset policy state for new episode."""
        self._cached_action_chunk = None
        self._stage_init_reward = None
        self._last_normalized_reward = 0.0
        self._last_scale = 0.0
    
    def reset_stage(self):
        """Reset stage-specific state (call when stage changes)."""
        self._stage_init_reward = None
    
    def get_normalized_reward(self) -> float:
        """Get last normalized reward (0=start, 1=reached target)."""
        return self._last_normalized_reward
    
    def get_last_scale(self) -> float:
        """Get last guidance scale used."""
        return self._last_scale

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        generate_new_chunk: bool = False,
        use_guidance: bool = False,
        keypoints: Optional[np.ndarray] = None,
        guidance_fns: Optional[dict] = None,  # Dict of {stage_num: guidance_fn}
        guide_scale: float = 1.0,
        start_ratio: Optional[float] = None,
        use_diversity: bool = True,
        diversity_scale: float = 1.0,
        MCMC_steps: int = 4,
        verbose: bool = False,
        use_fkd: bool = False,
        fkd_config: Optional[dict] = None,
        global_step: int = 0,
        current_stage: int = 1,
        sigmoid_k: float = 12.0,
        sigmoid_x0: float = 0.7,
        **kwargs  # Ignore other params for backward compatibility
    ) -> Tensor:

        self.eval()
        
        if ACTION in batch:
            batch.pop(ACTION)

        if generate_new_chunk:
            if use_guidance and guidance_fns:
                action_chunk = self._sample_actions_guided(
                    batch=batch,
                    keypoints=keypoints,
                    guidance_fn=guidance_fns,  # Just use the functions passed in
                    guide_scale=guide_scale,
                    start_ratio=start_ratio,
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
            
            self._cached_action_chunk = action_chunk
        else:
            action_chunk = self._cached_action_chunk
        
        return self._postprocessor(action_chunk) if self._postprocessor else action_chunk

    def _sample_actions_guided(
        self,
        batch: dict[str, Tensor],
        keypoints: Optional[np.ndarray] = None,
        guidance_fn: Optional[Callable] = None,
        guide_scale: float = 1.0,
        start_ratio: Optional[float] = None,
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
        Core guided sampling with reward-based adaptive guidance.
        
        Guidance strategy:
        1. time > start_time: Diversity guidance (spread trajectories)
        2. time <= start_time: Keypoint gradient guidance with adaptive scale
           - Scale adapts based on reward value (higher reward = softer guidance)
        """
        # Prepare inputs (use parent's preprocessing)
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        
        num_steps = self._inference_steps
        bsize = tokens.shape[0]
        device = tokens.device

        # Sample initial noise
        actions_shape = (bsize, self.model.config.chunk_size, self.model.config.max_action_dim)
        noise = self.model.sample_noise(actions_shape, device)

        # Embed prefix (reuse parent's logic)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        # Cache prefix key-values
        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Setup guidance
        start_time = start_ratio if start_ratio is None else 0.8

        keypoints_tensor = None
        if keypoints is not None:
            keypoints_tensor = torch.tensor(keypoints, device=device, dtype=torch.float32)
        
        use_keypoint_guidance = guidance_fn is not None and keypoints_tensor is not None

        # Initialize FKD if enabled
        fkd = self._init_fkd(fkd_config, bsize, num_steps, start_time, 
                             keypoints_tensor, guidance_fn, device) if use_fkd else None

        # Flow matching loop with guidance
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        step_idx = 0
        
        # Reward tracking for debugging
        reward_history = []
        
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # Get velocity from model
            v_t = self.model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                x_t=x_t,
                timestep=expanded_time,
            )

            # Apply guidance
            if use_diversity and time > start_time and bsize > 1:
                div_grad = self._compute_diversity_gradient(x_t[:, :, :self._original_action_dim], verbose=(verbose and step_idx == 0))
                if div_grad is not None:
                    v_t[:, :self._action_chunk_horizon, :3] += diversity_scale * div_grad[:, :self._action_chunk_horizon, :3]

            elif use_keypoint_guidance and time <= start_time:
                kp_grad, reward_value = self._compute_keypoint_gradient(
                    x_t[:, :, :self._original_action_dim], 
                    keypoints_tensor, 
                    guidance_fn, 
                    verbose=(verbose and step_idx == int(start_time * num_steps))
                )
                
                if kp_grad is not None:
                    # time_factor = (1 - time.item())
                    
                    # Reward-based adaptive scaling
                    # 暂时记录 reward，在 chunk 结束后用最后一个 step 的 reward 作为 baseline
                    # 计算归一化reward: 0(开始) → 1(到达目标)
                    if self._stage_init_reward is not None and self._stage_init_reward < -1e-6:
                        normalized_reward = 1 - (reward_value / self._stage_init_reward)
                        normalized_reward = max(0.0, min(1.2, normalized_reward))
                    else:
                        normalized_reward = 0.0  # No baseline yet, treat as start
                    
                    self._last_normalized_reward = normalized_reward
                    
                    # Track reward for debugging
                    reward_history.append((step_idx, reward_value, normalized_reward))
                    
                    # 引导强度：采用 Sigmoid 映射实现从"全力引导"到"优雅放手"的非线性过渡
                    # norm_reward: 0 (起点) -> 1 (终点)
                    # 当 norm_reward < sigmoid_x0 - margin 时，强度接近 1.0
                    # 当 norm_reward = sigmoid_x0 时，强度刚好为 0.5
                    # 使用配置传入的参数
                    guidance_strength = 1.0 / (1.0 + np.exp(sigmoid_k * (normalized_reward - sigmoid_x0)))
                    
                    scale = guide_scale * guidance_strength 
                    self._last_scale = scale
                    
                    if verbose and step_idx == int(start_time * num_steps):
                        init_str = f"{self._stage_init_reward:.6f}" if self._stage_init_reward is not None else "None"
                        log.info(
                            f"[Step {global_step}] Stage {current_stage} | "
                            f"reward={reward_value:.6f}, init={init_str}, "
                            f"norm_r={normalized_reward:.3f}, sig_strength={guidance_strength:.3f}, scale={scale:.2f}"
                        )
                    
                    v_t[:, :self._action_chunk_horizon, :3] -= scale * kp_grad[:, :self._action_chunk_horizon, :3]
                    
            # Euler step
            x_t = x_t + dt * v_t

            # FKD resampling
            if fkd is not None and time <= start_time:
                x_t, _ = fkd.resample(sampling_idx=step_idx, latents=x_t, x0_preds=x_t)

            time = time + dt
            step_idx += 1

        # Use the final reward of the first chunk as baseline for normalization
        if reward_history and self._stage_init_reward is None:
            final_reward = reward_history[-1][1]  # Last step's reward
            self._stage_init_reward = final_reward
            log.info(f"[Step {global_step}] Stage {current_stage} init_reward={final_reward:.6f} (from first chunk's final step)")

        # Unpad and return
        return x_t[:, :self._action_chunk_horizon, :self._original_action_dim]

    def _init_fkd(self, fkd_config, bsize, num_steps, start_time, 
                  keypoints_tensor, guidance_fn, device):
        """Initialize FKD for particle resampling."""
        if fkd_config is None or bsize <= 1 or keypoints_tensor is None:
            return None
        
        def reward_fn(x0_preds):
            traj = self._sample_to_trajectory_3d(x0_preds)[:, 1:self._action_chunk_horizon, :3]
            rewards = []
            for b in range(traj.shape[0]):
                try:
                    if isinstance(guidance_fn, list):
                        r = sum(fn(keypoints_tensor, traj[b:b+1]) for fn in guidance_fn)
                    else:
                        r = guidance_fn(keypoints_tensor, traj[b:b+1])
                    rewards.append(float(r.item()) if hasattr(r, 'item') else float(r))
                except Exception:
                    rewards.append(0.0)
            return torch.tensor(rewards, device=device, dtype=torch.float32)
        
        return FKD(
            potential_type=fkd_config.get('potential_type', 'max'),
            lmbda=fkd_config.get('lmbda', 10.0),
            num_particles=bsize,
            adaptive_resampling=fkd_config.get('adaptive_resampling', True),
            resample_frequency=fkd_config.get('resample_frequency', 5),
            resampling_t_start=int(start_time * num_steps),
            resampling_t_end=num_steps,
            timesteps=torch.linspace(1.0, 0.0, num_steps + 1, device=device),
            reward_fn=reward_fn,
            reward_min_value=float('-inf'),
            device=device,
        )

    def _sample_to_trajectory_3d(self, sample: Tensor) -> Tensor:
        """Convert action sample to 3D EE trajectory."""
        if self._adapter is None or self._postprocessor is None:
            raise RuntimeError("Call post_init() first")
        
        device, dtype = sample.device, sample.dtype
        actions = self._postprocessor(sample).to(device, dtype)
        batch_size = sample.shape[0]

        action_transition = {"action": actions}
        action_transition = self._adapter.env_postprocessor(action_transition)
        actions = action_transition["action"]

        if batch_size == 1:
            traj = self._adapter.delta_actions_to_ee_trajectory(
                actions.squeeze(0)[:self._action_chunk_horizon]
            )
            return traj.unsqueeze(0).to(device, dtype)
        
        trajs = [
            self._adapter.delta_actions_to_ee_trajectory(actions[b, :self._action_chunk_horizon])
            for b in range(batch_size)
        ]
        return torch.stack(trajs, dim=0).to(device, dtype)

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
            grad_norm: Original gradient norm (before normalization) - key for adaptive scaling
        """
        if not guidance_fn or self._adapter is None:
            return None, 0.0

        try:
            with torch.enable_grad():
                sample_grad = sample.detach().requires_grad_(True)
                traj = self._sample_to_trajectory_3d(sample_grad)[:, :self._action_chunk_horizon, :3]
                
                # Keep batch dimension for consistency - guidance functions expect (B, T, 3)
                # Don't squeeze even when batch_size=1
                traj_input = traj
                
                if isinstance(guidance_fn, list):
                    reward = sum(fn(keypoints_tensor, traj_input) for fn in guidance_fn)
                else:
                    reward = guidance_fn(keypoints_tensor, traj_input)
                
                if reward is None or not reward.requires_grad:
                    return None, 0.0
                
                if reward.dim() > 0:
                    reward = reward.sum()

                reward_value = reward.item() if hasattr(reward, 'item') else float(reward)
                
                grad = torch.autograd.grad(reward, sample_grad)[0]
                grad_norm_value = torch.norm(grad).item()
                normalized_grad = grad / (grad_norm_value + 1e-8) if grad_norm_value > 1e-8 else grad
                
                if verbose:
                    log.info(f"Reward: {reward_value:.4f}")
                
                return normalized_grad, reward_value
        except Exception as e:
            if verbose or not hasattr(self, '_guidance_error_logged'):
                log.warning(f"Guidance function error: {e}")
                self._guidance_error_logged = True
            return None, 0.0

    def _compute_diversity_gradient(self, sample: Tensor, verbose: bool = False) -> Optional[Tensor]:
        """Compute RBF diversity gradient."""
        batch_size = sample.shape[0]
        if batch_size < 2 or self._adapter is None:
            return None

        with torch.enable_grad():
            sample_grad = sample.detach().requires_grad_(True)
            traj = self._sample_to_trajectory_3d(sample_grad)[:, :self._action_chunk_horizon, :3]
            traj_flat = traj.reshape(batch_size, -1)

            # Pairwise distance potential
            diff = traj_flat.unsqueeze(1) - traj_flat.unsqueeze(0)
            dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-6)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=sample.device)
            potential = (1.0 / (dist + 1e-6) * mask.float()).sum()

            grad = torch.autograd.grad(potential, sample_grad)[0]
            grad_norm = torch.norm(grad)
            # Normalize to unit norm for stability
            normalized_grad = grad / (grad_norm + 1e-8) if grad_norm > 1e-8 else grad

            return normalized_grad
