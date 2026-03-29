"""
Feynman-Kac Diffusion (FKD) steering mechanism for robot / trajectory diffusion.

This module is a simplified移植 of the FKD particle algorithm in the Fk-Diffusion-Steering repository,

The core idea is:
- Maintain num_particles parallel sampling "particle trajectories" (usually the batch dimension)
- On several time steps, decode these particles (e.g., convert to 3D trajectory) and score them using reward_fn
- Use the Feynman-Kac corresponding potential to calculate weights w_i = exp(lambda * potential_i)
- Resample the particles according to the weights (duplicate good particles, discard bad particles), thus steering the sampling process towards high-reward samples during inference.

This implementation is a non-gradient particle filter / SMC, used to配合 guided_diffusion_sampler 中已有的
reward functions (e.g., trajectory reward based on keypoints); FKD does not rely on the differentiability of the reward.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

TensorOrList = Union[torch.Tensor, List]


def _list_tensor_index(x: TensorOrList, indices: torch.Tensor) -> TensorOrList:
    """Index on list / Tensor using one-dimensional indices."""
    if isinstance(x, list):
        return [x[int(i)] for i in indices]
    return x[indices]


class PotentialType(Enum):
    """Different Feynman-Kac potential types。

    - DIFF: Only use the current reward and the difference between the current reward and the history reward r_t - r_{t-1}
    - MAX: Maintain the history maximum reward max_t r_t
    - ADD: Accumulate the reward sum \sum_t r_t
    - RT : Only use the current reward r_t (no memory)
    """

    DIFF = "diff"
    MAX = "max"
    ADD = "add"
    RT = "rt"


class FKD:
    """Feynman-Kac Diffusion particle resampler.

    Usage:
    - Build a FKD instance before sampling
    - Call .resample(sampling_idx, latents, x0_preds) at each diffusion time step

    Parameters
    ----
    potential_type:
        Potential function type, see PotentialType.
    lmbda:
        Weight scaling factor λ,the larger the λ, the more aggressive the steering.
    num_particles:
        Number of particles (usually equal to batch_size).
    adaptive_resampling:
        Whether to resample according to the effective sample number ESS.
    resample_frequency:
        Resampling step length (unit: time step index), e.g., 1 means resample at every step.
    resampling_t_start, resampling_t_end:
        Allowable resampling time step interval (inclusive endpoints), closed interval.
    time_steps:
        Total sampling time steps (used to determine the end).
    reward_fn:
        Calculation reward function, signature reward_fn(population_samples) -> Tensor/ list,
        where population_samples = latent_to_decode_fn(x0_preds).
    reward_min_value:
        Initial minimum value of population_rs (useful for MAX potential type).
    latent_to_decode_fn:
        Function to convert x0 prediction tensor to "scorable space", e.g.:
        - Return the action sequence itself (identity)
        - Decode the action sequence to a 3D trajectory
    device:
        Device on which computations will be performed.
    """

    def __init__(
        self,
        *,
        potential_type: Union[PotentialType, str],
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        timesteps: Union[torch.Tensor, List[int]],
        reward_fn: Callable[[TensorOrList], Union[torch.Tensor, List[float]]],
        reward_min_value: float = 0.0,
        device: torch.device | str = torch.device("cuda"),
    ) -> None:
        self.potential_type = (
            potential_type
            if isinstance(potential_type, PotentialType)
            else PotentialType(potential_type)
        )
        self.lmbda = float(lmbda)
        self.num_particles = int(num_particles)
        self.adaptive_resampling = bool(adaptive_resampling)
        self.resample_frequency = int(resample_frequency)
        self.resampling_t_start = int(resampling_t_start)
        self.resampling_t_end = int(resampling_t_end)

        self.reward_fn = reward_fn

        self.device = torch.device(device)

        # population_rs: "history reward statistics" of the current particle population (slightly different meanings depending on potential_type)
        self.population_rs = (
            torch.ones(self.num_particles, device=self.device) * reward_min_value
        )
        # Product of potentials, used to correct MAX / ADD / RT at the end
        self.product_of_potentials = torch.ones(self.num_particles).to(self.device)

        # Incremental check
        self._last_idx_sampled = -1
        self._reached_terminal_sample = False

        # Process discrete time steps of the scheduler (e.g., [999, ..., 0])
        if isinstance(timesteps, torch.Tensor):
            self.timesteps = timesteps.detach().cpu().long().tolist()
        else:
            self.timesteps = [int(t) for t in timesteps]
        self.time_steps = len(self.timesteps)

        # Build mapping from timestep to index, for checking order and determining the end
        self.t_to_index = {int(t): i for i, t in enumerate(self.timesteps)}

        # Pre-construct "resampling time set" represented by timestep t
        # Idea: find the index range corresponding to start/end, then take a subsequence according to resample_frequency,
        # then map back to the specific timestep values.
        if self.timesteps:
            # Default from the position of given start/end; if not aligned, backtrack to the ends
            start_idx = self.t_to_index.get(self.resampling_t_start, 0)
            end_idx = self.t_to_index.get(self.resampling_t_end, self.time_steps - 1)
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

            step = max(self.resample_frequency, 1)
            idxs = list(range(start_idx, end_idx + 1, step))
            resampling_ts = [self.timesteps[i] for i in idxs]

            # Ensure the last timestep is definitely included (end)
            terminal_t = self.timesteps[-1]
            if terminal_t not in resampling_ts:
                resampling_ts.append(terminal_t)

            self.resampling_interval = np.array(resampling_ts, dtype=int)
        else:
            self.resampling_interval = np.array([], dtype=int)

    @property
    def reached_terminal(self) -> bool:
        """Whether the last time step has been reached (only as a state query interface)."""
        return self._reached_terminal_sample

    def _compute_reward(
        self, x0_preds: TensorOrList
    ) -> Tuple[torch.Tensor, TensorOrList]:
        """Decode and calculate reward.

        Returns
        ----
        rs_candidates:
            Tensor of shape (num_particles,), representing the current reward of each particle.
        population_samples:
            Decoded samples (optional for upper layer usage).
        """
        # 解码（如动作 -> 3D 轨迹）
        population_samples = x0_preds
        rs_candidates = self.reward_fn(population_samples)

        if isinstance(rs_candidates, list):
            rs_candidates = torch.tensor(rs_candidates, device=self.device)
        else:
            rs_candidates = rs_candidates.to(self.device)

        if rs_candidates.ndim != 1:
            rs_candidates = rs_candidates.view(-1)

        return rs_candidates, population_samples

    def resample(
        self,
        *,
        sampling_idx: int,
        latents: TensorOrList,
        x0_preds: TensorOrList,
    ) -> Tuple[TensorOrList, Optional[TensorOrList]]:
        """Resample the particle population at the given time step (optional).

        Parameters
        ----
        sampling_idx:
            Current scheduler time step t (e.g., 999, 997, ..., 0).
        latents:
            Current noisy state x_t (e.g., action noise trajectory), typically of shape (B, T, D).
        x0_preds:
            "Predicted clean sample" x_0 corresponding to x_t, used for reward calculation;
            if no separate x0 is available, the latents can be passed in directly.

        Returns
        ----
        resampled_xt, resampled_x0
            Resampled x_t (used for continued diffusion) and corresponding x_0 prediction (used for reward calculation / final result).
        """
        # For clarity, internally use xt / x0 to represent the current particle state
        xt = latents
        x0 = x0_preds

        # sampling_idx represents the "scheduler time step t" in the current implementation, e.g., 999->0.
        sampling_t = int(sampling_idx)

        # Map timestep to its index in scheduler.timesteps, to check order and determine the end
        current_idx = self.t_to_index.get(sampling_t, None)
        if current_idx is None:
            # Unknown timestep, skip FKD logic
            return latents, None

        if current_idx <= self._last_idx_sampled:
            raise ValueError(
                f"timestep index {current_idx} must be greater than {self._last_idx_sampled},"
                "please ensure the order of scheduler.timesteps."
            )
        self._last_idx_sampled = current_idx

        # End: scheduler last time step
        at_terminal_sample = current_idx == (self.time_steps - 1)
        self._reached_terminal_sample = at_terminal_sample

        # Not at resampling time point, return directly (here using the set of "specific timestep values")
        if sampling_t not in self.resampling_interval:
            return xt, None

        # Calculate reward
        rs_candidates, population_samples = self._compute_reward(x0)

        # Calculate potential & weights
        if self.potential_type == PotentialType.MAX:
            rs_candidates = torch.max(rs_candidates, self.population_rs)
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.ADD:
            rs_candidates = rs_candidates + self.population_rs
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.DIFF:
            diffs = rs_candidates - self.population_rs
            w = torch.exp(self.lmbda * diffs)
        elif self.potential_type == PotentialType.RT:
            w = torch.exp(self.lmbda * rs_candidates)
        else:
            raise ValueError(f"Unknown potential type: {self.potential_type}")

        # At the end, perform Feynman–Kac correction for MAX / ADD / RT
        if at_terminal_sample and self.potential_type in (
            PotentialType.MAX,
            PotentialType.ADD,
            PotentialType.RT,
        ):
            w = torch.exp(self.lmbda * rs_candidates) / self.product_of_potentials

        # Numerical stability: clip and handle NaN
        w = torch.clamp(w, 0, 1e10)
        w[torch.isnan(w)] = 0.0

        # If all 0, fall back to uniform sampling
        if float(w.sum()) == 0.0:
            w = torch.ones_like(w)
        

        # Decide whether to resample based on ESS (if adaptive resampling is enabled)
        do_resample = True
        if self.adaptive_resampling or at_terminal_sample:
            normalized_w = w / w.sum()
            ess = 1.0 / (normalized_w.pow(2).sum() + 1e-12)
            if ess >= 0.5 * self.num_particles and not at_terminal_sample:
                # Effective sample size is sufficient, no forced resampling
                do_resample = False

        if do_resample:
            indices = torch.multinomial(
                w, num_samples=self.num_particles, replacement=True
            )
            resampled_xt = _list_tensor_index(xt, indices)
            self.population_rs = rs_candidates[indices]
            resampled_x0 = _list_tensor_index(population_samples, indices)
            # Update accumulated potential (only accumulated on the resampling path)
            self.product_of_potentials = (
                self.product_of_potentials[indices] * w[indices]
            )
        else:
            # No resampling, only update history reward
            resampled_xt = xt
            resampled_x0 = population_samples
            self.population_rs = rs_candidates

        # If already reached the last time step, sort particles by reward from largest to smallest,
        # ensure index 0 corresponds to the sample with the highest reward, for downstream direct sample[0].
        if at_terminal_sample:
            sort_indices = torch.argsort(self.population_rs, descending=True)
            resampled_xt = _list_tensor_index(resampled_xt, sort_indices)
            resampled_x0 = _list_tensor_index(resampled_x0, sort_indices)
            self.population_rs = self.population_rs[sort_indices]
            self.product_of_potentials = self.product_of_potentials[sort_indices]

        return resampled_xt, resampled_x0


__all__ = ["FKD", "PotentialType"]


