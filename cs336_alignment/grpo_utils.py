from typing import Callable, Literal

import torch

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normaliz_by_std: bool,
):
    assert len(rollout_responses) == len(repeated_ground_truths)

    raw_rewards = []
    advantages = []
    means = []
    stds = []
    max_r = []
    min_r = []
    for i in range(0, len(rollout_responses), group_size):
        rewards = []
        for response, gt in zip(rollout_responses[i:i+group_size], repeated_ground_truths[i:i+group_size]):
            reward = reward_fn(response, gt)["reward"]
            rewards.append(reward)
        group_raw_rewards = torch.tensor(rewards)
        mean = group_raw_rewards.mean()
        std = group_raw_rewards.std()
        group_advantages = group_raw_rewards - mean

        if normaliz_by_std:
            group_advantages = group_advantages / (std + advantage_eps)

        raw_rewards.append(group_raw_rewards)
        advantages.append(group_advantages)
        means.append(mean)
        stds.append(std)
        max_r.append(group_raw_rewards.max())
        min_r.append(group_raw_rewards.min())

    return (
        torch.concat(advantages),
        torch.concat(raw_rewards),
        {
            "mean": means,
            "std": stds,
            "max_reward": max_r,
            "min_reward": min_r
        }
    )

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return - raw_rewards_or_advantages.expand(policy_log_probs.shape) * policy_log_probs

def compute_gpro_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    prob_ratios = torch.exp(policy_log_probs - old_log_probs)
    clip_prob_ratios = torch.clamp(prob_ratios, min=1-cliprange, max=1+cliprange)
    boardcast_advantages = advantages.expand(policy_log_probs.shape)
    return (
        -torch.minimum(prob_ratios*boardcast_advantages, clip_prob_ratios*boardcast_advantages),
        {
            "LHS lower than RHS": prob_ratios < clip_prob_ratios
        }
    )

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type not in ["no_baseline", "reinforce_with_baseline", "grpo_clip"]:
        raise ValueError('loss_type must in ["no_baseline", "reinforce_with_baseline", "grpo_clip"]')

    metadatas = {}
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("when loss is no_baseline, raw_rewards must exist")
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("when loss is no_baseline, advantages must exist")
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("when loss is no_baseline, advantages must exist")
        loss, metadata = compute_gpro_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        metadatas |= metadata
    return (
        loss,
        metadatas
    )

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    mask = mask.to(dtype=tensor.dtype)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_loss,metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    ) # (B, S)
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1) # (B,)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, metadata