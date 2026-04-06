import json
import random
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import wandb


# def tokenize_prompt_and_output(
#     prompt_strs: list[str],
#     output_strs: list[str],
#     tokenizer: PreTrainedTokenizerBase,
# ) -> dict[str, Tensor]:
#     """Tokenize the prompt and output strings, and construct a mask that is 1
#     for the response tokens and 0 for other tokens (prompt or padding).

#     Args:
#         prompt_strs: list[str], the prompt strings.
#         output_strs: list[str], the output strings.
#         tokenizer: PreTrainedTokenizerBase, the tokenizer to use.
#     """
#     assert len(prompt_strs) == len(output_strs)

#     prompt_ids = tokenizer(prompt_strs, add_special_tokens=False).input_ids
#     output_ids = tokenizer(output_strs, add_special_tokens=False).input_ids

#     prompt_output_ids = [{"input_ids": p + o} for p, o in zip(prompt_ids, output_ids)]
#     full_ids = tokenizer.pad(prompt_output_ids, padding=True, return_tensors="pt").input_ids

#     B, L = full_ids.shape
#     prompt_lens = torch.tensor([len(p) for p in prompt_ids])
#     output_lens = torch.tensor([len(o) for o in output_ids])
#     positions = torch.arange(L).unsqueeze(0).expand(B, L)
#     response_mask = (
#         (positions >= prompt_lens.unsqueeze(1))
#         & (positions < (prompt_lens + output_lens).unsqueeze(1))
#     ).float()

#     input_ids = full_ids[:, :-1]
#     labels = full_ids[:, 1:]
#     response_mask = response_mask[:, 1:]
#     return {
#         "input_ids": input_ids,
#         "labels": labels,
#         "response_mask": response_mask,
#     }

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 1024,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizerBase, the tokenizer to use.
        max_length: int, max token length for the concatenated prompt+output.
    """
    assert len(prompt_strs) == len(output_strs)

    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False).input_ids
    output_ids = tokenizer(output_strs, add_special_tokens=False).input_ids

    combined = []
    truncated_output_lens = []
    for p, o in zip(prompt_ids, output_ids):
        concat = (p + o)[:max_length]
        combined.append({"input_ids": concat})
        truncated_output_lens.append(max(0, len(concat) - len(p)))

    full_ids = tokenizer.pad(combined, padding=True, return_tensors="pt").input_ids

    B, L = full_ids.shape
    prompt_lens = torch.tensor([len(p) for p in prompt_ids])
    output_lens = torch.tensor(truncated_output_lens)
    positions = torch.arange(L).unsqueeze(0).expand(B, L)
    response_mask = (
        (positions >= prompt_lens.unsqueeze(1))
        & (positions < (prompt_lens + output_lens).unsqueeze(1))
    ).float()

    input_ids = full_ids[:, :-1]
    labels = full_ids[:, 1:]
    response_mask = response_mask[:, 1:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Get the entropy of the next-token predictions.

    Args:
        logits: (batch_size, sequence_length, vocab_size), unnormalized logits.

    Returns:
        (batch_size, sequence_length), entropy for each position.
    """
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -(torch.exp(log_probs) * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Get the conditional log-probs of the response given the prompt,
    and optionally the entropy of the next token predictions.

    Args:
        model: the model to score.
        input_ids: (batch_size, sequence_length), tokenized prompt and output.
        labels: (batch_size, sequence_length), shifted input_ids.
        return_token_entropy: whether to return the entropy.

    Returns:
        dict with "log_probs" (batch_size, sequence_length) and
        optionally "token_entropy" (batch_size, sequence_length).
        Prompt/padding masking is done externally.
    """
    logits = model(input_ids).logits # (B,S,V)
    normalized_probs = torch.log_softmax(logits, dim=-1) # (B,S,V), stable log-probs
    log_probs = normalized_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (B,S)
    
    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits.detach())
    return result


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: the tensor to sum and normalize.
        mask: binary mask (1 = include, 0 = exclude).
        dim: dimension to sum along (None = all dimensions).
        normalize_constant: divisor for normalization.

    Returns:
        Normalized masked sum.
    """
    mask = mask.to(dtype=tensor.dtype)
    return (tensor * mask).sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Perform one SFT microbatch train step:
      - compute masked negative log-likelihood
      - normalize by normalize_constant
      - scale by gradient_accumulation_steps
      - call backward()

    Args:
        policy_log_probs: (B, T) log p(x_t | x_<t)
        response_mask: (B, T) 1 for response tokens else 0
        gradient_accumulation_steps: number of microbatches per optimizer step
        normalize_constant: divisor for masked sum (e.g. total response tokens)

    Returns:
        loss: scalar tensor (scaled for gradient accumulation)
        metadata: dict with unscaled nll for logging
    """
    nll = masked_normalize(-policy_log_probs, response_mask, normalize_constant, dim=-1).mean()
    scaled_loss = nll / gradient_accumulation_steps
    scaled_loss.backward()
    return scaled_loss, {"nll": nll.detach()}


class Logger:
    def __init__(self, project: str, name: str):
        self.run = wandb.init(
            entity="ltao02845-sun-yat-sen-university",
            project=project,
            name=name,
        )
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        # everything that starts with train/ is tied to train_step
        wandb.define_metric("train/*", step_metric="train_step")
        # everything that starts with eval/ is tied to eval_step
        wandb.define_metric("eval/*", step_metric="eval_step")
        self.train_step = 0
        self.eval_step = 0

    def log_train(self, item_dict: dict[str, Any]):
        self.train_step += 1
        log_dict = {"train_step": self.train_step}
        for name,value in item_dict.items():
            log_dict[f"train/{name}"] = value
        self.run.log(log_dict)

    def log_eval(self, item_dict: dict[str, Any]):
        self.eval_step += 1
        log_dict = {"train_step": self.eval_step}
        for name,value in item_dict.items():
            log_dict[f"eval/{name}"] = value
        self.run.log(log_dict)

    def finish(self):
        self.run.finish()

class SFTDataset(Dataset):
    def __init__(self, path: str, sample_num: int = 0, seed: int = 0):
        self.data = json.load(open(path, "r"))
        
        if sample_num > 0:
            rnd = random.Random(seed)
            rnd.shuffle(self.data)
            self.data = self.data[:sample_num]

    def align(self, divisor: int):
        """Truncate to largest multiple of divisor so every minibatch is full-sized."""
        n = (len(self.data) // divisor) * divisor
        self.data = self.data[:n]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
