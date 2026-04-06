import argparse
import random
from pathlib import Path
from typing import Literal, Callable

import torch

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, get_cosine_schedule_with_warmup

from sft import evaluate, init_vllm, load_policy_into_vllm_instance
from utils import tokenize_prompt_and_output, get_response_log_probs
from utils import Logger, SFTDataset
from grpo_utils import compute_group_normalized_rewards, grpo_microbatch_train_step
from drgrpo_grader import r1_zero_reward_fn

def copy_model(model: PreTrainedModel):
    new_model = AutoModelForCausalLM.from_config(model.config, torch_dtype=model.dtype)
    new_model.load_state_dict(model.state_dict())
    new_model.to(model.device)
    return new_model

def sample_batch(questions, groundtruths, n_prompts_per_rollout_batch):
    rnd_indices = random.sample(range(len(questions)), n_prompts_per_rollout_batch)
    sample_questions = [questions[i] for i in rnd_indices]
    sample_groundtruths = [groundtruths[i] for i in rnd_indices]
    return sample_questions, sample_groundtruths

def grpo(
    init_policy: PreTrainedModel,
    model_id: str,
    vllm_device: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    task_questions: list[str],
    task_groundtruths: list[str],
    tokenizer: AutoTokenizer,
    eval_prompts: list[str],
    eval_groundtruths: list[str],
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: Literal[
        "no_baseline", 
        "reinforce_with_baseline", 
        "grpo_clip",
    ] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    logger: Logger | None = None,
    save_model_dir: str | None = None,
    model_name: str | None = None,
):
    if loss_type != "grpo_clip" and epochs_per_rollout_batch > 1:
        raise ValueError(
            f"Off-policy training (epochs_per_rollout_batch={epochs_per_rollout_batch}) "
            f"requires importance sampling correction. Use loss_type='grpo_clip' "
            f"or set epochs_per_rollout_batch=1."
        )

    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    random.seed(seed)
    torch.manual_seed(seed)

    policy = copy_model(init_policy)
    vllm = init_vllm(model_id, vllm_device, gpu_memory_utilization=gpu_memory_utilization)
    rollout_params = SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95)
    )

    optim_steps_per_grpo_step = (n_microbatches_per_rollout_batch * epochs_per_rollout_batch) // gradient_accumulation_steps
    total_optim_steps = max(1, optim_steps_per_grpo_step * n_grpo_steps)
    warmup_steps = max(1, int(total_optim_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps,
    )

    eval_every_n_grpo_steps = 5
    
    for grpo_step in range(n_grpo_steps):
        rollout_prompts, rollout_groundtruths = sample_batch(task_questions, task_groundtruths, n_prompts_per_rollout_batch)
        load_policy_into_vllm_instance(policy, vllm)

        outputs = vllm.generate(rollout_prompts, rollout_params)

        rollout_responses = []
        for i in range(len(outputs)):
            for j in range(group_size):
                rollout_responses.append(outputs[i].outputs[j].text)
        repeated_prompts = [p for p in rollout_prompts for _ in range(group_size)]
        repeated_ground_truths = [gt for gt in rollout_groundtruths for _ in range(group_size)]
        
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn,
            rollout_responses,
            repeated_ground_truths,
            group_size,
            advantage_eps,
            use_std_normalization
        )

        advantages = advantages.unsqueeze(-1) # (B,1)
        raw_rewards = raw_rewards.unsqueeze(-1) # (B,1)

        train_batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer, max_length=sampling_max_tokens)
        input_ids = train_batch["input_ids"] # (B,S)
        labels = train_batch["labels"] # (B,S)
        masks = train_batch["response_mask"] # (B,S)

        with torch.inference_mode():
            result = get_response_log_probs(policy, input_ids.to(policy.device), labels.to(policy.device), return_token_entropy=False)
            old_log_probs = result["log_probs"].cpu()

        policy.train()
        micro_step = 0
        optim_step = 0
        for epoch in range(epochs_per_rollout_batch): # each epoch should iterate through the rollout batch
            perm = torch.randperm(rollout_batch_size)
            input_ids, labels, masks, old_log_probs = input_ids[perm], labels[perm], masks[perm], old_log_probs[perm]
            advantages, raw_rewards = advantages[perm], raw_rewards[perm]
            optimizer.zero_grad()

            train_batch_loss = 0.0
            train_batch_entropy = 0.0
            for micro_batch_step in range(n_microbatches_per_rollout_batch):
                start = micro_batch_step * micro_train_batch_size
                end = start + micro_train_batch_size
                micro_batch_input_ids = input_ids[start:end, :].to(policy.device)
                micro_batch_labels = labels[start:end, :].to(policy.device)
                micro_batch_masks = masks[start:end, :].to(policy.device)
                micro_batch_old_log_probs = old_log_probs[start:end, :].to(policy.device)
                micro_batch_advantages = advantages[start:end].to(policy.device)
                micro_batch_raw_rewards = raw_rewards[start:end].to(policy.device)

                result = get_response_log_probs(policy, micro_batch_input_ids, micro_batch_labels, return_token_entropy=True)
                policy_log_probs = result["log_probs"]
                entropy = result["token_entropy"]
                loss, train_metadata = grpo_microbatch_train_step(
                    policy_log_probs,
                    micro_batch_masks,
                    gradient_accumulation_steps,
                    loss_type,
                    micro_batch_raw_rewards,
                    micro_batch_advantages,
                    micro_batch_old_log_probs,
                    cliprange=0.2,
                )

                entropy_value = (
                    (entropy * micro_batch_masks).sum() / micro_batch_masks.sum().clamp_min(1)
                ).item()
                train_batch_entropy += entropy_value
                train_batch_loss += loss.item()
                micro_step += 1

                if micro_step % gradient_accumulation_steps == 0:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    optim_step += 1

                    avg_loss = train_batch_loss / gradient_accumulation_steps
                    avg_entropy = train_batch_entropy / gradient_accumulation_steps
                    train_batch_loss = 0.0
                    train_batch_entropy = 0.0

                    policy.eval()
                    with torch.no_grad():
                        train_rewards = evaluate(policy, vllm, rollout_prompts, rollout_groundtruths)
                    policy.train()

                    lr = optimizer.param_groups[0]["lr"]
                    clip_msg = ""
                    cf = train_metadata.get("clip_fraction")
                    if cf is not None:
                        clip_msg = f", clip_frac:{float(cf):.4f}"

                    print(
                        f"[train] grpo_step:{grpo_step + 1} epoch:{epoch + 1} "
                        f"optim_step:{optim_step} micro:{micro_step} "
                        f"loss:{avg_loss:.4f} entropy:{avg_entropy:.4f} "
                        f"grad_norm:{float(gradient_norm):.4f} lr:{lr:.2e} "
                        f"reward:{train_rewards['total_reward']:.4f} "
                        f"fmt:{train_rewards['format_reward']:.4f} "
                        f"ans:{train_rewards['answer_reward']:.4f}"
                        f"{clip_msg}"
                    )
                    if logger:
                        log_dict = {
                            "loss": avg_loss,
                            "entropy": avg_entropy,
                            "gradient_norm": gradient_norm.item(),
                            "total_reward": train_rewards["total_reward"],
                            "format_reward": train_rewards["format_reward"],
                            "answer_reward": train_rewards["answer_reward"],
                        }
                        log_dict |= train_metadata
                        logger.log_train(log_dict)

        if grpo_step % eval_every_n_grpo_steps == 0:
            policy.eval()
            with torch.no_grad():
                eval_rewards = evaluate(policy, vllm, eval_prompts, eval_groundtruths)
            policy.train()

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [eval] grpo_step:{grpo_step + 1} optim_step:{optim_step} lr:{lr:.2e} "
                f"total_reward:{eval_rewards['total_reward']:.4f} "
                f"format_reward:{eval_rewards['format_reward']:.4f} "
                f"answer_reward:{eval_rewards['answer_reward']:.4f}"
            )
            if logger:
                log_dict = {
                    "total_reward": eval_rewards["total_reward"],
                    "format_reward": eval_rewards["format_reward"],
                    "answer_reward": eval_rewards["answer_reward"],
                }
                logger.log_eval(log_dict)

    if save_model_dir and model_name:
        save_path = save_model_dir + model_name
        policy.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path) 

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--epochs-per-rollout", type=int, default=1)
    parser.add_argument("--loss-type", type=str, default="reinforce_with_baseline",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    args = parser.parse_args()

    BASE_DIR = "/root/autodl-tmp/"
    PROJECT_ROOT = Path(__file__).resolve().parents[0]
    SEED = 42
    LOG = False
    SAVE_MODEL = False
    MODEL_ID = BASE_DIR + "model/Qwen-2.5-Math-1.5B-Base"
    TRAIN_DEVICE_ID = 1
    VLLM_DEVICE_ID = 0
    VLLM_GPU_MEMORY_UTILIZATION = 0.9
    EVAL_EXAMPLE_NUM = 1024

    train_data_dir = BASE_DIR + "dataset/sft-data/sft-reason/train.jsonl"
    eval_data_dir = BASE_DIR + "dataset/sft-data/sft-reason/val.jsonl"
    save_dir = BASE_DIR + "model/Qwen-2.5-Math-1.5B-Base-RL/"

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Dual-GPU mode requires at least 2 visible CUDA devices.")
    train_device = torch.device(f"cuda:{TRAIN_DEVICE_ID}")
    vllm_device = f"cuda:{VLLM_DEVICE_ID}"

    train_data = SFTDataset(train_data_dir)
    eval_data = SFTDataset(eval_data_dir, EVAL_EXAMPLE_NUM)

    r1_zero_prompt = open(str(PROJECT_ROOT) + "/prompts/r1_zero.prompt", "r").read()

    train_prompts = [r1_zero_prompt.format(question=ds['problem']) for ds in train_data]
    train_gts = [str(ds["expected_answer"]) for ds in train_data]
    assert len(train_prompts) == len(train_gts)

    eval_prompts = [r1_zero_prompt.format(question=ds['problem']) for ds in eval_data]
    eval_gts = [str(ds["expected_answer"]) for ds in eval_data]
    assert len(eval_prompts) == len(eval_gts)

    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(train_device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    grpo_model_name = f"grpo_G{args.group_size}_{args.loss_type}"
    logger = Logger(project="grpo", name=grpo_model_name) if LOG else None

    grpo(
        init_policy=policy,
        model_id=MODEL_ID,
        vllm_device=vllm_device,
        reward_fn=r1_zero_reward_fn,
        task_questions=train_prompts,
        task_groundtruths=train_gts,
        tokenizer=tokenizer,
        eval_prompts=eval_prompts,
        eval_groundtruths=eval_gts,
        group_size=args.group_size,
        epochs_per_rollout_batch=args.epochs_per_rollout,
        loss_type=args.loss_type,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        logger=logger,
        save_model_dir=save_dir if SAVE_MODEL else None,
        model_name=grpo_model_name if SAVE_MODEL else None,
        seed=SEED,
    )

    if logger:
        logger.finish()
