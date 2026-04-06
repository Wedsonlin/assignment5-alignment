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
from grpo_utils import compute_group_normalized_rewards, grpo_microbatch_train_step, masked_mean
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

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
        "grpo_no_clip",
    ] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    length_normalization: Literal["grpo","dr.grpo"] = "grpo",
    warmup_ratio: float = 0.1,
    seed: int = 42,
    logger: Logger | None = None,
    save_model_dir: str | None = None,
    model_name: str | None = None,
):
    if loss_type not in ["grpo_clip", "grpo_no_clip"] and epochs_per_rollout_batch > 1:
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

    eval_every_n_grpo_steps = 10
    
    for grpo_step in range(n_grpo_steps):
        rollout_prompts, rollout_groundtruths = sample_batch(task_questions, task_groundtruths, n_prompts_per_rollout_batch)
        load_policy_into_vllm_instance(policy, vllm)

        outputs = vllm.generate(rollout_prompts, rollout_params, use_tqdm=False)

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

        advantages = advantages.unsqueeze(-1).to(policy.device) # (B,1)
        raw_rewards = raw_rewards.unsqueeze(-1).to(policy.device) # (B,1)

        train_batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        input_ids = train_batch["input_ids"].to(policy.device) # (B,S)
        labels = train_batch["labels"].to(policy.device) # (B,S)
        masks = train_batch["response_mask"].to(policy.device) # (B,S)
        total_response_tokens = masks.sum().item()

        old_log_probs = None
        if loss_type in ["grpo_clip", "grpo_no_clip"] or epochs_per_rollout_batch > 1 or train_batch_size != rollout_batch_size:
            old_log_probs_list = []
            policy.eval()
            with torch.inference_mode():
                for mb in range(0, rollout_batch_size, micro_train_batch_size):
                    mb_end = mb + micro_train_batch_size
                    result = get_response_log_probs(
                        policy,
                        input_ids[mb:mb_end, :],
                        labels[mb:mb_end, :],
                        return_token_entropy=False,
                    )
                    old_log_probs_list.append(result["log_probs"].detach().clone())
            old_log_probs = torch.cat(old_log_probs_list, dim=0).to(policy.device) # (B, S)

        policy.train()
        micro_step = 0
        optim_step = 0
        for epoch in range(epochs_per_rollout_batch): # each epoch should iterate through the rollout batch
            perm = torch.randperm(rollout_batch_size, device=input_ids.device)
            input_ids, labels, masks = input_ids[perm], labels[perm], masks[perm]
            advantages, raw_rewards = advantages[perm], raw_rewards[perm]
            if old_log_probs is not None:
                old_log_probs = old_log_probs[perm]
            optimizer.zero_grad()

            train_batch_loss = 0.0
            train_batch_entropy = 0.0
            for micro_batch_step in range(n_microbatches_per_rollout_batch):
                start = micro_batch_step * micro_train_batch_size
                end = start + micro_train_batch_size
                micro_batch_input_ids = input_ids[start:end, :]
                micro_batch_labels = labels[start:end, :]
                micro_batch_masks = masks[start:end, :]
                micro_batch_advantages = advantages[start:end, :]
                micro_batch_raw_rewards = raw_rewards[start:end, :]
                micro_batch_old_log_probs = old_log_probs[start:end, :] if old_log_probs is not None else None

                result = get_response_log_probs(
                    policy, 
                    micro_batch_input_ids, 
                    micro_batch_labels, 
                    return_token_entropy=True
                )
                policy_log_probs = result["log_probs"]
                entropy = result["token_entropy"]
                loss, train_metadata = grpo_microbatch_train_step(
                    policy_log_probs = policy_log_probs,
                    response_mask = micro_batch_masks,
                    gradient_accumulation_steps = gradient_accumulation_steps,
                    loss_type = loss_type,
                    raw_rewards = micro_batch_raw_rewards,
                    advantages= micro_batch_advantages,
                    old_log_probs = micro_batch_old_log_probs,
                    cliprange = 0.2,
                    constant_normalize_factor = total_response_tokens if length_normalization == "dr.grpo" else None
                )

                with torch.no_grad():
                    entropy_value = masked_mean(entropy, micro_batch_masks, dim=None).item()
                    train_batch_entropy += entropy_value
                    train_batch_loss += loss.item()
                    micro_step += 1

                if micro_step % gradient_accumulation_steps == 0:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    optim_step += 1

                    avg_entropy = train_batch_entropy / gradient_accumulation_steps

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
                        f"loss:{train_batch_loss:.4f} entropy:{avg_entropy:.4f} "
                        f"grad_norm:{float(gradient_norm):.6f} lr:{lr:.2e} "
                        f"reward:{train_rewards['total_reward']:.4f} "
                        f"fmt:{train_rewards['format_reward']:.4f} "
                        f"ans:{train_rewards['answer_reward']:.4f}"
                        f"{clip_msg}"
                    )
                    if logger:
                        log_dict = {
                            "loss": train_batch_loss,
                            "entropy": avg_entropy,
                            "gradient_norm": gradient_norm.item(),
                            "total_reward": train_rewards["total_reward"],
                            "format_reward": train_rewards["format_reward"],
                            "answer_reward": train_rewards["answer_reward"],
                        }
                        log_dict |= train_metadata
                        logger.log_train(log_dict)
                        
                    train_batch_loss = 0.0
                    train_batch_entropy = 0.0

        if (grpo_step + 1) % eval_every_n_grpo_steps == 0:
            policy.eval()
            with torch.no_grad():
                eval_rewards = evaluate(policy, vllm, eval_prompts, eval_groundtruths)
            policy.train()

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[eval] grpo_step:{grpo_step + 1} optim_step:{optim_step} lr:{lr:.2e} "
                f"total_reward:{eval_rewards['total_reward']:.4f} "
                f"format_reward:{eval_rewards['format_reward']:.4f} "
                f"answer_reward:{eval_rewards['answer_reward']:.4f} "
                f"response_length:{eval_rewards['response_length']:.4f}"
            )
            if logger:
                log_dict = {
                    "total_reward": eval_rewards["total_reward"],
                    "format_reward": eval_rewards["format_reward"],
                    "answer_reward": eval_rewards["answer_reward"],
                    "response_length": eval_rewards["response_length"],
                }
                logger.log_eval(log_dict)

    if save_model_dir and model_name:
        save_path = save_model_dir + model_name
        policy.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path) 

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--length-normalization", type=str, default="grpo",
                        choices=["grpo","dr.grpo"])
    parser.add_argument("--use-std-normalization", action="store_true")
    parser.add_argument("--n-grpo-steps", type=int, default=200)
    parser.add_argument("--epochs-per-rollout", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=128)
    parser.add_argument("--loss-type", type=str, default="reinforce_with_baseline",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"])
    parser.add_argument("--prompt-type", type=str, default="r1-zero",
                        choices=["r1-zero", "question-only"])
    parser.add_argument("--project-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
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
    
    prompt_dir = "/prompts/r1_zero.prompt" if args.prompt_type == "r1-zero" else "/prompts/question_only.prompt"
    prompt_template = open(str(PROJECT_ROOT) + prompt_dir, "r").read()
    reward_fn = r1_zero_reward_fn if args.prompt_type == "r1-zero" else question_only_reward_fn

    train_prompts = [prompt_template.format(question=ds['problem']) for ds in train_data]
    train_gts = [str(ds["expected_answer"]) for ds in train_data]
    assert len(train_prompts) == len(train_gts)

    eval_prompts = [prompt_template.format(question=ds['problem']) for ds in eval_data]
    eval_gts = [str(ds["expected_answer"]) for ds in eval_data]
    assert len(eval_prompts) == len(eval_gts)

    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(train_device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    project_name = "default" if args.project_name is None else args.project_name
    model_name = "default" if args.model_name is None else args.model_name
    logger = Logger(project=project_name, name=model_name) if LOG else None

    grpo(
        init_policy=policy,
        model_id=MODEL_ID,
        vllm_device=vllm_device,
        reward_fn=reward_fn,
        task_questions=train_prompts,
        task_groundtruths=train_gts,
        tokenizer=tokenizer,
        eval_prompts=eval_prompts,
        eval_groundtruths=eval_gts,
        n_grpo_steps=args.n_grpo_steps,
        learning_rate=args.lr,
        length_normalization=args.length_normalization,
        use_std_normalization=args.use_std_normalization,
        epochs_per_rollout_batch=args.epochs_per_rollout,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_type=args.loss_type,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        logger=logger,
        save_model_dir=save_dir if SAVE_MODEL else None,
        model_name=model_name if SAVE_MODEL else None,
        seed=SEED,
    )

    if logger:
        logger.finish()
