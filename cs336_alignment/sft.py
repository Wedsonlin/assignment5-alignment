import argparse
import json
import random
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, get_cosine_schedule_with_warmup
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch

from utils import Logger, SFTDataset
from utils import (
                tokenize_prompt_and_output, 
                get_response_log_probs, 
                sft_microbatch_train_step,
            )
from evaluate import evaluate_vllm
from drgrpo_grader import r1_zero_reward_fn


def init_vllm(model_id: str, device: str, seed: int = 42, gpu_memory_utilization: float = 0.85):
    """ Start the inference process, 
        here we use vLLM to hold a model on a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1) # avoid parallel division
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    # GPU0 -> CPU -> GPU1, more stable copy
    state_dict = {name: tensor.detach().cpu() for name, tensor in policy.state_dict().items()}
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def evaluate(policy: PreTrainedModel, llm: LLM, eval_prompts: list[str], eval_gts: list[str]):
    load_policy_into_vllm_instance(policy, llm)
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )
    result = evaluate_vllm(llm, r1_zero_reward_fn, eval_prompts, eval_gts, sampling_params)
    return result

def sft(
    sft_prompts: list[str],
    sft_responses: list[str],
    policy: PreTrainedModel,
    tokenizer: AutoTokenizer,
    optimizer,
    train_device: torch.device,
    epoch_size: int,
    microbatch_size: int,
    gradient_accumulation_steps: int,
    max_length: int = 1024,
    eval_llm: LLM | None = None,
    eval_prompts: list[str] | None = None,
    eval_gts: list[str] | None = None,
    eval_every_n_optim_steps: int = 4,
    logger: Logger | None = None,
    warmup_ratio: float = 0.1,
    save_model_dir: str | None = None,
    sft_model_name: str | None = None,
):
    optim_steps_per_epoch = len(sft_prompts) // (microbatch_size * gradient_accumulation_steps)
    total_optim_steps = max(1, optim_steps_per_epoch * epoch_size)
    warmup_steps = max(1, int(total_optim_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps,
    )

    train_batch = tokenize_prompt_and_output(sft_prompts, sft_responses, tokenizer, max_length)
    input_ids = train_batch["input_ids"]   # (B,S)
    labels = train_batch["labels"]         # (B,S)
    masks = train_batch["response_mask"]   # (B,S)

    policy.train()
    micro_step = 0
    optim_step = 0
    for epoch in range(epoch_size):
        perm = torch.randperm(input_ids.shape[0])
        input_ids, labels, masks = input_ids[perm], labels[perm], masks[perm]
        optimizer.zero_grad()

        for i in range(0, input_ids.shape[0], microbatch_size):
            end = min(i + microbatch_size, input_ids.shape[0])
            batch_input_ids = input_ids[i:end, :].to(train_device)
            batch_labels = labels[i:end, :].to(train_device)
            batch_masks = masks[i:end, :].to(train_device)

            result = get_response_log_probs(policy, batch_input_ids, batch_labels, return_token_entropy=True)
            log_probs = result["log_probs"]
            entropy = result["token_entropy"]

            loss, metadata = sft_microbatch_train_step(log_probs, batch_masks, gradient_accumulation_steps)
            nll = metadata['nll'].item()
            micro_step += 1
            print(f"epoch:{epoch+1}, micro:{micro_step}, train/nll:{nll:.4f}")
            if logger:
                entropy_value = ((entropy * batch_masks).sum() / batch_masks.sum().clamp_min(1)).item()
                logger.log_train({"loss": nll, "entropy": entropy_value})

            if micro_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                if eval_prompts and eval_gts and optim_step % eval_every_n_optim_steps == 0:
                    policy.eval()
                    with torch.no_grad():
                        eval_result = evaluate(policy, eval_llm, eval_prompts, eval_gts)
                    policy.train()

                    print(f"  [eval] optim_step:{optim_step}, lr:{scheduler.get_last_lr()[0]:.2e}, "
                          f"eval/total_reward:{eval_result['total_reward']:.4f}, "
                          f"eval/format_reward:{eval_result['format_reward']:.4f}, "
                          f"eval/answer_reward:{eval_result['answer_reward']:.4f}, "
                          f"eval/response_length:{eval_result['response_length']:.4f}")
                    if logger:
                        logger.log_eval({
                            "total_reward": eval_result["total_reward"],
                            "format_reward": eval_result["format_reward"],
                            "answer_reward": eval_result["answer_reward"],
                            "response_length": eval_result["response_length"],
                        })

    if save_model_dir and sft_model_name:
        save_path = save_model_dir + sft_model_name
        policy.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    return policy


if __name__ == "__main__":
    BASE_DIR = "/root/autodl-tmp/"
    PROJECT_ROOT = Path(__file__).resolve().parents[0]
    SEED = 42
    LOG = True
    SAVE_MODEL = True
    MODEL_ID = BASE_DIR + "model/Qwen-2.5-Math-1.5B-Base"
    TRAIN_DEVICE_ID = 1
    VLLM_DEVICE_ID = 0
    VLLM_GPU_MEMORY_UTILIZATION = 0.85

    sft_data_dir = BASE_DIR + "dataset/sft-data/sft-reason/sft_gpt-oss-120b.jsonl"
    filtered_sft_data_dir = BASE_DIR + "dataset/sft-data/sft-reason/sft_gpt-oss-120b_filtered.jsonl"
    eval_data_dir = BASE_DIR + "dataset/sft-data/sft-reason/val.jsonl"
    save_dir = BASE_DIR + "model/Qwen-2.5-Math-1.5B-Base-SFT/"
    epoch_size = 3
    microbatch_size = 2
    gradient_accumulation_steps = 16
    warmup_ratio = 0.1
    eval_every_n_optim_steps = 4

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_num", type=int, default=0,
                        help="Number of training samples to use (0 = full dataset)")
    parser.add_argument("--filtered", action="store_true",
                        help="Use the filtered training dataset")
    args = parser.parse_args()
    sample_num = args.sample_num
    filtered = args.filtered

    label = "all" if sample_num == 0 else str(sample_num)
    suffix = "_filtered" if filtered else ""
    sft_model_name = f"num_example_{label}{suffix}"

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Dual-GPU mode requires at least 2 visible CUDA devices.")
    train_device = torch.device(f"cuda:{TRAIN_DEVICE_ID}")
    vllm_device = f"cuda:{VLLM_DEVICE_ID}"

    sft_data = SFTDataset(filtered_sft_data_dir if filtered else sft_data_dir, sample_num, SEED)
    sft_data.align(microbatch_size * gradient_accumulation_steps)
    eval_data = SFTDataset(eval_data_dir)

    r1_zero_prompt = open(str(PROJECT_ROOT) + "/prompts/r1_zero.prompt", "r").read()

    prompts = [r1_zero_prompt.format(question=ds['problem']) for ds in sft_data]
    responses = [ds["reasoning_trace"] for ds in sft_data]
    assert len(prompts) == len(responses)

    eval_prompts = [r1_zero_prompt.format(question=ds['problem']) for ds in eval_data]
    eval_gts = [str(ds["expected_answer"]) for ds in eval_data]
    assert len(eval_prompts) == len(eval_gts)

    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(train_device)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=2e-5
    )
    llm = init_vllm(
        model_id=MODEL_ID,
        device=vllm_device,
        seed=SEED,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION
    )
    logger = Logger(project="sft", name=sft_model_name) if LOG else None

    sft(
        sft_prompts=prompts,
        sft_responses=responses,
        policy=policy,
        tokenizer=tokenizer,
        optimizer=optimizer,
        eval_llm=llm,
        train_device=train_device,
        epoch_size=epoch_size,
        microbatch_size=microbatch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_prompts=eval_prompts,
        eval_gts=eval_gts,
        eval_every_n_optim_steps=eval_every_n_optim_steps,
        logger=logger,
        save_model_dir=save_dir,
        sft_model_name=sft_model_name,
    )

    if logger:
        logger.finish()
