import json
import random
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch

from utils import Logger, SFTDataset
from utils import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step
from evaluate import evaluate_vllm
from drgrpo_grader import r1_zero_reward_fn

PROJECT_ROOT = Path(__file__).resolve().parents[0]
SEED = 42
SAMPLE_NUM = 128
LOG = False
SAVE_MODEL = False
MODEL_ID = "/home/lin/cs336/model/Qwen-2.5-Math-1.5B-Base"
# Default dual-GPU setup (no env switch): train on cuda:0, evaluate with vLLM on cuda:1.
TRAIN_DEVICE_ID = 0
VLLM_DEVICE_ID = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.85

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
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
            gpu_memory_utilization=gpu_memory_utilization
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    # GPU0 -> CPU -> GPU1, more stable copy
    state_dict = {name: tensor.detach().cpu() for name, tensor in policy.state_dict().items()}
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def evaluate(policy: PreTrainedModel, llm: LLM, eval_prompts, eval_gts, vllm_device: str):
    load_policy_into_vllm_instance(policy, llm)
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )
    result = evaluate_vllm(llm, r1_zero_reward_fn, eval_prompts, eval_gts, sampling_params)
    return result

sft_data_dir = "/home/lin/cs336/dataset/sft-data/sft-reason/sft_gpt-oss-120b.jsonl"
eval_data_dir = "/home/lin/cs336/dataset/sft-data/sft-reason/val.jsonl"
save_dir = "/home/lin/cs336/model/Qwen-2.5-Math-1.5B-Base-SFT"
microbatch_size = 2
gradient_accumulation_steps = 16

if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Dual-GPU mode requires at least 2 visible CUDA devices.")
    train_device = torch.device(f"cuda:{TRAIN_DEVICE_ID}")
    vllm_device = f"cuda:{VLLM_DEVICE_ID}"

    sft_data = SFTDataset(path=sft_data_dir, sample_num=SAMPLE_NUM, seed=SEED)
    eval_data = SFTDataset(path=sft_data_dir)

    r1_zero_prompt = open(str(PROJECT_ROOT) + "/prompts/r1_zero.prompt", "r").read()

    prompts = [r1_zero_prompt.format(question=ds['problem']) for ds in sft_data]
    responses = [ds["reasoning_trace"] for ds in sft_data]
    assert len(prompts) == len(responses)

    eval_prompts = [r1_zero_prompt.format(question=ds['problem']) for ds in eval_data]
    eval_gts = [str(ds["expected_answer"]) for ds in eval_data]
    assert len(eval_prompts) == len(eval_gts)

    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="flash_attention_2",
    ).to(train_device)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01,
        betas=(0.9, 0.99),
        eps=1e-8
    )
    llm = init_vllm(
        model_id=MODEL_ID,
        device=vllm_device,
        seed=SEED,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION
    )

    if LOG:
        logger = Logger(project="sft", name=f"num_example_{SAMPLE_NUM}")

    train_batch = tokenize_prompt_and_output(sampled_prompts, sampled_responses, tokenizer)
    input_ids = train_batch["input_ids"].to(train_device) # (B,S)
    labels = train_batch["labels"].to(train_device) # (B,S)
    masks = train_batch["response_mask"].to(train_device) # (B,S)

    policy.train()
    optimizer.zero_grad()
    step = 1
    for i in range(0, input_ids.shape[0], microbatch_size):
        end = min(i + microbatch_size, input_ids.shape[0])
        batch_input_ids = input_ids[i:end, :]
        batch_labels = labels[i:end, :]
        batch_masks = masks[i:end, :]

        result = get_response_log_probs(policy, batch_input_ids, batch_labels, return_token_entropy=True)
        log_probs = result["log_probs"]
        entropy = result["token_entropy"]

        loss, _ = sft_microbatch_train_step(log_probs, batch_masks, gradient_accumulation_steps)
        if LOG: # each microbatch
            entropy_value = ((entropy * batch_masks).sum() / batch_masks.sum().clamp_min(1)).item()
            logger.log_train(loss=loss.item(), entropy=entropy_value)

        if step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            policy.eval()
            with torch.no_grad():
                eval_result = evaluate(policy, llm, eval_prompts, eval_gts, vllm_device)
            policy.train()

            print(f"it:{step}, train/loss:{loss.item():.4f}, eval/acc:{eval_result['acc']:.4f}, eval/format_acc:{eval_result['format_acc']:.4f}")
            if LOG: # each minibatch
                logger.log_eval( 
                    acc=eval_result["acc"],
                    format_acc=eval_result["format_acc"]
                )

        step += 1

    if SAVE_MODEL:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    if LOG:
        logger.finish()
