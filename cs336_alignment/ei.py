import random
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import Logger, SFTDataset
from sft import init_vllm, load_policy_into_vllm_instance, sft, evaluate
from drgrpo_grader import r1_zero_reward_fn


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

    ei_data_dir = BASE_DIR + "dataset/sft-data/sft-reason/train.jsonl"
    eval_data_dir = BASE_DIR + "dataset/sft-data/sft-reason/val.jsonl"
    save_dir = BASE_DIR + "model/Qwen-2.5-Math-1.5B-Base-EI/"
    epoch_size = 3

    # sft params
    microbatch_size = 2
    gradient_accumulation_steps = 16
    warmup_ratio = 0.1
    eval_every_n_optim_steps = 4

    # ei params
    sampling_temperature = 0.5
    sampling_max_tokens = 1024
    sampling_min_tokens = 4
    sampling_prompt_num = 4
    G = 4
    n_ei_steps = 5

    ei_model_name = f"G_{G}_Epoch_{epoch_size}"

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Dual-GPU mode requires at least 2 visible CUDA devices.")
    train_device = torch.device(f"cuda:{TRAIN_DEVICE_ID}")
    vllm_device = f"cuda:{VLLM_DEVICE_ID}"

    ei_data = SFTDataset(ei_data_dir)
    eval_data = SFTDataset(eval_data_dir)

    r1_zero_prompt = open(str(PROJECT_ROOT) + "/prompts/r1_zero.prompt", "r").read()

    ei_prompts = [r1_zero_prompt.format(question=ds['problem']) for ds in ei_data]
    ei_gts = [str(ds["expected_answer"]) for ds in ei_data]
    assert len(ei_prompts) == len(ei_gts)

    eval_prompts = [r1_zero_prompt.format(question=ds['problem']) for ds in eval_data]
    eval_gts = [str(ds["expected_answer"]) for ds in eval_data]
    assert len(eval_prompts) == len(eval_gts)

    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(train_device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-5)
    llm = init_vllm(
        model_id=MODEL_ID,
        device=vllm_device,
        seed=SEED,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    )

    logger = Logger(project="ei", name=ei_model_name) if LOG else None

    generation_params = SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=G,
        seed=SEED,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    for ei_step in range(n_ei_steps):
        # Step 3: sample a batch of questions
        rnd_idx = random.sample(range(len(ei_prompts)), sampling_prompt_num)
        sampled_prompts = [ei_prompts[idx] for idx in rnd_idx]
        sampled_gts = [ei_gts[idx] for idx in rnd_idx]

        # Step 4: sync current policy weights into vLLM
        load_policy_into_vllm_instance(policy, llm)

        # Step 5: sample G outputs per question via vLLM
        outputs = llm.generate(sampled_prompts, generation_params)

        # Step 6-7: compute rewards and filter correct responses
        filtered_prompts = []
        filtered_responses = []
        for i in range(len(outputs)):
            gt = sampled_gts[i]
            for j in range(G):
                response = outputs[i].outputs[j].text
                reward = r1_zero_reward_fn(response, gt)
                if reward["reward"] == 1.0:
                    filtered_prompts.append(sampled_prompts[i])
                    filtered_responses.append(response)

        print(f"[EI step {ei_step+1}/{n_ei_steps}] "
              f"sampled {len(sampled_prompts)} questions, "
              f"filtered {len(filtered_responses)} correct responses")

        if len(filtered_responses) == 0:
            print("  No correct responses found, skipping SFT for this step.")
            continue

        # Step 8: SFT on filtered correct responses
        policy = sft(
            sft_prompts=filtered_prompts,
            sft_responses=filtered_responses,
            policy=policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            eval_llm=llm,
            train_device=train_device,
            epoch_size=epoch_size,
            microbatch_size=microbatch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logger=logger
        )

        # Evaluate after each EI step
        policy.eval()
        with torch.no_grad():
            eval_result = evaluate(policy, llm, eval_prompts, eval_gts)
        policy.train()

        print(f"  [EI eval] step:{ei_step+1}, "
              f"eval/acc:{eval_result['acc']:.4f}, "
              f"eval/format_acc:{eval_result['format_acc']:.4f}")
        if logger:
            logger.log_eval(
                acc=eval_result["acc"],
                format_acc=eval_result["format_acc"],
            )

    if SAVE_MODEL:
        save_path = save_dir + ei_model_name
        policy.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    if logger:
        logger.finish()
