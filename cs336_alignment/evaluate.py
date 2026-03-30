import json
from pathlib import Path
from typing import Callable

import torch
from vllm import LLM, SamplingParams
from datasets import load_from_disk
from drgrpo_grader import r1_zero_reward_fn


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
) -> dict[str, object]:
    """Evaluate a language model on a list of prompts and compute metrics."""
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    correct = 0
    format_correct = 0
    eval_outcomes = []

    for output, gt in zip(outputs, ground_truths):
        response = output.outputs[0].text
        reward = reward_fn(response, gt)

        eval_outcomes.append({
            "prompt": output.prompt,
            "response": response,
            "ground_truth": gt,
            "reward": reward,
        })

        if reward["reward"] == 1.0:
            correct += 1
            format_correct += 1
        elif reward["format_reward"] == 1.0:
            format_correct += 1

    total = max(len(eval_outcomes), 1)
    return {
        "acc": correct / total,
        "format_acc": format_correct / total,
        "eval_outcomes": eval_outcomes,
    }

if __name__ == "__main__":
    ds = json.load(open("/home/lin/cs336/dataset/sft-data/sft-reason/val.jsonl","r"))

    project_root = Path(__file__).resolve().parent.parent
    prompt_template = (project_root / "cs336_alignment" / "prompts" / "r1_zero.prompt").read_text()
    prompts = [prompt_template.format(question=row["problem"]) for row in ds]
    ground_truths = [str(row["expected_answer"]) for row in ds]

    llm = LLM(model="/home/lin/cs336/model/Qwen-2.5-Math-1.5B-Base")
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    result = evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truths, sampling_params)
    print(f"acc: {result['acc']:.4f}, format_acc: {result['format_acc']:.4f}")
    json.dump(result["eval_outcomes"], open(project_root / "eval_outcomes.json", "w"), ensure_ascii=False, indent=2)
