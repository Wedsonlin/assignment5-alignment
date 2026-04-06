import json
from pathlib import Path
from typing import Callable

from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
    return_outcomes: bool = False,
) -> dict[str, float]:
    """Evaluate a language model on a list of prompts and compute metrics."""
    outputs = vllm_model.generate(prompts, eval_sampling_params, use_tqdm=False)

    correct = 0
    format_correct = 0
    answer_correct = 0
    response_length_sum = 0
    eval_outcomes = [] if return_outcomes else None

    for output, gt in zip(outputs, ground_truths):
        request_output = output.outputs[0]
        response = request_output.text
        token_ids = getattr(request_output, "token_ids", None)
        response_length_sum += len(token_ids) if token_ids is not None else len(response)
        reward = reward_fn(response, gt)

        if return_outcomes and eval_outcomes is not None:
            eval_outcomes.append({
                "prompt": output.prompt,
                "response": response,
                "ground_truth": gt,
                "reward": reward,
            })

        if reward["reward"] == 1.0:
            correct += 1
        if reward["format_reward"] == 1.0:
            format_correct += 1
        if reward["answer_reward"] == 1.0:
            answer_correct += 1

    total = max(len(eval_outcomes), 1)
    result = {
        "total_reward": correct / total,
        "format_reward": format_correct / total,
        "answer_reward": answer_correct / total,
        "response_length": response_length_sum / total,
    }
    if return_outcomes and eval_outcomes is not None:
        result["eval_outcomes"] = eval_outcomes
    return result

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

    result = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompts,
        ground_truths,
        sampling_params,
        return_outcomes=True,
    )
    print(
        f"total_reward: {result['total_reward']:.4f}, "
        f"format_reward: {result['format_reward']:.4f}, "
        f"answer_reward: {result['answer_reward']:.4f}"
    )
    json.dump(result["eval_outcomes"], open(project_root / "eval_outcomes.json", "w"), ensure_ascii=False, indent=2)
