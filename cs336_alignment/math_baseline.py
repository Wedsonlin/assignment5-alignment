from vllm import LLM, SamplingParams
from typing import Callable
from drgrpo_grader import r1_zero_reward_fn
from datasets import load_dataset,load_from_disk
from tqdm import tqdm
import json

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams
) -> None:
    """ Evaluate a language model on a list of prompts, 
        compute evaluation metrics, and serialize results to disk. 
    """  
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    eval_outcomes = []
    for i in range(len(outputs)):
        prompt = outputs[i].prompt
        response = outputs[i].outputs[0].text
        reward = r1_zero_reward_fn(response, ground_truths[i])
        eval_outcomes.append({
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truths[i],
            "reward": reward
        })
    json.dump(eval_outcomes, open("../eval_outcomes.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ds = load_from_disk("/home/lin/cs336/dataset/MATH/data_split")
    
    # split = ds.train_test_split(
    #     test_size=0.3,
    #     seed=42,
    #     shuffle=True
    # )
    # split.save_to_disk("/home/lin/cs336/dataset/MATH/data_split")

    r1_zero_prompt = open("./prompts/r1_zero.prompt", "r").read()
    prompts = [r1_zero_prompt.format(question=row['problem']) for row in ds['test']]
    ground_truths = [row['solution'] for row in ds['test']]

    model_path = "/home/lin/cs336/model/Qwen-2.5-Math-1.5B-Base"
    llm = LLM(model=model_path)

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )
    
    evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truths, sampling_params)
