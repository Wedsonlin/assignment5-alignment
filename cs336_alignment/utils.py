import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenzier: PreTrainedTokenizer, the tokenizer to use.
    """
    assert len(prompt_strs) == len(output_strs)

    # tokenize prompt and output separately
    prompt_ids = tokenizer(prompt_strs).input_ids
    output_ids = tokenizer(output_strs).input_ids

    # concatenate and pad, default using tokenizer.pad_token_id
    prompt_output_ids = [{"input_ids":x+y} for x,y in zip(prompt_ids,output_ids)]
    full_ids = tokenizer.pad(prompt_output_ids,padding=True,return_tensors="pt").input_ids
    
    # set response mask
    response_mask = torch.zeros(full_ids.shape)
    for i in range(len(prompt_strs)):
        response_mask[i,len(prompt_ids[i]):len(prompt_ids[i])+len(output_ids[i])] = 1

    input_ids = full_ids[:,:-1]
    labels = full_ids[:,1:]
    response_mask = response_mask[:,1:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }
        
def compute_entropy(logits: Tensor) -> Tensor:
    """Get the entropy of the next-token predictions
    
    Args:
        logits: torch.Tensor, Tensor of shape (batch_size, sequence_length, vocab_size) containing unnormalized logits

    Return:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token prediction.
    """
    log_probs = logits - torch.logsumexp(logits,dim=-1,keepdim=True)
    return -(torch.exp(log_probs) * log_probs).sum(dim=-1)