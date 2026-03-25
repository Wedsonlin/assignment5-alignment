from transformers import AutoModelForCausalLM, AutoTokenizer


save_directory = "/home/lin/cs336/model/Qwen-2.5-Math-1.5B-Base-SFT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "/home/lin/cs336/model/Qwen-2.5-Math-1.5B-Base",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/lin/cs336/model/Qwen-2.5-Math-1.5B-Base",
    )
    
    input_ids = train_batch['input_ids'].to(device)
    labels = train_batch['labels'].to(device)

    logits = model(input_ids).logits
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    