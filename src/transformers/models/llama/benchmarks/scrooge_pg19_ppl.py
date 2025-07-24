import torch
from transformers import AutoTokenizer, SentenceLlamaForCausalLM
import datasets

if __name__ == "__main__":

    model_class = SentenceLlamaForCausalLM
    checkpoint_dir = "./sentence_Llama-3.2-1B_pretrain_with_end_of_sentence_full_BTLCR6IG/checkpoint-2000"

    model = model_class.from_pretrained(checkpoint_dir, torch_dtype=torch.float32)
    model.eval()
    model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    dataset = datasets.load_dataset("deepmind/pg19", split="test")

    


    for item in dataset:

        input_ids = tokenizer.encode(item["text"], return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        print(outputs)