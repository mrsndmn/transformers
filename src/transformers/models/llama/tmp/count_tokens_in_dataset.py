import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer

def count_tokens_in_dataset(dataset, tokenizer, batch_size=32):

    total_tokens = 0

    batch_size = 32

    orig_len = len(dataset)

    dataset = dataset.shuffle(seed=42).select(range(10000))

    sequence_lengths = []

    for batch in tqdm(dataset.iter(batch_size=batch_size), total=(len(dataset) // batch_size)):
        # pass
        tokenized = tokenizer(batch['text'], padding=True, max_length=1024, return_tensors='pt')

        sequence_lengths.extend(tokenized.attention_mask.sum(dim=-1).cpu().numpy().tolist())

        total_tokens += tokenized.attention_mask.sum().item()

    print("total_tokens",total_tokens)
    print('approx total tokens',total_tokens / 10000 * orig_len)

    # plot density of sequence lengths
    plt.hist(sequence_lengths, bins=10000)
    plt.xlim(0, 2048)
    plt.show()
    plt.savefig('sequence_lengths.png')

    return total_tokens, sequence_lengths


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    # dataset = load_dataset("roneneldan/TinyStories", "default", split="train")
    # dataset = load_dataset("allenai/c4", "en", split="train", num_proc=32)

    # data_files = [ f"data/CC-MAIN-2024-10/000_{i:05}.parquet" for i in range(50) ]
    data_files = []
    for i in range(6):
        for j in range(10):
            data_files.append(f"sample/100BT/{i:03}_{j:05}.parquet")

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", data_files=data_files, num_proc=16)
    dataset = dataset['train']

    tokenizer.pad_token = tokenizer.eos_token

    total_tokens, sequence_lengths = count_tokens_in_dataset(dataset, tokenizer)

