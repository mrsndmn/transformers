import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
from transformers.models.llama.modeling_sentence_llama import SentenceLlamaForCausalLM


if __name__ == '__main__':

    checkpoint_path = "./sentence_slm2_1.7B_pretrain_with_end_of_sentence_full_VYE9JVA0/checkpoint-6000"

    with torch.no_grad():
        adaptive_llama = SentenceLlamaForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        # adaptive_llama.config._attn_implementation = "sentence_attention"
        adaptive_llama.config._attn_implementation = "eager"

        adaptive_llama.to('cuda')

        adaptive_llama.eval()

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        text = "Family Life: How to choose a wedding dress. Choose an a-line fit for a pear-shape or apple-shape."
        inputs = tokenizer([text], return_tensors="pt")
        inputs.to('cuda')

        outputs = adaptive_llama(**inputs, output_attentions=True, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        seq_len = inputs['input_ids'].shape[1]

        sentence_level_hidden_states = []

        print("hidden_states", len(hidden_states), [ hs.shape for hs in hidden_states ])

        for i in range(len(hidden_states)):
            hidden_state = hidden_states[i]

            sentence_level_hidden_states.append(hidden_state)

            if len(sentence_level_hidden_states) > 1:
                print(f"mean diff {i}", (sentence_level_hidden_states[-2] - hidden_state).mean())

        for i, attention in enumerate(outputs.attentions):

            attention_mean_heads = attention.mean(dim=1)
            plt.imshow(attention_mean_heads.permute(1, 2, 0).to(torch.float32).cpu().numpy())
            plt.title(f"Layer {i} mean attention")
            plt.tight_layout()
            plt.savefig(f"src/transformers/models/llama/tmp/attention_weights_analyze/attention_mean_heads_{i}.png")
            plt.close()

            if i == 2:
                breakpoint()

            # print(i, attention_mean_heads.shape)
