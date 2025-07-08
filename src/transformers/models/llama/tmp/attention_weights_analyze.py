import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM


if __name__ == '__main__':

    with torch.no_grad():
        adaptive_llama = AdaptiveLlamaForCausalLM.from_pretrained(
            "./adaptive_slm2_1.7B_pretrain_with_end_of_sentence_token_ftte_w_0.100_l_8-16_8MTABS8F/checkpoint-8000",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        adaptive_llama.to('cuda')

        adaptive_llama.eval()

        tokenizer = AutoTokenizer.from_pretrained("./adaptive_slm2_1.7B_pretrain_with_end_of_sentence_token_ftte_w_0.100_l_8-16_8MTABS8F/checkpoint-8000")

        text = "Family Life: How to choose a wedding dress. Choose an a-line fit for a pear-shape or apple-shape. One of the most important aspects of the dress is the fit. Before you go shopping, you should figure out what fit you are looking for. If your bust is too high, you will end up with a boxy shape that lacks shape and does not allow you to move your head. Also, you are likely to move and check yourself in mirrors to get a better idea of your proportions."
        inputs = tokenizer([text], return_tensors="pt")
        inputs.to('cuda')

        outputs = adaptive_llama(**inputs, output_attentions=True, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        seq_len = inputs['input_ids'].shape[1]

        sentence_level_hidden_states = []

        print("hidden_states", len(hidden_states), [ hs.shape for hs in hidden_states ])

        for i in range(len(hidden_states)):
            hidden_state = hidden_states[i]
            print(i, hidden_state.shape)

            if hidden_state.shape[1] != seq_len:
                sentence_level_hidden_states.append(hidden_state)

                if len(sentence_level_hidden_states) > 1:
                    print(f"mean diff {i}", (sentence_level_hidden_states[-2] - hidden_state).mean())


        for i, attention in enumerate(outputs.attentions):

            attention_mean_heads = attention.mean(dim=1)
            plt.imshow(attention_mean_heads.permute(1, 2, 0).to(torch.float32).cpu().numpy())
            plt.savefig(f"src/transformers/models/llama/tmp/attention_weights_analyze/attention_mean_heads_{i}.png")
            plt.close()

            print(i, attention_mean_heads.shape)

        breakpoint()