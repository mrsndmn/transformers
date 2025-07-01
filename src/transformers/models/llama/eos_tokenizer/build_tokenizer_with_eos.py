from transformers import AutoTokenizer

from tokenizers.normalizers import Replace
from tokenizers.normalizers import Sequence

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

    end_of_sentence_token = '<end_of_sentence>'
    token_id = tokenizer.add_special_tokens({'additional_special_tokens': [end_of_sentence_token]})
    print(f"Added {end_of_sentence_token} token with ID: {token_id}")

    new_normalizer_sequence = Sequence(
        [
            # Adding ReplaceNormalizer to replace '@' with '[AT]'
            Replace(pattern='. ',  content=f'. {end_of_sentence_token}'),
            Replace(pattern='.\n', content=f'.\n{end_of_sentence_token}'),
            Replace(pattern='? ',  content=f'? {end_of_sentence_token}'),
            Replace(pattern='?\n', content=f'?\n{end_of_sentence_token}'),
            Replace(pattern='! ',  content=f'! {end_of_sentence_token}'),
            Replace(pattern='!\n', content=f'!\n{end_of_sentence_token}'),
            Replace(pattern=r'\. ',  content=f'. {end_of_sentence_token}'),
            Replace(pattern=r'!$',   content=f'!{end_of_sentence_token}'),
            Replace(pattern=r'\?$',  content=f'?{end_of_sentence_token}'),
        ]
    )

    # Update the tokenizer backend with the modified normalizer
    tokenizer.backend_tokenizer.normalizer = new_normalizer_sequence

    tokenizer.save_pretrained("/tmp/tokenizer_with_end_of_sentence_token")

    tokenizer_loaded = AutoTokenizer.from_pretrained("/tmp/tokenizer_with_end_of_sentence_token")

    result = tokenizer_loaded("Hello, how are you? ")

    breakpoint()