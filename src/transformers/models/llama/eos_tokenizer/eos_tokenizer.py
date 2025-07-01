from transformers import PreTrainedTokenizer
import re
from typing import Any

class EOSTokenizer(PreTrainedTokenizer):

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        tokenizer = super().from_pretrained(*args, **kwargs)

        tokenizer.end_of_sentence_token = '<end_of_sentence>'
        if tokenizer.end_of_sentence_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [tokenizer.end_of_sentence_token]})
            print(f"Added <end_of_sentence> token with ID: {tokenizer.convert_tokens_to_ids(tokenizer.end_of_sentence_token)}")

        tokenizer.end_of_sentence_token_id = tokenizer.convert_tokens_to_ids(tokenizer.end_of_sentence_token)

        return tokenizer

    def prepare_for_tokenization(
        self, text: str, is_split_into_words: bool = False, **kwargs
    ) -> tuple[str, dict[str, Any]]:

        end_of_sentence_token = self.end_of_sentence_token
        patterns = [
            (r'\. ', f'. {end_of_sentence_token}'),
            (r'\.\n', f'.\n{end_of_sentence_token}'),
            (r'\? ', f'? {end_of_sentence_token}'),
            (r'\?\n', f'?\n{end_of_sentence_token}'),
            (r'! ', f'! {end_of_sentence_token}'),
            (r'!\n', f'!\n{end_of_sentence_token}'),
            (r'\. ', f'. {end_of_sentence_token}'),
            (r'!$', f'!{end_of_sentence_token}'),
            (r'\?$', f'?{end_of_sentence_token}'),
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)

        return (text, kwargs)


if __name__ == "__main__":
    tokenizer = EOSTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

    assert tokenizer.encode("Hello, how are you? ")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer.encode("This is a test sentence. ")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer.encode("What do you think? ")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer.encode("Amazing! ")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer.encode("This ends with a period.\n")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer.encode("This ends with a question mark?\n")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer.encode("This ends with an exclamation mark!\n")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer.encode("Multiple sentences. Here is another one. And a third one.")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer.encode("Questions? Yes! And more questions?")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
