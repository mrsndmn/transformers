import time
import re
from transformers import AutoTokenizer
from tokenizers.normalizers import Replace, Sequence


def dummy_normalize_text(text, end_of_sentence_token):
    """
    Dummy Python implementation of the same normalization logic
    """
    # Apply the same patterns as the tokenizer normalizer
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

    normalized_text = text
    for pattern, replacement in patterns:
        normalized_text = re.sub(pattern, replacement, normalized_text)

    return normalized_text


def benchmark_normalization(test_texts, tokenizer, end_of_sentence_token, num_iterations=1000):
    """
    Benchmark the tokenizer normalizer vs dummy implementation
    """
    print(f"Benchmarking with {num_iterations} iterations...")
    print("=" * 50)

    # Benchmark tokenizer normalizer
    start_time = time.time()
    for _ in range(num_iterations):
        for text in test_texts:
            # Use the tokenizer's normalizer directly
            normalized = tokenizer.backend_tokenizer.normalizer.normalize_str(text)
    tokenizer_time = time.time() - start_time

    # Benchmark dummy implementation
    start_time = time.time()
    for _ in range(num_iterations):
        for text in test_texts:
            normalized = dummy_normalize_text(text, end_of_sentence_token)
    dummy_time = time.time() - start_time

    print(f"Tokenizer normalizer time: {tokenizer_time:.4f} seconds")
    print(f"Dummy implementation time: {dummy_time:.4f} seconds")
    print(f"Speedup: {dummy_time / tokenizer_time:.2f}x")
    print(f"Tokenizer is {dummy_time / tokenizer_time:.2f}x faster than dummy implementation")

    return tokenizer_time, dummy_time


def test_normalization_equivalence(test_texts, tokenizer, end_of_sentence_token):
    """
    Test that both implementations produce the same results
    """
    print("\nTesting normalization equivalence...")
    print("=" * 50)

    for i, text in enumerate(test_texts):
        tokenizer_result = tokenizer.backend_tokenizer.normalizer.normalize_str(text)
        dummy_result = dummy_normalize_text(text, end_of_sentence_token)

        print(f"Test {i+1}:")
        print(f"  Original: '{text}'")
        print(f"  Tokenizer: '{tokenizer_result}'")
        print(f"  Dummy: '{dummy_result}'")
        print(f"  Match: {tokenizer_result == dummy_result}")
        assert tokenizer_result == dummy_result, f"Tokenizer and dummy implementations do not match for test {i+1}"
        print()


if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

    end_of_sentence_token = '<end_of_sentence>'
    token_id = tokenizer.add_special_tokens({'additional_special_tokens': [end_of_sentence_token]})
    print(f"Added {end_of_sentence_token} token with ID: {token_id}")

    # Create normalizer sequence
    new_normalizer_sequence = Sequence([
        Replace(pattern='. ', content=f'. {end_of_sentence_token}'),
        Replace(pattern='.\n', content=f'.\n{end_of_sentence_token}'),
        Replace(pattern='? ', content=f'? {end_of_sentence_token}'),
        Replace(pattern='?\n', content=f'?\n{end_of_sentence_token}'),
        Replace(pattern='! ', content=f'! {end_of_sentence_token}'),
        Replace(pattern='!\n', content=f'!\n{end_of_sentence_token}'),
        Replace(pattern=r'\. ', content=f'. {end_of_sentence_token}'),
        Replace(pattern=r'!$', content=f'!{end_of_sentence_token}'),
        Replace(pattern=r'\?$', content=f'?{end_of_sentence_token}'),
    ])

    # Update the tokenizer backend with the modified normalizer
    tokenizer.backend_tokenizer.normalizer = new_normalizer_sequence

    # Test texts for benchmarking
    test_texts = [
        "Hello, how are you? ",
        "This is a test sentence. ",
        "What do you think? ",
        "Amazing! ",
        "This ends with a period.\n",
        "This ends with a question mark?\n",
        "This ends with an exclamation mark!\n",
        "Multiple sentences. Here is another one. And a third one.",
        "Questions? Yes! And more questions?",
    ]

    # Test equivalence first
    test_normalization_equivalence(test_texts, tokenizer, end_of_sentence_token)

    # Run benchmark
    tokenizer_time, dummy_time = benchmark_normalization(test_texts, tokenizer, end_of_sentence_token)

    # Save tokenizer
    tokenizer.save_pretrained("/tmp/tokenizer_with_end_of_sentence_token")
    print(f"\nTokenizer saved to /tmp/tokenizer_with_end_of_sentence_token")

    # Test loaded tokenizer
    tokenizer_loaded = AutoTokenizer.from_pretrained("/tmp/tokenizer_with_end_of_sentence_token")
    result = tokenizer_loaded("Hello, how are you? ")
    print(f"Test result: {result}")