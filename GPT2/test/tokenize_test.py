import torch.testing
from tokenize_ import Tokenizer

def test_tokenizer_custom_vocab():

    vocab_text = "Hello, world! This is a tokenizer test. Can it handle unknown tokens?"
    test_text = "Hello, tokenizer! Testing unknown words and end of text."

    custom_tokenizer = Tokenizer(
        vocab_text=vocab_text,
        create_vocab=True,
        unk=True,
        end_of_text=True,
        vocab_start=1
    )

    expected_vocab = {
        '!': 1, ',': 2, '.': 3, '?': 4, 'Can': 5, 'Hello': 6,
        'This': 7, 'a': 8, 'handle': 9, 'is': 10, 'it': 11,
        'test': 12, 'tokenizer': 13, 'tokens': 14, 'unknown': 15,
        'world': 16, '<|unk|>': 17, '<|endoftext|>': 18
    }

    torch.testing.assert_close(
        custom_tokenizer.vocab,
        expected_vocab,
        msg="Custom vocabulary does not match the expected values."
    )

    encoded_ids_custom = custom_tokenizer.encode(test_text, use_custom=True)
    expected_encoded = [6, 2, 13, 1, 17, 15, 17, 17, 17, 17, 17, 3]
    torch.testing.assert_close(
        encoded_ids_custom,
        expected_encoded,
        msg="Encoded IDs do not match the expected values."
    )

    decoded_text_custom = custom_tokenizer.decode(encoded_ids_custom, use_custom=True)
    expected_decoded = "Hello , tokenizer ! <|unk|> unknown <|unk|> <|unk|> <|unk|> <|unk|> <|unk|> ."
    assert decoded_text_custom == expected_decoded, \
        f"Decoded text mismatch: expected '{expected_decoded}', got '{decoded_text_custom}'"

    print("Custom vocabulary tokenizer test passed!")


def test_tokenizer_pretrained_gpt2():

    test_text = "Hello, tokenizer! Testing unknown words and end of text."
    pretrained_tokenizer = Tokenizer(encoding="gpt2")
    encoded_ids_gpt2 = pretrained_tokenizer.encode(test_text)
    decoded_text_gpt2 = pretrained_tokenizer.decode(encoded_ids_gpt2)

    assert decoded_text_gpt2 == test_text, \
        f"Decoded text mismatch: expected '{test_text}', got '{decoded_text_gpt2}'"

    print("Pre-trained GPT-2 tokenizer test passed!")



if __name__ == "__main__":
    test_tokenizer_custom_vocab()
    test_tokenizer_pretrained_gpt2()