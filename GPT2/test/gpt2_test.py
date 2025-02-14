import torch
import torch.testing as testing
from gpt2 import GPT2

def test_gpt2_model():

    batch_size = 4
    seq_length = 16
    vocabulary_size = 50257

    x = torch.randint(0, vocabulary_size, (batch_size, seq_length))

    model = GPT2(
        input_dimension=1024,
        output_dimension=1024,
        num_heads=16,
        context_length=seq_length,
        dropout_rate=0.1,
        qkv_bias=False,
        layer_norm_epsilon=1e-5,
        ff_scaling_value=4,
        num_transformers=24,
        vocabulary_size=vocabulary_size,
        use_custom=False
    )
    logits = model(x)

    expected_shape = (batch_size, seq_length, vocabulary_size)
    testing.assert_close(logits.shape, expected_shape)

    print(f"Logits shape is as expected: {logits.shape}")



if __name__ == "__main__":
    test_gpt2_model()