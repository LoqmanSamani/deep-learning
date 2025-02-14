import torch
from self_attention import MultiHeadAttention
import torch.testing as testing

def test_multihead_attention():

    batch_size = 2
    num_tokens = 10
    input_dim = 32
    output_dim = 64
    num_heads = 8
    context_length = 512

    x = torch.rand(batch_size, num_tokens, input_dim)
    attention_layer = MultiHeadAttention(input_dim, output_dim, num_heads, context_length)
    output = attention_layer(x)
    expected_shape = (batch_size, num_tokens, output_dim)
    testing.assert_close(output.shape, expected_shape)

    print(f"Output shape is as expected: {output.shape}")


if __name__ == "__main__":
    test_multihead_attention()