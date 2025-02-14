import torch
import torch.testing as testing
from transformer import Transformer


def test_transformer():
    batch_size = 2
    context_length = 512
    input_dimension = 512

    model = Transformer(
        input_dimension=input_dimension,
        output_dimension=input_dimension,
        num_heads=8,
        context_length=context_length,
        dropout_rate=0.1,
        qkv_bias=True,
        layer_norm_epsilon=1e-5,
        ff_scaling_value=4
    )
    dummy_input = torch.randn(batch_size, context_length, input_dimension)
    output = model(dummy_input)
    expected_shape = (batch_size, context_length, input_dimension)
    testing.assert_close(output.shape, expected_shape)

    print(f"Test passed. Output shape: {output.shape}")



if __name__ == "__main__":
    test_transformer()