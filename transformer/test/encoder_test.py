import torch
from torch.testing import assert_allclose
from encoder import Encoder

def test_encoder():

    batch_size = 2
    context_length = 512
    vocabulary_size = 10000
    input_dimension = 512
    output_dimension = 512
    num_layers = 6
    num_heads = 8
    dropout_rate = 0.1
    qkv_bias = False
    scaling_value = 4
    epsilon = 1e-5

    dummy_input = torch.randint(0, vocabulary_size, (batch_size, context_length))
    encoder = Encoder(
        vocabulary_size=vocabulary_size,
        num_layers=num_layers,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        num_heads=num_heads,
        context_length=context_length,
        dropout_rate=dropout_rate,
        qkv_bias=qkv_bias,
        scaling_value=scaling_value,
        epsilon=epsilon
    )


    encoder_output = encoder(dummy_input)
    assert encoder_output.shape == (batch_size, context_length, output_dimension), (
        f"Expected output shape {(batch_size, context_length, output_dimension)}, "
        f"but got {encoder_output.shape}."
    )
    assert encoder_output.dtype == torch.float32, (
        f"Expected output dtype torch.float32, but got {encoder_output.dtype}."
    )

    encoder_output.sum().backward()
    for param in encoder.parameters():
        assert param.grad is not None, "Gradient not computed for some parameters."

    torch.manual_seed(42)
    encoder.eval()
    output1 = encoder(dummy_input)
    output2 = encoder(dummy_input)
    assert_allclose(output1, output2, atol=1e-5, rtol=1e-5, msg="Outputs are not consistent in evaluation mode.")

    print("All tests passed for the Encoder class.")



test_encoder()
