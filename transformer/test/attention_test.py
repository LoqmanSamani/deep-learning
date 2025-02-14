import torch
from attention import MultiHeadAttention

def test_multi_head_attention():
    batch_size = 2
    sequence_length = 16
    input_dimension = 512
    output_dimension = 512
    num_heads = 8
    dropout_rate = 0.1
    context_length = 512

    mha = MultiHeadAttention(
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        num_heads=num_heads,
        context_length=context_length,
        dropout_rate=dropout_rate,
        qkv_bias=True
    )
    x = torch.randn(batch_size, sequence_length, input_dimension)

    output = mha(x)

    assert output.shape == (batch_size, sequence_length, output_dimension), (
        f"Output shape mismatch. Expected {(batch_size, sequence_length, output_dimension)}, "
        f"but got {output.shape}."
    )
    print("Output shape is correct:", output.shape)

    for seq_len in [1, 8, 32]:
        x = torch.randn(batch_size, seq_len, input_dimension)
        output = mha(x)
        assert output.shape == (batch_size, seq_len, output_dimension), (
            f"Output shape mismatch for sequence length {seq_len}. "
            f"Expected {(batch_size, seq_len, output_dimension)}, but got {output.shape}."
        )
        print(f"Test passed for sequence length {seq_len}")

    for batch in [1, 4, 8]:
        x = torch.randn(batch, sequence_length, input_dimension)
        output = mha(x)
        assert output.shape == (batch, sequence_length, output_dimension), (
            f"Output shape mismatch for batch size {batch}. "
            f"Expected {(batch, sequence_length, output_dimension)}, but got {output.shape}."
        )
        print(f"Test passed for batch size {batch}")

    print("All tests passed!")

test_multi_head_attention()