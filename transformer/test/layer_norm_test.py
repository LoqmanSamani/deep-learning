import torch
from layer_norm import LayerNorm


def test_layer_norm():
    batch_size = 2
    seq_len = 5
    embedding_dim = 512

    layer_norm = LayerNorm(embedding_dimension=embedding_dim)

    x = torch.randn(batch_size, seq_len, embedding_dim)

    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(variance + layer_norm.epsilon)
    x_norm = (x - mean) / std

    assert torch.allclose(x_norm.mean(dim=-1), torch.zeros_like(x_norm.mean(dim=-1)), atol=1e-5), (
        "Mean of normalized output is not close to zero!"
    )
    output = layer_norm(x)
    assert output.shape == x.shape, "Output shape mismatch!"

    print("LayerNorm test passed successfully!")


test_layer_norm()
