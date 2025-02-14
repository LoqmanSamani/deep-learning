import torch
import torch.testing as testing
from layer_normalization import LayerNorm




def test_layer_norm():

    batch_size = 2
    seq_length = 5
    embedding_dimension = 4
    epsilon = 1e-5

    layer_norm = LayerNorm(embedding_dimension, epsilon)
    x = torch.randn(batch_size, seq_length, embedding_dimension)

    print("Input tensor:")
    print(x)

    output = layer_norm(x)
    print("\nOutput tensor after LayerNorm:")
    print(output)

    mean = output.mean(dim=-1)
    std = output.std(dim=-1)

    testing.assert_close(mean, torch.zeros_like(mean), atol=1e-6, rtol=0)
    testing.assert_close(std, torch.ones_like(std), atol=1e-6, rtol=0)

    print("\nTest passed: LayerNorm works as expected.")




if __name__ == "__main__":
    test_layer_norm()
