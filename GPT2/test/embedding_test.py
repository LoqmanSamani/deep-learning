import torch
import torch.testing
from embedding import Embedding

def test_embedding_layer():

    vocabulary_size = 50
    embedding_dimension = 50
    context_length = 50

    embedding_layer = Embedding(vocabulary_size, embedding_dimension, context_length)
    token_ids = torch.tensor([
        [4, 8, 3, 1, 2],
        [4, 9, 7, 5, 3]
    ])

    output = embedding_layer(token_ids)
    expected_shape = (token_ids.shape[0], token_ids.shape[1], embedding_dimension)

    assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
    torch.testing.assert_close(output, output, msg="Embedding output contains invalid values.")

    print("Embedding test passed!")



if __name__ == "__main__":
    test_embedding_layer()