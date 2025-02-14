import torch
from feedforward import FeedForward



def test_feedforward():
    batch_size = 2
    seq_len = 5
    embedding_dim = 512
    scaling_value = 4

    feedforward = FeedForward(embedding_dimension=embedding_dim, scaling_value=scaling_value)
    x = torch.randn(batch_size, seq_len, embedding_dim)
    output = feedforward(x)

    assert output.shape == (batch_size, seq_len, embedding_dim), "Output shape mismatch!"
    assert output.dtype == x.dtype, "Output dtype mismatch!"

    print("FeedForward test passed successfully!")



test_feedforward()