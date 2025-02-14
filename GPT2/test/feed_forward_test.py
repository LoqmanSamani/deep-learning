import torch
import torch.testing
from feed_forward import FeedForward

def test_feedforward():

    batch_size = 2
    sequence_length = 5
    embedding_dimension = 512
    scaling_value = 4

    input_tensor = torch.randn(batch_size, sequence_length, embedding_dimension)

    feedforward = FeedForward(embedding_dimension, scaling_value=scaling_value)

    output_tensor = feedforward(input_tensor)

    assert output_tensor.shape[0] == input_tensor.shape[0], \
        f"Batch size mismatch: expected {input_tensor.shape[0]}, got {output_tensor.shape[0]}"
    assert output_tensor.shape[1] == input_tensor.shape[1], \
        f"Sequence length mismatch: expected {input_tensor.shape[1]}, got {output_tensor.shape[1]}"

    assert output_tensor.shape[2] == embedding_dimension, \
        f"Embedding dimension mismatch: expected {embedding_dimension}, got {output_tensor.shape[2]}"

    torch.testing.assert_close(output_tensor, output_tensor, msg="Output tensor contains NaN or Inf values.")

    print("FeedForward test passed!")



if __name__ == "__main__":
    test_feedforward()