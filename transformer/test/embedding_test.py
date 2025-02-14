import torch
from embedding import Embedding




def test_embedding_layer():

    vocab_size = 10000
    embedding_dim = 512
    context_len = 128
    batch_size = 2
    seq_len = 10

    embedding_layer = Embedding(vocab_size, embedding_dim, context_len)

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = embedding_layer(token_ids)

    assert output.shape == (batch_size, seq_len, embedding_dim), (
        f"Output shape mismatch. Expected {(batch_size, seq_len, embedding_dim)}, "
        f"but got {output.shape}."
    )
    print("Output shape is correct:", output.shape)

    token_embeds, positional_encodings = embedding_layer.token_embedding(token_ids), \
                                         embedding_layer.positional_encoding[:, :seq_len, :]

    expected_output = token_embeds + positional_encodings
    assert torch.allclose(output, expected_output, atol=1e-6), "Output does not match the expected token + positional encoding."

    print("Embedding layer test passed successfully!")

test_embedding_layer()
