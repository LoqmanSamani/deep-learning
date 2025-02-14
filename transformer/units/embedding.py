import torch
import math




class Embedding(torch.nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_dimension=512,
        context_length=512
    ):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension
        )
        self.embedding_dimension = embedding_dimension
        self.context_length = context_length

        self.register_buffer("positional_encoding", self._generate_positional_encoding(context_length))

    def _generate_positional_encoding(self, seq_len):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dimension, 2, dtype=torch.float) *
                             -(math.log(10000.0) / self.embedding_dimension))

        pos_enc = torch.zeros((seq_len, self.embedding_dimension), device=position.device)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        return pos_enc.unsqueeze(0)

    def forward(self, token_ids):
        assert token_ids.dim() == 2, "Input token_ids should be of shape (batch_size, seq_len)"

        token_embedded = self.token_embedding(token_ids)
        seq_len = token_ids.size(1)

        if seq_len > self.context_length:
            position_encoded = self._generate_positional_encoding(seq_len).to(token_embedded.device)
        else:
            position_encoded = self.positional_encoding[:, :seq_len, :].to(token_embedded.device)

        return token_embedded + position_encoded



