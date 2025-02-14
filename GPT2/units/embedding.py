import torch


class Embedding(torch.nn.Module):
    """
    A combined embedding layer for token embeddings and positional encodings.

    This class provides a mechanism to map token IDs to learned embeddings
    and add positional encodings to account for the order of tokens in a sequence.
    Both the token embeddings and positional encodings are learnable parameters.

    Methods:
        positional_encoding(token_ids):
            Computes the positional encodings based on token positions in the sequence.
        token_embedding(token_ids):
            Computes the token embeddings based on token IDs.
        forward(token_ids):
            Combines token embeddings and positional encodings to produce the final embeddings.
    """

    def __init__(self, vocabulary_size, embedding_dimension, context_length):
        """
        Initializes the embedding layer with token and positional embeddings.

        Args:
            vocabulary_size (int): Total number of unique tokens in the vocabulary.
            embedding_dimension (int): Dimensionality of each embedding vector.
            context_length (int): Maximum sequence length supported by positional encodings.
        """
        super().__init__()
        # Embedding layer for token IDs. Maps token indices to learnable embedding vectors.
        self.tok_embed = torch.nn.Embedding(
            num_embeddings=vocabulary_size,  # Vocabulary size (total unique tokens).
            embedding_dim=embedding_dimension  # Size of each embedding vector.
        )

        # Embedding layer for positional encodings. Maps positions to learnable embedding vectors.
        self.pos_encod = torch.nn.Embedding(
            num_embeddings=context_length,  # Maximum sequence length supported.
            embedding_dim=embedding_dimension  # Size of each positional encoding vector.
        )

    def positional_encoding(self, token_ids):
        """
        Computes the positional encodings for a given sequence of token IDs.

        Args:
            token_ids (Tensor): A tensor of shape (batch_size, seq_length) containing token IDs.

        Returns:
            Tensor: A tensor of shape (seq_length, embedding_dimension) representing positional encodings.
        """
        # Generate positional indices for the sequence (0, 1, 2, ..., seq_length-1).
        positional_indices = torch.arange(token_ids.shape[-1], device=token_ids.device).unsqueeze(0)
        # Look up the learnable positional encodings for these indices.
        return self.pos_encod(positional_indices)

    def token_embedding(self, token_ids):
        """
        Computes the token embeddings for a given sequence of token IDs.

        Args:
            token_ids (Tensor): A tensor of shape (batch_size, seq_length) containing token IDs.

        Returns:
            Tensor: A tensor of shape (batch_size, seq_length, embedding_dimension) representing token embeddings.
        """
        # Look up the learnable token embeddings for the provided token IDs.
        return self.tok_embed(token_ids)

    def forward(self, token_ids):
        """
        Combines token embeddings and positional encodings to produce final embeddings.

        Args:
            token_ids (Tensor): A tensor of shape (batch_size, seq_length) containing token IDs.

        Returns:
            Tensor: A tensor of shape (batch_size, seq_length, embedding_dimension) where each token embedding
                    is enriched with positional encoding.
        """
        # Get token embeddings using the token IDs.
        token_embeds = self.token_embedding(token_ids=token_ids)

        # Get positional encodings for the sequence positions.
        position_embeds = self.positional_encoding(token_ids=token_ids)

        # Combine the token embeddings and positional encodings.
        return token_embeds + position_embeds

