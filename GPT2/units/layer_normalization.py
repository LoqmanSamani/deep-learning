import torch


class LayerNorm(torch.nn.Module):
    """
    Layer Normalization (LayerNorm) for normalizing inputs across the features dimension.

    This implementation normalizes the input tensor `x` along the last dimension (features)
    by subtracting the mean and dividing by the standard deviation. It also applies a learnable
    scaling and shifting transformation (gamma and beta) to the normalized tensor.

    Args:
        embedding_dimension (int): The size of the feature dimension (e.g., number of features in the input).
        epsilon (float, optional): A small constant added to the variance for numerical stability (default: 1e-5).

    Attributes:
        scale (torch.nn.Parameter): Learnable parameter that scales the normalized output.
        shift (torch.nn.Parameter): Learnable parameter that shifts the normalized output.
    """

    def __init__(self, embedding_dimension, epsilon=1e-5):
        """
        Initializes the LayerNorm layer with learnable scaling and shifting parameters.

        Args:
            embedding_dimension (int): The size of the feature dimension to normalize over.
            epsilon (float): Small value to avoid division by zero in variance computation.
        """
        super().__init__()

        # Initialize epsilon, scale (gamma), and shift (beta) parameters
        self.epsilon = epsilon
        self.scale = torch.nn.parameter.Parameter(torch.ones(embedding_dimension))  # Scaling parameter (gamma)
        self.shift = torch.nn.parameter.Parameter(torch.zeros(embedding_dimension))  # Shifting parameter (beta)

    def forward(self, x):
        """
        Forward pass through the LayerNorm layer.

        This function normalizes the input tensor `x` along the last dimension (features),
        and applies the learnable scale and shift transformations.

        Args:
            x (torch.Tensor): The input tensor to normalize. Expected shape:
                              (batch_size, num_tokens, embedding_dimension)

        Returns:
            torch.Tensor: The normalized and scaled tensor with the same shape as the input tensor.
        """
        # Calculate mean and variance along the last dimension (features)
        mean = x.mean(dim=-1, keepdims=True)  # Mean of the features (along last dimension)
        variance = x.var(dim=-1, keepdims=True, unbiased=False)  # Variance of the features (along last dimension)

        # Normalize the input (subtract mean and divide by std deviation)
        x_norm = (x - mean) / torch.sqrt(variance + self.epsilon)  # Normalize with added epsilon for stability

        # Apply learnable scale (gamma) and shift (beta) to the normalized input
        # Scaling: multiplication by scale (gamma)
        # Shifting: adding shift (beta)
        return self.scale * x_norm + self.shift


