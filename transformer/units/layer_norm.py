import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, embedding_dimension=512, epsilon=1e-5):
        super().__init__()

        self.epsilon = epsilon
        self.gamma = torch.nn.parameter.Parameter(torch.ones(embedding_dimension))
        self.beta = torch.nn.parameter.Parameter(torch.zeros(embedding_dimension))

    def forward(self, x):

        mean = x.mean(dim=-1, keepdims=True)
        variance = x.var(dim=-1, keepdims=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(variance + self.epsilon)

        return self.gamma * x_norm + self.beta