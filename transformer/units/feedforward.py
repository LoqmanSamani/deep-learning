import torch



class FeedForward(torch.nn.Module):
    def __init__(self, embedding_dimension=512, scaling_value=4):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension * scaling_value,
                bias=True
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=embedding_dimension * scaling_value,
                out_features=embedding_dimension,
                bias=True
            )
        )

    def forward(self, x):

        return self.layers(x)

