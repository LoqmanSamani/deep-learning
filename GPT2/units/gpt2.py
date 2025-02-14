import torch
from transformer import Transformer
from embedding import Embedding
from layer_normalization import LayerNorm



class GPT2(torch.nn.Module):
    def __init__(
            self,
            input_dimension=1024,
            output_dimension=1024,
            num_heads=16,
            context_length=1024,
            dropout_rate=0.1,
            qkv_bias=False,
            layer_norm_epsilon=1e-5,
            ff_scaling_value=4,
            num_transformers=24,
            vocabulary_size=50257,
            use_custom=False

    ):

        super().__init__()

        self.use_custom = use_custom

        self.embedding = Embedding(
            vocabulary_size=vocabulary_size,
            embedding_dimension=input_dimension,
            context_length=context_length
        )
        self.transformers = torch.nn.Sequential(
            *[Transformer(
                input_dimension=input_dimension,
                output_dimension=output_dimension,
                num_heads=num_heads,
                context_length=context_length,
                dropout_rate=dropout_rate,
                qkv_bias=qkv_bias,
                layer_norm_epsilon=layer_norm_epsilon,
                ff_scaling_value=ff_scaling_value
            ) for _ in range(num_transformers)]
        )
        self.final_layer_norm = LayerNorm(
            embedding_dimension=input_dimension,
            epsilon=layer_norm_epsilon
        )
        self.linear_output = torch.nn.Linear(
            in_features=input_dimension,
            out_features=vocabulary_size,
            bias=False
        )
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):

        x = self.embedding(x)
        x = self.dropout(x)
        x = self.transformers(x)
        x = self.final_layer_norm(x)
        logits = self.linear_output(x)

        return logits









