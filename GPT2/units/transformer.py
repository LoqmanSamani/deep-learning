import torch
from feed_forward import FeedForward
from self_attention import MultiHeadAttention
from layer_normalization import LayerNorm


class Transformer(torch.nn.Module):
    def __init__(
            self,
            input_dimension=512,
            output_dimension=512,
            num_heads=8,
            context_length=512,
            dropout_rate=0.1,
            qkv_bias=False,
            layer_norm_epsilon=1e-5,
            ff_scaling_value=4
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            num_heads=num_heads,
            context_length=context_length,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias
        )
        self.layer_norm1 = LayerNorm(
            embedding_dimension=input_dimension,
            epsilon=layer_norm_epsilon
        )
        self.layer_norm2 = LayerNorm(
            embedding_dimension=input_dimension,
            epsilon=layer_norm_epsilon
        )
        self.feed_forward = FeedForward(
            embedding_dimension=input_dimension,
            scaling_value=ff_scaling_value
        )
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        

    def forward(self, x):

        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut
        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
