import torch
from attention import MultiHeadAttention as mha
from feedforward import FeedForward as ff
from layer_norm import LayerNorm as ln
from embedding import Embedding as embed



class Decoder(torch.nn.Module):
    def __init__(
        self,
        vocabulary_size,
        num_layers=6,
        input_dimension=512,
        output_dimension=512,
        num_heads=8,
        context_length=512,
        dropout_rate=0.1,
        qkv_bias=False,
        scaling_value=4,
        epsilon=1e-5
    ):
        super().__init__()
        self.embedding = embed(
            vocabulary_size=vocabulary_size,
            embedding_dimension=input_dimension,
            context_length=context_length
        )
        self.layers = torch.nn.ModuleList([
            DecoderLayer(
                input_dimension=input_dimension,
                output_dimension=output_dimension,
                num_heads=num_heads,
                context_length=context_length,
                dropout_rate=dropout_rate,
                qkv_bias=qkv_bias,
                scaling_value=scaling_value,
                epsilon=epsilon
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, y=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, y)
        return x




class DecoderLayer(torch.nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        num_heads,
        context_length,
        dropout_rate,
        qkv_bias,
        scaling_value,
        epsilon
    ):
        super().__init__()
        self.self_attention = mha(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            num_heads=num_heads,
            context_length=context_length,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias
        )
        self.cross_attention = mha(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            num_heads=num_heads,
            context_length=context_length,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias
        )
        self.feedforward = ff(
            embedding_dimension=input_dimension,
            scaling_value=scaling_value
        )
        self.norm1 = ln(
            embedding_dimension=input_dimension,
            epsilon=epsilon
        )
        self.norm2 = ln(
            embedding_dimension=input_dimension,
            epsilon=epsilon
        )
        self.norm3 = ln(
            embedding_dimension=input_dimension,
            epsilon=epsilon
        )

    def forward(self, x, y):

        residual = x
        x = self.self_attention(x=x, apply_mask=True)
        x = self.norm1(residual + x)

        if y is not None:
            residual = x
            x = self.cross_attention(x=x, y=y)
            x = self.norm2(residual + x)

        residual = x
        x = self.feedforward(x)
        x = self.norm3(residual + x)

        return x
