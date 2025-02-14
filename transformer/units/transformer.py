import torch
from encoder import Encoder
from decoder import Decoder


class Transformer(torch.nn.Module):
    def __init__(
            self,
            vocabulary_size,
            encoder_num_layers=6,
            decoder_num_layers=6,
            encoder_input_dimension=512,
            decoder_input_dimension=512,
            encoder_output_dimension=512,
            decoder_output_dimension=512,
            encoder_num_heads=8,
            decoder_num_heads=8,
            encoder_context_length=512,
            decoder_context_length=512,
            encoder_dropout_rate=0.1,
            decoder_dropout_rate=0.1,
            encoder_qkv_bias=False,
            decoder_qkv_bias=False,
            encoder_scaling_value=4,
            decoder_scaling_value=4,
            epsilon=1e-5
    ):
        super().__init__()
        self.encoder = Encoder(
            vocabulary_size=vocabulary_size,
            num_layers=encoder_num_layers,
            input_dimension=encoder_input_dimension,
            output_dimension=encoder_output_dimension,
            num_heads=encoder_num_heads,
            context_length=encoder_context_length,
            dropout_rate=encoder_dropout_rate,
            qkv_bias=encoder_qkv_bias,
            scaling_value=encoder_scaling_value,
            epsilon=epsilon
        )

        self.decoder = Decoder(
            vocabulary_size=vocabulary_size,
            num_layers=decoder_num_layers,
            input_dimension=decoder_input_dimension,
            output_dimension=decoder_output_dimension,
            num_heads=decoder_num_heads,
            context_length=decoder_context_length,
            dropout_rate=decoder_dropout_rate,
            qkv_bias=decoder_qkv_bias,
            scaling_value=decoder_scaling_value,
            epsilon=epsilon

        )
        self.linear = torch.nn.Linear(
            in_features=decoder_output_dimension,
            out_features=vocabulary_size,
            bias=False
        )


    def forward(self, encoder_token_ids, decoder_token_ids):
        y = self.encoder(x=encoder_token_ids)
        x = self.decoder(x=decoder_token_ids, y=y)
        logits = self.linear(x)

        return logits

