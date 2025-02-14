import torch
from transformer import Transformer



def test_transformer():

    vocabulary_size = 10000
    encoder_context_length = 128
    decoder_context_length = 128
    batch_size = 32

    model = Transformer(
        vocabulary_size=vocabulary_size,
        encoder_num_layers=6,
        decoder_num_layers=6,
        encoder_input_dimension=512,
        decoder_input_dimension=512,
        encoder_output_dimension=512,
        decoder_output_dimension=512,
        encoder_num_heads=8,
        decoder_num_heads=8,
        encoder_context_length=encoder_context_length,
        decoder_context_length=decoder_context_length,
        encoder_dropout_rate=0.1,
        decoder_dropout_rate=0.1,
        encoder_qkv_bias=False,
        decoder_qkv_bias=False,
        encoder_scaling_value=4,
        decoder_scaling_value=4,
        epsilon=1e-5
    )

    encoder_token_ids = torch.randint(
        low=0, high=vocabulary_size, size=(batch_size, encoder_context_length)
    )
    decoder_token_ids = torch.randint(
        low=0, high=vocabulary_size, size=(batch_size, decoder_context_length)
    )

    logits = model(encoder_token_ids, decoder_token_ids)

    assert logits.shape == (batch_size, decoder_context_length, vocabulary_size), \
        f"Expected output shape {(batch_size, decoder_context_length, vocabulary_size)}, but got {logits.shape}"

    print("Transformer test passed! Output shape:", logits.shape)




test_transformer()
