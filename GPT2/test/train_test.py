import torch
import torch.testing as testing
from gpt2 import GPT2
from train_ import Train


def test_gpt2_training():

    batch_size = 4
    seq_length = 256
    vocabulary_size = 50257
    train_ratio = 0.90
    num_epochs = 2

    model = GPT2(
        input_dimension=768,
        output_dimension=768,
        num_heads=12,
        context_length=seq_length,
        dropout_rate=0.1,
        qkv_bias=False,
        layer_norm_epsilon=1e-5,
        ff_scaling_value=4,
        num_transformers=12,
        vocabulary_size=vocabulary_size,
        use_custom=False
    )

    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]


    train = Train(
        model=model,
        train_data=train_data,
        validation_data=val_data,
        batch_size=2,
        num_epochs=num_epochs,
        max_context_length=256,
        stride=256,
        print_sample=True
    )

    train_losses, validation_losses, track_tokens_seen = train()

    assert isinstance(train_losses, list) and all(
        isinstance(x, float) for x in train_losses), "Train losses must be a list of floats."
    assert isinstance(validation_losses, list) and all(
        isinstance(x, float) for x in validation_losses), "Validation losses must be a list of floats."

    assert len(track_tokens_seen) == len(train_losses), "Tracked tokens must match the number of training loss steps."

    if len(train_losses) > 1:
        assert train_losses[-1] <= train_losses[0], "Training loss should decrease over time."
    if len(validation_losses) > 1:
        assert validation_losses[-1] <= validation_losses[0], "Validation loss should decrease over time."

    print("Test passed: Training process is functioning correctly.")


if __name__ == "__main__":
    test_gpt2_training()
