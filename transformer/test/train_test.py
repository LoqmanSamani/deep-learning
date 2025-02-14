import torch
from torch.utils.data import Dataset
from transformer import Transformer
from train import Train



class MockDataset(Dataset):
    def __init__(self, vocab_size, num_samples, seq_len):

        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        encoder_inputs = torch.randint(1, self.vocab_size, (self.seq_len,))
        decoder_inputs = torch.randint(1, self.vocab_size, (self.seq_len,))
        labels = torch.randint(1, self.vocab_size, (self.seq_len,))
        return encoder_inputs, decoder_inputs, labels


def test_train_class():

    vocab_size = 100
    seq_len = 10
    num_samples = 100
    batch_size = 8
    num_epochs = 30
    learning_rate = 0.05

    train_dataset = MockDataset(vocab_size, num_samples, seq_len)
    val_dataset = MockDataset(vocab_size, num_samples // 2, seq_len)

    model = Transformer(
        vocabulary_size=vocab_size,
        encoder_num_layers=2,
        decoder_num_layers=2,
        encoder_input_dimension=32,
        decoder_input_dimension=32,
        encoder_output_dimension=32,
        decoder_output_dimension=32,
        encoder_num_heads=2,
        decoder_num_heads=2,
        encoder_context_length=seq_len,
        decoder_context_length=seq_len
    )
    trainer = Train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    print("Starting test training...")
    trainer.train()
    print("Test training complete.")
    sample_encoder_inputs = torch.randint(1, vocab_size, (1, seq_len))
    sample_decoder_inputs = torch.randint(1, vocab_size, (1, seq_len))
    model.eval()
    with torch.no_grad():
        outputs = model(encoder_token_ids=sample_encoder_inputs, decoder_token_ids=sample_decoder_inputs)
        assert outputs.size() == (1, seq_len, vocab_size), "Model output shape mismatch!"

    print("Test passed: Model outputs are valid and training ran successfully.")




if __name__ == "__main__":
    test_train_class()
