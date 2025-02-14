import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Train:
    def __init__(
            self,
            model,
            train_dataset,
            val_dataset=None,
            batch_size=32,
            learning_rate=1e-4,
            num_epochs=10,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            gradient_clipping=1.0
    ):

        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.device = device
        self.gradient_clipping = gradient_clipping


    def train_one_epoch(self, epoch):

        self.model.train()
        epoch_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for batch in progress_bar:
            encoder_inputs, decoder_inputs, labels = batch
            encoder_inputs = encoder_inputs.to(self.device)
            decoder_inputs = decoder_inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(
                encoder_token_ids=encoder_inputs,
                decoder_token_ids=decoder_inputs
            )
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)

            loss = self.criterion(outputs, labels)
            epoch_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

            self.optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        # avg_loss = epoch_loss / len(self.train_loader)
        # print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")

    def validate(self):

        if not self.val_loader:
            print("No validation dataset provided.")
            return
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                encoder_inputs, decoder_inputs, labels = batch
                encoder_inputs = encoder_inputs.to(self.device)
                decoder_inputs = decoder_inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(
                    encoder_token_ids=encoder_inputs,
                    decoder_token_ids=decoder_inputs
                )
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)

                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        # print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def train(self):

        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            if self.val_loader:
                self.validate()
