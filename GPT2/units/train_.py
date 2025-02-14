import torch
from tokenize_ import Tokenizer
from gpt2 import GPT2
from data_loader import DataLoader



class Train(torch.nn.Module):
    def __init__(
            self,
            model,
            train_data,
            validation_data,
            batch_size,
            num_epochs,
            max_context_length,
            stride,
            top_k=1,
            temperature=0,
            evaluation_frequency=5,
            evaluation_iterations=5, # number of batches their losses will be calculated to evaluate the model
            print_sample=False, # if True after each epoch the quality of the model generation samples will be examined.
            max_new_tokens=50,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            device=None,
            learning_rate=1e-5,
            weight_decay=0.1,
            vocab_text=None,
            create_vocab=False,
            encoding="gpt2",
            unk=False,
            end_of_text=False,
            vocab_start=1,
            use_custom=False,
            start_context="Education is not the learning of",
            eos_id=False

    ):
        super().__init__()
        train_obj = DataLoader(
            data=train_data,
            batch_size=batch_size,
            max_length=max_context_length,
            stride=stride,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            vocab_text=vocab_text,
            create_vocab=create_vocab,
            encoding=encoding,
            unk=unk,
            end_of_text=end_of_text,
            vocab_start=vocab_start,
            use_custom=use_custom
        )
        validation_obj = DataLoader(
            data=validation_data,
            batch_size=batch_size,
            max_length=max_context_length,
            stride=stride,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            vocab_text=vocab_text,
            create_vocab=create_vocab,
            encoding=encoding,
            unk=unk,
            end_of_text=end_of_text,
            vocab_start=vocab_start,
            use_custom=use_custom
        )
        self.tokenizer = Tokenizer(
            vocab_text=vocab_text,
            create_vocab=create_vocab,
            encoding=encoding,
            unk=unk,
            end_of_text=end_of_text,
            vocab_start=vocab_start
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.train_loader = train_obj.create_dataloader()
        self.validation_loader = validation_obj.create_dataloader()
        self.model = model
        self.num_epochs = num_epochs
        self.use_custom = use_custom
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_context = start_context
        self.evaluation_frequency = evaluation_frequency
        self.evaluation_iterations = evaluation_iterations
        self.print_sample = print_sample
        self.top_k = top_k
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.eos_id = eos_id  # stops generating early if end-of-sequence token is encountered.


    def model_evaluation(self, data_loader, num_batches):

        total_loss = 0.0

        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                logits = self.model(input_batch)

                loss = torch.nn.functional.cross_entropy(
                    input=logits.flatten(0, 1),
                    target=target_batch.flatten()
                )
                total_loss += loss.item()
            else:
                break

        return total_loss / num_batches



    def generate_sample(self):
        self.model.eval()
        # context_size = self.model.position_embed.weight.shape[0]
        ids = self.tokenizer.encode(
            text=self.start_context,
            use_custom=self.use_custom
        )
        ids = torch.tensor(ids).unsqueeze(0).to(self.device)

        context_size = self.model.embedding.pos_encod.weight.shape[0]
        for _ in range(self.max_new_tokens):
            ids = ids[:, -context_size:]
            with torch.no_grad():
                logits = self.model(ids)
            logits = logits[:, -1, :]
            top_logits, _ = torch.topk(logits, self.top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                condition=logits < min_val,
                input=torch.tensor(float('-inf')).to(logits.device),
                other=logits
            )
            if self.temperature > 0.0:
                logits = logits / self.temperature
                probs = torch.softmax(logits, dim=-1)
                ids_next = torch.multinomial(probs, num_samples=1)
            else:
                ids_next = torch.argmax(logits, dim=-1, keepdim=True)
            if ids_next == self.eos_id:
                break
            ids = torch.cat((ids, ids_next), dim=1)

        ids = ids.squeeze(0)
        generated_text = self.tokenizer.decode(
            ids=ids.tolist(),
            use_custom=self.use_custom
        )

        print(generated_text.replace("\n", " "))
        self.model.train()


    def forward(self):

        train_losses = []
        validation_losses = []
        track_tokens_seen = []
        tokens_seen = 0
        num_iterations = -1

        for epoch in range(self.num_epochs):

            self.model.train()
            for input_batch, target_batch in self.train_loader:

                self.optimizer.zero_grad()

                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                logits = self.model(input_batch)

                loss = torch.nn.functional.cross_entropy(
                    input=logits.flatten(0, 1),
                    target=target_batch.flatten()
                )
                loss.backward()
                self.optimizer.step()
                tokens_seen += input_batch.numel()
                num_iterations += 1

                if num_iterations % self.evaluation_frequency == 0:
                    self.model.eval()
                    with torch.no_grad():
                        train_loss = self.model_evaluation(
                            data_loader=self.train_loader,
                            num_batches=self.evaluation_iterations
                        )
                        validation_loss = self.model_evaluation(
                            data_loader=self.validation_loader,
                            num_batches=self.evaluation_iterations
                        )
                    train_losses.append(train_loss)
                    validation_losses.append(validation_loss)
                    self.model.train()
                    track_tokens_seen.append(tokens_seen)
                    print(
                        f"Epoch {epoch + 1}; Step {num_iterations:06d}: "
                        f"Train loss {train_loss:.3f}, "
                        f"Validation loss {validation_loss:.3f}"
                    )

            if self.print_sample:
                self.generate_sample()

        return train_losses, validation_losses, track_tokens_seen
