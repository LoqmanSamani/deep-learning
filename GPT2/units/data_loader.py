import torch
from tokenize_ import Tokenizer


class DataLoader(torch.utils.data.DataLoader):
    """
    Custom DataLoader for managing dataset batching and processing.

    Args:
        data (str): Input text data to be tokenized and processed into datasets.
        tokenizer (Tokenizer): An instance of the Tokenizer class for text encoding.
        batch_size (int): Number of samples per batch.
        max_length (int): Maximum length of a sequence in tokens.
        stride (int): Step size for sliding window over tokenized data.
        shuffle (bool): Whether to shuffle data for each epoch.
        drop_last (bool): Drop the last incomplete batch if the dataset size isn't divisible by batch size.
        num_workers (int): Number of worker processes for data loading.
        vocab_text (str, optional): Text for creating custom vocabulary. Defaults to None.
        create_vocab (bool, optional): Whether to create a custom vocabulary. Defaults to False.
        encoding (str, optional): Encoding scheme for tokenization (default: "gpt2").
        unk (bool, optional): Include `<|unk|>` token in the custom vocabulary. Defaults to False.
        end_of_text (bool, optional): Include `<|endoftext|>` token in the custom vocabulary. Defaults to False.
        vocab_start (int, optional): Starting index for vocabulary tokens. Defaults to 1.
        use_custom (bool, optional): Whether to use custom vocabulary for tokenization. Defaults to False.
    """

    def __init__(self, data, batch_size, max_length, stride, shuffle, drop_last, num_workers,
                 vocab_text=None, create_vocab=False, encoding="gpt2", unk=False, end_of_text=False, vocab_start=1,
                 use_custom=False):

        self.data = data
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.vocab_text = vocab_text
        self.create_vocab = create_vocab
        self.encoding = encoding
        self.unk = unk
        self.end_of_text = end_of_text
        self.vocab_stride = vocab_start
        self.use_custom = use_custom

    def create_dataloader(self):
        """
        Creates and returns a PyTorch DataLoader instance for batching tokenized text.

        Returns:
            torch.utils.data.DataLoader: A DataLoader for the processed dataset.
        """
        # Initialize the tokenizer with custom or pre-trained vocabulary
        tokenizer = Tokenizer(
            vocab_text=self.vocab_text,
            create_vocab=self.create_vocab,
            encoding=self.encoding,
            unk=self.unk,
            end_of_text=self.end_of_text,
            vocab_start=self.vocab_stride
        )

        # Create a dataset from the tokenized text
        dataset = Dataset(
            data=self.data,
            tokenizer=tokenizer,
            max_length=self.max_length,
            stride=self.stride,
            use_custom=self.use_custom
        )

        # Wrap the dataset with PyTorch DataLoader for efficient batch processing
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers
        )

        return dataloader


class Dataset(torch.utils.data.Dataset):
    """
    Custom Dataset for preparing input-target token sequences.

    Args:
        data (str): Input text data to tokenize and create dataset from.
        tokenizer (Tokenizer): An instance of the Tokenizer class for encoding text.
        max_length (int): Maximum length of tokenized input sequences.
        stride (int): Step size for sliding window over tokenized data.
        use_custom (bool): Whether to use a custom vocabulary for tokenization.
    """

    def __init__(self, data, tokenizer, max_length, stride, use_custom):
        # Initialize the input and target token sequences
        self.input_ids = []
        self.target_ids = []

        # Tokenize the data into token IDs
        token_ids = tokenizer.encode(
            text=data,
            use_custom=use_custom
        )

        # Create input and target sequences using a sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # Input sequence of max_length
            target_chunk = token_ids[i + 1: i + max_length + 1]  # Target sequence shifted by 1
            self.input_ids.append(torch.tensor(input_chunk))  # Store as a tensor
            self.target_ids.append(torch.tensor(target_chunk))  # Store as a tensor

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of input-target sequence pairs.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target token tensors.
        """
        return self.input_ids[idx], self.target_ids[idx]
