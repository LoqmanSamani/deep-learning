import torch



class MultiHeadAttention(torch.nn.Module):
    """Implements scaled dot-product multi-head attention."""
    def __init__(
        self,
        input_dimension=512,
        output_dimension=512,
        num_heads=8,
        context_length=512,
        dropout_rate=0.1,
        qkv_bias=False,
    ):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.num_heads = num_heads
        self.head_dimension = self.input_dimension // self.num_heads
        assert self.input_dimension % self.num_heads == 0, "Input dimension must be divisible by the number of heads."

        self.Wq = torch.nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.Wk = torch.nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.Wv = torch.nn.Linear(input_dimension, output_dimension, bias=qkv_bias)

        self.out_project = torch.nn.Linear(output_dimension, output_dimension)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        )

    def create_causal_mask(self, seq_length):
        """create a causal mask dynamically."""
        return torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1)

    def forward(self, x, y=None, apply_mask=False):
        batch_size, num_tokens, input_dimension = x.shape
        assert input_dimension == self.input_dimension, "Input dimension mismatch."

        if y is not None:
            Q = self.Wq(y)
            K = self.Wk(y)
            V = self.Wv(x)
        else:
            Q = self.Wq(x)
            K = self.Wk(x)
            V = self.Wv(x)

        Q = Q.view(batch_size, num_tokens, self.num_heads, self.head_dimension).transpose(1, 2)
        K = K.view(batch_size, num_tokens, self.num_heads, self.head_dimension).transpose(1, 2)
        V = V.view(batch_size, num_tokens, self.num_heads, self.head_dimension).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dimension ** 0.5)
        if apply_mask:
            causal_mask = self.mask[:num_tokens, :num_tokens]
            attention_scores = attention_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = torch.matmul(attention_weights, V)
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.output_dimension)
        output = self.out_project(context_vector)

        return output




