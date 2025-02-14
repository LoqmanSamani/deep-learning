import torch
import unittest
from decoder import Decoder

class TestDecoder(unittest.TestCase):
    def setUp(self):

        self.vocabulary_size = 50257
        self.num_layers = 6
        self.input_dimension = 512
        self.output_dimension = 512
        self.num_heads = 8
        self.context_length = 512
        self.dropout_rate = 0.1
        self.qkv_bias = False
        self.scaling_value = 4
        self.epsilon = 1e-5
        self.batch_size = 2
        self.sequence_length = 256


        self.decoder = Decoder(
            vocabulary_size=self.vocabulary_size,
            num_layers=self.num_layers,
            input_dimension=self.input_dimension,
            output_dimension=self.output_dimension,
            num_heads=self.num_heads,
            context_length=self.context_length,
            dropout_rate=self.dropout_rate,
            qkv_bias=self.qkv_bias,
            scaling_value=self.scaling_value,
            epsilon=self.epsilon
        )

    def test_forward(self):

        input_tensor = torch.randint(0, self.vocabulary_size, (self.batch_size, self.sequence_length))
        y_tensor = torch.randn(self.batch_size, self.sequence_length, self.input_dimension)

        output = self.decoder(input_tensor, y=y_tensor)

        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.output_dimension))

    def test_no_y_input(self):

        input_tensor = torch.randint(0, self.vocabulary_size, (self.batch_size, self.sequence_length))

        output = self.decoder(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.output_dimension))

    def test_multiple_layers(self):

        input_tensor = torch.randint(0, self.vocabulary_size, (self.batch_size, self.sequence_length))
        y_tensor = torch.randn(self.batch_size, self.sequence_length, self.input_dimension)

        output = self.decoder(input_tensor, y=y_tensor)

        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.output_dimension))





if __name__ == '__main__':
    unittest.main()
