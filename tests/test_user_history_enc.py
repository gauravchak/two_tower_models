import torch
import unittest

import numpy as np
import random

from src.user_history_encoder import UserHistoryEncoder


class TestUserHistoryEncoder(unittest.TestCase):
    def setUp(self):
        # Set the random seed for PyTorch
        torch.manual_seed(42)

        # Set the random seed for Numpy
        np.random.seed(42)

        # Set the random seed for Python's built-in random module
        random.seed(42)

    def test_forward(self):
        item_id_embedding_dim = 64
        history_len = 128
        num_attention_heads = 4
        num_attention_layers = 12

        batch_size = 32
        user_history = torch.randn(batch_size, history_len, item_id_embedding_dim)

        model = UserHistoryEncoder(
            item_id_embedding_dim=item_id_embedding_dim,
            history_len=history_len,
            num_attention_heads=num_attention_heads,
            num_attention_layers=num_attention_layers,
            use_positional_encoding=True,
        )

        output = model(user_history)

        # Check output shape
        expected_output_shape = (
            batch_size,
            model.get_output_dim() / item_id_embedding_dim,
            item_id_embedding_dim,
        )
        self.assertEqual(output.shape, expected_output_shape)

    def test_values(self):
        # Set parameters for the encoder
        item_id_embedding_dim = 2
        history_len = 3
        num_attention_heads = 1
        num_attention_layers = 1
        use_positional_encoding = False  # Turn off positional encoding for testing

        # Initialize the encoder
        encoder = UserHistoryEncoder(
            item_id_embedding_dim=item_id_embedding_dim,
            history_len=history_len,
            num_attention_heads=num_attention_heads,
            num_attention_layers=num_attention_layers,
            use_positional_encoding=use_positional_encoding,
        )

        # Prepare a dummy input
        user_history = torch.tensor(
            [
                [[1, 2], [3, 4], [-1, 0]],
            ],
            dtype=torch.float32,
        )  # [B, H, DI]

        # Forward pass through the encoder
        output = encoder(user_history)

        # Define the expected output manually based on the calculation
        expected_output = torch.tensor(
            [
                [[0.8240, 0.7119], [1.0000, 2.0000]],
            ],
            dtype=torch.float32,
        )

        # Check if the output matches the expected output
        self.assertTrue(torch.allclose(input=output, other=expected_output, atol=1e-3))

    def test_values_pos(self):
        # Set parameters for the encoder
        item_id_embedding_dim = 2
        history_len = 3
        num_attention_heads = 1
        num_attention_layers = 1
        use_positional_encoding = True

        # Initialize the encoder
        encoder = UserHistoryEncoder(
            item_id_embedding_dim=item_id_embedding_dim,
            history_len=history_len,
            num_attention_heads=num_attention_heads,
            num_attention_layers=num_attention_layers,
            use_positional_encoding=use_positional_encoding,
        )

        # Prepare a dummy input
        user_history = torch.tensor(
            [
                [[1, 2], [3, 4], [-1, 0]],
            ],
            dtype=torch.float32,
        )  # [B, H, DI]

        # Forward pass through the encoder
        output = encoder(user_history)

        # Define the expected output manually based on the calculation
        expected_output = torch.tensor(
            [
                [[1.4978, 1.2425], [1.0000, 2.0000]],
            ],
            dtype=torch.float32,
        )

        # Check if the output matches the expected output
        self.assertTrue(torch.allclose(input=output, other=expected_output, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
