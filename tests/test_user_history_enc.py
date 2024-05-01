import torch
import unittest

from src.user_history_encoder import UserHistoryEncoder


class TestUserHistoryEncoder(unittest.TestCase):
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
        )

        output = model(user_history)

        # Check output shape
        expected_output_shape = (
            batch_size,
            model.get_output_dim() / item_id_embedding_dim,
            item_id_embedding_dim,
        )
        self.assertEqual(output.shape, expected_output_shape)


if __name__ == "__main__":
    unittest.main()
