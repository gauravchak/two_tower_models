import torch
import unittest

from src.two_tower_with_user_history_encoder import TwoTowerWithUserHistoryEncoder
from src.baseline_mips_module import BaselineMIPSModule


class TestTwoTowerWithUserHistoryEncoder(unittest.TestCase):
    def setUp(self):
        num_items = 10
        user_id_hash_size = 100
        user_id_embedding_dim = 50
        user_features_size = 20
        item_id_hash_size = 150
        item_id_embedding_dim = 40
        item_features_size = 30
        user_value_weights = [0.1, 0.2, 0.3]
        user_history_seqlen: int = 128
        corpus_size: int = 1001
        mips_module = BaselineMIPSModule(
            corpus_size=corpus_size, embedding_dim=item_id_embedding_dim
        )

        self.candidate_generator: TwoTowerWithUserHistoryEncoder = TwoTowerWithUserHistoryEncoder(
            num_items=num_items,
            user_id_hash_size=user_id_hash_size,
            user_id_embedding_dim=user_id_embedding_dim,
            user_features_size=user_features_size,
            user_history_seqlen=user_history_seqlen,
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            item_features_size=item_features_size,
            user_value_weights=user_value_weights,
            mips_module=mips_module,
        )

        self.batch_size = 32
        self.user_id = torch.randint(0, user_id_hash_size, (self.batch_size,))
        self.user_features = torch.randn(self.batch_size, user_features_size)
        self.user_history = torch.randint(
            low=0, high=num_items, size=(self.batch_size, user_history_seqlen)
        )

    def test_forward_pass(self):
        """
        Checks that
        1. forward function works
        2. for each query the number of items returned is as configured
        3. the indices of items are valid, i.e. >= 0 and < corpus_size
        """
        item_recommendations = self.candidate_generator(
            self.user_id, self.user_features, self.user_history
        )
        self.assertEqual(
            item_recommendations.shape,
            torch.Size([self.batch_size, self.candidate_generator.num_items]),
        )
        self.assertTrue(torch.all(item_recommendations >= 0))
        self.assertTrue(
            torch.all(
                item_recommendations < self.candidate_generator.mips_module.corpus_size
            )
        )


if __name__ == "__main__":
    unittest.main()
