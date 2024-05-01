import torch
import unittest

from src.two_tower_base_retrieval import TwoTowerBaseRetrieval 
from src.baseline_mips_module import BaselineMIPSModule


class TestTwoTowerBaseRetrieval(unittest.TestCase):
    def setUp(self):
        num_items = 10
        user_id_hash_size = 100
        user_id_embedding_dim = 50
        user_features_size = 20
        item_id_hash_size = 150
        item_id_embedding_dim = 40
        item_features_size = 30
        user_value_weights = [0.1, 0.2, 0.3]
        mips_module = BaselineMIPSModule(corpus_size=1000, embedding_dim=item_id_embedding_dim)

        self.module = TwoTowerBaseRetrieval(
            num_items=num_items,
            user_id_hash_size=user_id_hash_size,
            user_id_embedding_dim=user_id_embedding_dim,
            user_features_size=user_features_size,
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            item_features_size=item_features_size,
            user_value_weights=user_value_weights,
            mips_module=mips_module
        )

        self.batch_size = 32
        self.user_id = torch.randint(0, user_id_hash_size, (self.batch_size,))
        self.user_features = torch.randn(self.batch_size, user_features_size)
        self.user_history = torch.randint(0, num_items, (self.batch_size, 10))

    def test_forward_pass(self):
        item_recommendations = self.module(self.user_id, self.user_features, self.user_history)
        print(item_recommendations)
        self.assertEqual(item_recommendations.shape, torch.Size([self.batch_size, self.module.num_items]))
        self.assertTrue(torch.all(item_recommendations >= 0))
        self.assertTrue(torch.all(item_recommendations < self.module.mips_module.corpus_size))

if __name__ == '__main__':
    unittest.main()
