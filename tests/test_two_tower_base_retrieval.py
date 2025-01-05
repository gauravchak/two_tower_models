import torch
import unittest

from src.two_tower_base_retrieval import TwoTowerBaseRetrieval
from src.baseline_mips_module import BaselineMIPSModule


class TestTwoTowerBaseRetrieval(unittest.TestCase):
    def setUp(self):
        self.num_items = 10
        self.user_id_hash_size = 100
        self.user_id_embedding_dim = 50
        self.user_features_size = 20
        self.item_id_hash_size = 150
        self.item_id_embedding_dim = 40
        self.item_features_size = 30
        self.tasknum: int = 3
        self.user_value_weights = [0.1, 0.2, 0.3]  # dimension = self.tasknum
        self.user_history_seqlen: int = 128
        self.corpus_size: int = 1001
        self.mips_module = BaselineMIPSModule(
            corpus_size=self.corpus_size,
            embedding_dim=self.item_id_embedding_dim,
        )

        self.candidate_generator = TwoTowerBaseRetrieval(
            num_items=self.num_items,
            user_id_hash_size=self.user_id_hash_size,
            user_id_embedding_dim=self.user_id_embedding_dim,
            user_features_size=self.user_features_size,
            item_id_hash_size=self.item_id_hash_size,
            item_id_embedding_dim=self.item_id_embedding_dim,
            item_features_size=self.item_features_size,
            user_value_weights=self.user_value_weights,
            mips_module=self.mips_module,
        )

        self.batch_size = 32
        self.user_id = torch.randint(
            0, self.user_id_hash_size, (self.batch_size,)
        )
        self.user_features = torch.randn(
            self.batch_size, self.user_features_size
        )
        self.user_history = torch.randint(
            low=0,
            high=self.num_items,
            size=(self.batch_size, self.user_history_seqlen),
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
                item_recommendations
                < self.candidate_generator.mips_module.corpus_size
            )
        )

    def test_train_forward(self):
        self.batch_size = 32
        self.user_id = torch.randint(
            0, self.user_id_hash_size, (self.batch_size,)
        )
        self.user_features = torch.randn(
            self.batch_size, self.user_features_size
        )
        self.user_history = torch.randint(
            low=0,
            high=self.num_items,
            size=(self.batch_size, self.user_history_seqlen),
        )
        item_id = torch.randint(0, self.item_id_hash_size, (self.batch_size,))
        item_features = torch.randn(self.batch_size, self.item_features_size)
        position = torch.randint(
            0, 100, (self.batch_size,)
        )  # Assuming position range
        labels = torch.randint(
            0, 2, (self.batch_size, self.tasknum), dtype=torch.float32
        )  # binary

        # Compute the loss using the train_forward method
        loss = self.candidate_generator.train_forward(
            self.user_id,
            self.user_features,
            self.user_history,
            item_id,
            item_features,
            position,
            labels,
        )

        # Assert that the loss is a float
        self.assertIsInstance(loss.item(), float)


if __name__ == "__main__":
    unittest.main()
