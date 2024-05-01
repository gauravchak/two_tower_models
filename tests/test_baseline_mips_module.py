import torch
import unittest

from src.baseline_mips_module import BaselineMIPSModule


class TestBaselineMIPSModule(unittest.TestCase):
    def setUp(self):
        self.corpus_size = 100
        self.embedding_dim = 50
        self.num_items = 10
        self.batch_size = 32
        self.query_embedding = torch.randn(self.batch_size, self.embedding_dim)
        self.module = BaselineMIPSModule(self.corpus_size, self.embedding_dim)

    def test_output_shapes(self):
        indices, mips_scores, embeddings = self.module(
            self.query_embedding, self.num_items
        )
        self.assertEqual(indices.shape, torch.Size([self.batch_size, self.num_items]))
        self.assertEqual(
            mips_scores.shape, torch.Size([self.batch_size, self.num_items])
        )
        self.assertEqual(
            embeddings.shape,
            torch.Size([self.batch_size, self.num_items, self.embedding_dim]),
        )

    def test_output_values(self):
        indices, mips_scores, embeddings = self.module(
            self.query_embedding, self.num_items
        )
        self.assertTrue(torch.all(indices >= 0))
        self.assertTrue(torch.all(indices < self.corpus_size))
        self.assertTrue(torch.all(mips_scores >= 0))
        self.assertEqual(indices.shape, mips_scores.shape)


if __name__ == "__main__":
    unittest.main()
