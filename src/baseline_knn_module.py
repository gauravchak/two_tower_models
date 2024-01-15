from typing import Tuple

import torch
import torch.nn as nn


class BaselineKNNModule(nn.Module):
    def __init__(
        self, 
        corpus_size:int, 
        embedding_dim:int
    ) -> None:
        """
        Initialize the BaselineKNNModule.

        Args:
            corpus_size (int): The size of the corpus.
            embedding_dim (int): The dimension of the embeddings.

        Returns:
            None
        """
        super(BaselineKNNModule, self).__init__()
        self.corpus_size = corpus_size
        self.embedding_dim = embedding_dim
        # Create a random corpus of size [corpus_size, embedding_dim]
        self.corpus = torch.randn(corpus_size, embedding_dim)  # [C, DI]

    def forward(
        self, 
        query_embedding: torch.Tensor,  # [B, DI]
        num_items: int,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns MIPS ids and embeddings for the given user embedding.

        Args:
            query_embedding (torch.Tensor): The user embedding tensor of shape [B, DU].
            num_items (int): The number of items.

        Returns:
            torch.Tensor: The MIPS ids tensor of shape [B] and embeddings tensor of shape [B, DI].
        """
        # Find the top num_items items in the corpus using torch.topk
        # Note: torch.topk returns a tuple of (values, indices)
        #   dots: [B, num_items]
        #   indices: [B, num_items]
        dots, indices = torch.topk(
            torch.matmul(query_embedding, self.corpus.T), 
            k=num_items, 
            dim=1
        )
        # Return the indices and embeddings
        return indices, self.corpus[indices]