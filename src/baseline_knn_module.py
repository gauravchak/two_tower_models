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
        num_items: int,  # (NI)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns MIPS ids and embeddings for the given user embedding.

        Args:
            query_embedding (torch.Tensor): The user embedding tensor of shape [B, DU].
            num_items (int): (NI) The number of items.

        Returns:
            torch.Tensor: The MIPS ids tensor of shape [B, NI] and embeddings tensor of shape [B, NI, DI].
        """
        # Find the top NI items in the corpus using torch.topk
        # Note: torch.topk returns a tuple of (values, indices)
        #   dots: [B, NI]
        #   indices: [B, NI]
        knn_scores, indices = torch.topk(
            torch.matmul(query_embedding, self.corpus.T), 
            k=num_items, 
            dim=1
        )
        # Expand indices to create a 3D tensor [B, num_items, 1]
        expanded_indices = indices.unsqueeze(2)

        # Use the expanded indices to gather embeddings from self.corpus
        embeddings = self.corpus[expanded_indices]

        # Squeeze to remove the extra dimension
        embeddings = embeddings.squeeze(2)

        # Return indices, knn_scores, and embeddings
        return indices, knn_scores, embeddings