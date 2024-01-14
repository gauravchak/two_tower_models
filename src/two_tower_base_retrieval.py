"""
This is a specific instance of a two-tower based candidate generator (retrieval)
in a recommender system.
Ref: https://recsysml.substack.com/p/two-tower-models-for-retrieval-of
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerBaseRetrieval(nn.Module):
    def __init__(
        self,
        user_id_hash_size: int,
        user_id_embedding_dim: int,
        user_features_size: int,
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        item_features_size: int,
        user_value_weights: List[float],
    ) -> None:
        """
        params:
            num_tasks (T): The tasks to compute estimates of
            user_id_hash_size: the size of the embedding table for users
            user_id_embedding_dim (DU): internal dimension
            user_features_size (IU): input feature size for users
            item_id_hash_size: the size of the embedding table for items
            item_id_embedding_dim (DI): internal dimension
            item_features_size: (II) input feature size for items
            cross_features_size: (IC) size of cross features
            user_value_weights: T dimensional weights, such that a linear
            combination of point-wise immediate rewards is the best predictor
            of long term user satisfaction.
        """
        super().__init__()
        self.user_value_weights = torch.tensor(user_value_weights)  # noqa TODO add device input.

        # Embedding layers for user and item ids
        self.user_id_embedding_arch = nn.Embedding(
            user_id_hash_size, user_id_embedding_dim)
        self.item_id_embedding_arch = nn.Embedding(
            item_id_hash_size, item_id_embedding_dim)        

    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        position: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        """
        pass

    def train_forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        position: torch.Tensor,  # [B]
        labels: torch.Tensor  # [B, T]
    ) -> float:
        """Compute the loss during training"""
        pass

