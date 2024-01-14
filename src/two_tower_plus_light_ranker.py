from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.two_tower_base_retrieval import TwoTowerBaseRetrieval

class TwoTowerPlusLightRanker(TwoTowerBaseRetrieval):
    def __init__(
        self,
        num_items: int,
        num_knn_items: int,
        user_id_hash_size: int,
        user_id_embedding_dim: int,
        user_features_size: int,
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        item_features_size: int,
        user_value_weights: List[float],
        knn_module: nn.Module,
    ) -> None:
        """
        params:
            num_items: the number of items to return per user/query
            num_knn_items: the number of items to retrieve using the knn module
            user_id_hash_size: the size of the embedding table for users
            user_id_embedding_dim (DU): internal dimension
            user_features_size (IU): input feature size for users
            item_id_hash_size: the size of the embedding table for items
            item_id_embedding_dim (DI): internal dimension
            item_features_size: (II) input feature size for items
            user_value_weights: T dimensional weights, such that a linear
                combination of point-wise immediate rewards is the best predictor
                of long term user satisfaction.
            knn_module: a module that computes the Maximum Inner Product Search (MIPS)
                over the item embeddings given the user embedding.
        """
        super().__init__(
            num_items,
            user_id_hash_size,
            user_id_embedding_dim,
            user_features_size,
            item_id_hash_size,
            item_id_embedding_dim,
            item_features_size,
            user_value_weights,
        )
        self.num_knn_items = num_knn_items

    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
    ) -> torch.Tensor:
        """
        1. Compute the knn and ranking user embeddings
        2. Get num_knn_items items using the knn user embedding and KNN module.
        3. Compute the item embeddings for the top num_items items.
        4. Compute the item scores using ranking user embedding.
        5. Compute a composite score using the item scores and user_value_weights.
        6. Return the top num_items items using the composite score.
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
    ) -> torch.Tensor:
        """
        """
        pass
