"""
TODO
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.two_tower_with_position_debiased_weights import TwoTowerWithPositionDebiasedWeights


class TwoTowerPlusLightRanker(TwoTowerWithPositionDebiasedWeights):
    """
    This extends TwoTowerWithPositionDebiasedWeights and adds a light ranker.
    During inference, adter retrieving the top num_knn_items items and their
    embeddings using the knn module, we will compute pointwise immediate
    reward estimates for each of these items. We will then use user_value_weights
    to combine these rewards into a single reward estimate for each item.
    We will then select the top num_items items based on this reward estimate.
    """
    def __init__(
        self,
        num_items: int,
        num_knn_items: int,
        user_id_hash_size: int,
        user_id_embedding_dim: int,
        user_features_size: int,
        user_history_seqlen: int,
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        item_features_size: int,
        user_value_weights: List[float],
        knn_module: nn.Module,
        enable_position_debiasing: bool,
    ) -> None:
        """
        params:
            num_items: the number of items to return per user/query
            num_knn_items: the number of items to retrieve using the knn module
            user_id_hash_size: the size of the embedding table for users
            user_id_embedding_dim (DU): internal dimension
            user_features_size (IU): input feature size for users
            user_history_seqlen (H): length of the user history sequence
            item_id_hash_size: the size of the embedding table for items
            item_id_embedding_dim (DI): internal dimension
            item_features_size: (II) input feature size for items
            user_value_weights: T dimensional weights, such that a linear
                combination of point-wise immediate rewards is the best predictor
                of long term user satisfaction.
            knn_module: a module that computes the Maximum Inner Product Search (MIPS)
                over the item embeddings given the user embedding.
            enable_position_debiasing: when enabled, we will debias the net_user_value
        """
        super().__init__(
            num_items=num_items,
            user_id_hash_size=user_id_hash_size,
            user_id_embedding_dim=user_id_embedding_dim,
            user_features_size=user_features_size,
            user_history_seqlen=user_history_seqlen,
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            item_features_size=item_features_size,
            user_value_weights=user_value_weights,
            knn_module=knn_module,
            enable_position_debiasing=enable_position_debiasing,
        )
        self.num_knn_items = num_knn_items
    
    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        """This is used for inference.

        1. Compute the user embedding for KNN
        2. Get num_knn_items items and their embeddings using the knn module
        3. Compute pointwise immediate reward estimates for each of these items
        4. Combine these rewards into a single reward estimate for each item
        5. Select the top num_items items based on this reward estimate

        Args:
            user_id (torch.Tensor): Tensor representing the user ID. Shape: [B]
            user_features (torch.Tensor): Tensor representing the user features. Shape: [B, IU]
            user_history (torch.Tensor): Tensor representing the user history. Shape: [B, H]

        Returns:
            torch.Tensor: Tensor representing the top num_items items. Shape: [B, num_items]
        """
        # Compute the user embedding
        knn_user_embedding = self.compute_user_embedding(
            user_id, user_features, user_history
        )
        # Query the knn module to get the top num_items items and their embeddings
        knn_items, knn_item_emb = self.knn_module(
            knn_user_embedding,
            self.num_knn_items
        )  # knn_items: [B, num_knn_items] knn_item_emb: [B, num_knn_items, DI]


        return top_items
