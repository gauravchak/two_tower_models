"""
This derives from TwoTowerBaseRetrieval and adds a user history encoder.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.user_history_encoder import UserHistoryEncoder
from src.two_tower_base_retrieval import TwoTowerBaseRetrieval


class TwoTowerWithUserHistoryEncoder(TwoTowerBaseRetrieval):
    """
    This derives from TwoTowerBaseRetrieval and adds a user history encoder.
    """

    def __init__(
        self,
        num_items: int,
        user_id_hash_size: int,
        user_id_embedding_dim: int,
        user_features_size: int,
        user_history_seqlen: int,
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        item_features_size: int,
        user_value_weights: List[float],
        mips_module: nn.Module,
    ) -> None:
        """
        params:
            num_items: the number of items to return per user/query
            user_id_hash_size: the size of the embedding table for users
            user_id_embedding_dim (DU): internal dimension
            user_features_size (IU): input feature size for users
            user_history_seqlen (H): length of the user history sequence
            item_id_hash_size: the size of the embedding table for items
            item_id_embedding_dim (DI): internal dimension
            item_features_size: (II) input feature size for items
            cross_features_size: (IC) size of cross features
            user_value_weights: T dimensional weights, such that a linear
                combination of point-wise immediate rewards is the best predictor
                of long term user satisfaction.
            mips_module: a module that computes the Maximum Inner Product Search (MIPS)
                over the item embeddings given the user embedding.
        """
        super().__init__(
            num_items=num_items,
            user_id_hash_size=user_id_hash_size,
            user_id_embedding_dim=user_id_embedding_dim,
            user_features_size=user_features_size,
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            item_features_size=item_features_size,
            user_value_weights=user_value_weights,
            mips_module=mips_module,
        )

        # Create a user history encoder
        # Ideally num_attention_heads and num_attention_layers should be
        # in a config and not hardcoded.
        self.user_history_encoder = UserHistoryEncoder(
            item_id_embedding_dim=item_id_embedding_dim,
            history_len=user_history_seqlen,
            num_attention_heads=4,
            num_attention_layers=3,
            use_positional_encoding=True,
        )

        user_history_output_dim = self.user_history_encoder.get_output_dim()
        # Create an arch to process the user_tower_input
        # Input dimension =
        #   user_id_embedding_dim from get_user_embedding
        #   user_id_embedding_dim from user_features_arch
        #   user_history_output_dim from user_history_encoder
        # Output dimension = item_id_embedding_dim
        # The output of this arch will be used for MIPS module.
        # Hence this needs to be same as the item tower output.
        self.user_tower_arch = nn.Linear(
            2 * user_id_embedding_dim + user_history_output_dim, item_id_embedding_dim
        )

    def process_user_features(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        """
        Process the user features and user history to generate the input for the user tower.

        Args:
            user_id (torch.Tensor): Tensor representing the user ID. Shape: [B]
            user_features (torch.Tensor): Tensor representing the user features. Shape: [B, IU]
            user_history (torch.Tensor): Tensor representing the user history. Shape: [B, H]

        Returns:
            torch.Tensor: Tensor representing the input for the user tower.
                Shape: [B, 2 * DU + 2 * DI]
        """
        # Pass the user history through the item embedding layer to convert to
        # embeddings from ids
        user_history_embedding = self.item_id_embedding_arch(user_history)  # [B, H, DI]

        # Pass the user history through the user history encoder
        user_history_summary = self.user_history_encoder(
            user_history_embedding
        )  # [B, 2, DI]
        user_history_summary = user_history_summary.view(
            user_history_summary.shape[0], -1
        )  # [B, 2 * DI]

        # Concatenate flattened user_history_summary with the user_tower_input
        # derived from the super class, presumably only using user_id and
        # user_features.
        user_tower_input = super().process_user_features(
            user_id=user_id, user_features=user_features, user_history=user_history
        )
        user_tower_input = torch.cat([user_tower_input, user_history_summary], dim=1)
        return user_tower_input
