"""
This derives from TwoTowerWithUserHistoryEncoder and adds both user and
position debiasing to the example weights.

We do user debiasing similar to TwoTowerWithUserDebiasedWeights
and position debiasing similar to TwoTowerWithPositionDebiasedWeights.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.two_tower_with_user_history_encoder import TwoTowerWithUserHistoryEncoder


class TwoTowerWithDebiasing(TwoTowerWithUserHistoryEncoder):
    """
    This derives from TwoTowerWithUserHistoryEncoder and adds both user and
    position debiasing to the example weights.

    We do user debiasing similar to TwoTowerWithUserDebiasedWeights
    and position debiasing similar to TwoTowerWithPositionDebiasedWeights.
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
            user_history_seqlen=user_history_seqlen,
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            item_features_size=item_features_size,
            user_value_weights=user_value_weights,
            mips_module=mips_module,
        )
        # Create an embedding arch to process position
        self.position_bias_net_user_value = nn.Embedding(
            num_embeddings=100, embedding_dim=1
        )
        # Create an MLP to process user_embedding and position bias.
        self.user_debias_net_user_value = nn.Sequential(
            nn.Linear(item_id_embedding_dim + 1, 1)
        )

    def debias_net_user_value(
        self,
        net_user_value: torch.Tensor,  # [B]
        position: torch.Tensor,  # [B]
        user_embedding: torch.Tensor,  # [B, DI]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the processed net_user_value and any losses to be added to the loss function.
        The way this is implemented is:
        - We use position to come up with an estimate of user value: E_nuv_position
        - We use an NN arch to model user value using user_embedding from the user
            tower and E_nuv_position. This cumulative user value estimate is
            E_nuv_user.

        Args:
            net_user_value (torch.Tensor): The input net_user_value tensor [B].
            position (torch.Tensor): The input position tensor of shape [B].
            user_embedding: same as what is used in MIPS  # [B, DI]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the processed
                net_user_value tensor and the loss tensor from estimating.
        """
        E_nuv_position = self.position_bias_net_user_value(position)  # [B, 1]

        # Estimate net_user_value
        E_nuv_user = self.user_debias_net_user_value(
            torch.cat([user_embedding, E_nuv_position], dim=-1)
        ).squeeze(
            1
        )  # [B]

        # Compute MSE loss between net_user_value and E_nuv_user
        E_nuv_position_loss = F.mse_loss(
            input=E_nuv_position, target=net_user_value, reduction="sum"
        )  # [1]

        # Compute MSE loss between net_user_value and E_nuv_user
        E_nuv_user_loss = F.mse_loss(
            input=E_nuv_user, target=net_user_value, reduction="sum"
        )  # [1]

        # Ensure that estimated_net_user_value is positive
        E_nuv_user = torch.clamp(
            E_nuv_user, min=1e-3  # Small positive number, choose as per your data
        )  # [B]

        # Compute the net_user_value without user bias
        # Since net_user_value >= 0
        # dividing by E_nuv_user maintains the invariant net_user_value >= 0
        net_user_value = net_user_value / E_nuv_user

        return net_user_value, E_nuv_user_loss + E_nuv_position_loss
