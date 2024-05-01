"""
This derives from TwoTowerWithUserHistoryEncoder and adds user
debiasing to the example weights.

We estimate net_user_value as a function of user features. Then we
ensure that the estimated net_user_value is a positive number. Then
we devide net_user_value by the estimated net_user_value to get the
user-debiased net_user_value.

Intuition: ney_user_value is the immediate reward that the user gets
from the recommendation. This is a point-wise reward. For users who
are intrisically more likely to click/watch/buy/action on recommendations,
we want data points with a similar level of positive reward to produce a
smaller positive gradient. This is because the recommendation produced
similar user value as a counterfactual recommendation would have. On the
contrary, for users who are intrinsically less likely to action on 
recommendations, we want data points with a similar level of positive
reward to produce a larger positive gradient. (vice-versa for hard 
negatives)

This is a form of importance sampling. Since the objective of the recommender
system is to maximize the net user value, we want to give more importance to
data points where the surprise to the net user value is higher. This is
achieved by dividing the net user value by the estimated net user value.

In terms of user distribution, we expect the estimated net user value to be
higher for power users and lower for casual users. 
i.e. P(action | power user) > P(action | casual user). 
Hence this approach ends up giving more importance to casual users. Thus the
name "user debiasing".
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.two_tower_with_user_history_encoder import TwoTowerWithUserHistoryEncoder


class TwoTowerWithUserDebiasedWeights(TwoTowerWithUserHistoryEncoder):
    """
    This derives from TwoTowerWithUserHistoryEncoder and adds user
    debiasing to the example weights.

    We estimate net_user_value as a function of user features. Then we
    ensure that the estimated net_user_value is a positive number. Then
    we devide net_user_value by the estimated net_user_value to get the
    user-debiased net_user_value.
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
        # Create an MLP to process user_embedding, which is
        # a shortcut to user features.
        self.user_debias_net_user_value = nn.Sequential(
            nn.Linear(item_id_embedding_dim, 1)
        )

    def debias_net_user_value(
        self,
        net_user_value: torch.Tensor,  # [B]
        position: torch.Tensor,  # [B]
        user_embedding: torch.Tensor,  # [B, DI]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the processed net_user_value and any losses to be added to the loss function.

        Args:
            net_user_value (torch.Tensor): The input net_user_value tensor [B].
            position (torch.Tensor): The input position tensor of shape [B].
            user_embedding: same as what is used in MIPS  # [B, DI]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the processed
                net_user_value tensor and the loss tensor from estimating.
        """
        # Estimate net_user_value from user_embedding
        estimated_net_user_value = self.user_debias_net_user_value(
            user_embedding
        ).squeeze(
            1
        )  # [B]
        # Ensure that estimated_net_user_value is positive
        estimated_net_user_value = torch.clamp(
            estimated_net_user_value,
            min=1e-1,  # Small positive number, choose as per your data
        )  # [B]
        # Compute MSE loss between net_user_value and estimated_net_user_value
        estimated_net_user_value_loss = F.mse_loss(
            input=estimated_net_user_value, target=net_user_value, reduction="sum"
        )  # [1]
        # Compute the net_user_value without user bias
        net_user_value = net_user_value / estimated_net_user_value
        return net_user_value, estimated_net_user_value_loss
