"""
This derives from TwoTowerWithUserHistoryEncoder and adds position debiasing to the example weights.

This is a specific example of what you can do in terms of modifying the weights. 
There is a lot more to try here. Softmax loss unfortunately is very sensitive to
how you are defining the positives. Hence, such experimentation is high ROI.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.two_tower_base_retrieval import TwoTowerWithUserHistoryEncoder


class TwoTowerWithPositionDebiasedWeights(TwoTowerWithUserHistoryEncoder):
    """
    This derives from TwoTowerWithUserHistoryEncoder and adds position debiasing to the example weights.

    This is a specific example of what you can do in terms of modifying the weights. 
    There is a lot more to try here. Softmax loss unfortunately is very sensitive to
    how you are defining the positives. Hence, such experimentation is high ROI.
    """
    def __init__(
        self,
        num_items: int,
        user_id_hash_size: int,
        user_id_embedding_dim: int,
        user_features_size: int,
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        item_features_size: int,
        user_value_weights: List[float],
        knn_module: nn.Module,
        enable_position_debiasing: bool = False,
    ) -> None:
        """
        params:
            num_items: the number of items to return per user/query
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
            knn_module: a module that computes the Maximum Inner Product Search (MIPS)
                over the item embeddings given the user embedding.
            enable_position_debiasing: when enabled, we will debias the net_user_value
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
            knn_module,
        )
        self.enable_position_debiasing = enable_position_debiasing
        if self.enable_position_debiasing:
            # Create an embedding arch to process position
            self.position_bias_net_user_value = nn.Embedding(
                num_embeddings=100,
                embedding_dim=1
            )

    def debias_net_user_value(
        self,
        net_user_value: torch.Tensor,  # [B]
        position: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the processed net_user_value and any losses to be added to the loss function.

        Args:
            net_user_value (torch.Tensor): The input net_user_value tensor of shape [B].
            position (torch.Tensor): The input position tensor of shape [B].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the processed 
                net_user_value tensor and the position_bias_loss tensor.
        """
        # Optionally debias the net_user_value by the part explained purely by position
        if self.enable_position_debiasing:
            # Compute the position bias
            position_bias = self.position_bias_net_user_value(position).squeeze(1)  # [B]
            # Compute MSE loss between net_user_value and position_bias
            position_bias_loss = F.mse_loss(
                input=net_user_value,
                target=position_bias,
                reduction="mean"
            )  # [1]
            # Compute the net_user_value without position bias
            net_user_value = net_user_value - position_bias
        return net_user_value, position_bias_loss

