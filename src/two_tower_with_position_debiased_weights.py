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

from src.two_tower_with_user_history_encoder import TwoTowerWithUserHistoryEncoder


class TwoTowerWithPositionDebiasedWeights(TwoTowerWithUserHistoryEncoder):
    """
    This derives from TwoTowerWithUserHistoryEncoder and adds position debiasing to
    the example weights. If you recall during training of two tower models we would
    compute softmax loss from all data points but then we would multiply the loss
    with an example level non-negative weight "net_user_value". This is expected to
    be 0 for hard negatives. This is how we in effect only learn from positives.

    Softmax loss unfortunately is very sensitive to how you are defining positives.
    Even a little allocation of positive weight to random items can hurt performance.
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
                net_user_value tensor and the position_bias_loss tensor.
        """
        # Estimate the part of user value that is explained purely by position.
        estimated_net_user_value = self.position_bias_net_user_value(position).squeeze(
            1
        )  # [B]

        # Compute MSE loss between net_user_value and estimated_net_user_value
        # The gradient from this loss will help to train position_bias_net_user_value
        estimated_net_user_value_loss = F.mse_loss(
            input=estimated_net_user_value, target=net_user_value, reduction="sum"
        )  # [1]

        # Ensure that estimated_net_user_value is positive
        estimated_net_user_value = torch.clamp(
            estimated_net_user_value,
            min=1e-3,  # Small positive number, choose as per your data
        )  # [B]

        # Compute the net_user_value without position bias
        net_user_value = net_user_value / estimated_net_user_value
        return net_user_value, estimated_net_user_value_loss
