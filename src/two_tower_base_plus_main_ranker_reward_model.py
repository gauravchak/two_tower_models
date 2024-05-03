"""
TODO
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.two_tower_with_debiasing import TwoTowerWithDebiasing


class TwoTowerWithMainRankerReward(TwoTowerWithDebiasing):
    """
        Build a proxy of the main ranking estimator model here.
        While training use that proxy to compute the reward for both positives
        and sampled negatives.
        Convert these rewards into probabilities using softmax-ranking, and 
        use these probabilities to compute the loss.
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
        Args:
            num_items: the number of items to return per user/query
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
        # Train a proxy ranker
        simple_ranker = nn.Linear(2 * item_id_embedding_dim + 1, len(user_value_weights))

    def train_forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        position: torch.Tensor,  # [B]
        labels: torch.Tensor  # [B, T]
    ) -> float:
        """
        This function computes the loss during training.

        Args:
            user_id (torch.Tensor): User IDs. Shape: [B].
            user_features (torch.Tensor): User features. Shape: [B, IU].
            user_history (torch.Tensor): User history. Shape: [B, H].
            item_id (torch.Tensor): Item IDs. Shape: [B].
            item_features (torch.Tensor): Item features. Shape: [B, II].
            position (torch.Tensor): Position. Shape: [B].
            labels (torch.Tensor): Labels. Shape: [B, T].

        Returns:
            float: The computed loss.

        Notes:
            - The loss is computed using softmax loss and weighted by the net_user_value.
            - Optionally, the net_user_value can be debiased by the part explained purely by position.
            - The loss is clamped to preserve positive net_user_value and normalized between 0 and 1.
        """
        # Compute the user embedding
        user_embedding = self.compute_user_embedding(
            user_id, user_features, user_history
        )  # [B, DI]
        # Compute item embeddings
        item_embeddings = self.compute_item_embeddings(
            item_id, item_features
        )  # [B, DI]

        # Compute the regular softmax loss with user-positives
        # and examples weighted by net_user_value that is debiased by position
        # and user features.
        loss = super().compute_training_loss(
            user_embedding,
            item_embeddings,
            position=position,
            labels=labels
        )

        # Compute an alignment loss using a proxy ranker as a reward model.

        # Compute per task logits [T, B, B] and loss [B] for proxy ranker

        # Compute ranker logits [T, B, B] for impressed and unimpressed items.
        # Convert to ranker_vm_score [B, X] using user_value_weights
        # Using F.softmax convert ranker_vm_score into a probability of ranker
        # showing item at top.
        # Use torch.kl_div with ranker_top_probs as target and 
        # F.log_softmax(logits) as input.

        return loss  # ()

