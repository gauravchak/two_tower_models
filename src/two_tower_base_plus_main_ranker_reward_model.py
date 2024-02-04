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
        # Compute logits for every pair of user and item
        logits = torch.matmul(user_embedding, item_embeddings.t())  # [B, B]

        # You should either try to handle the popularity bias 
        # of in-batch negatives using log-Q correction or
        # use random negatives. Mixed Negative Sampling paper suggests
        # random negatives is a better approach.
        # Here we are not implementing either due to time constraints.

        # Compute softmax loss
        # F.cross_entropy accepts target as 
        #   ground truth class indices or class probabilities;
        # Here we are using class indices
        target = torch.arange(logits.shape[0]).to(logits.device)  # [B]
        # We are not reducing to mean since not every row in the batch is a 
        # "positive" example. We are weighting the loss by the net_user_value
        # after this to give more weight to the positive examples and possibly
        # 0 weight to the hard-negative examples. Note that net_user_value is
        # assumed to be non-negative.
        loss = F.cross_entropy(
            input=logits,
            target=target,
            reduction="none"
        )  # [B]

        # Compute the weighted average of the labels using user_value_weights
        # In the simplest case, assume you have a single label per item.
        # This label is either 1 or 0 depending on whether the user engaged
        # with this item when recommended. Then the net_user_value is 1 when
        # the user has engaged with the item and 0 otherwise.
        net_user_value = torch.matmul(labels, self.user_value_weights)  # [B]

        # Optionally debias the net_user_value by the part explained purely 
        # by position. Not implemented in this version. Hence net_user_value
        # is unchanged and additional_loss is 0.
        net_user_value, additional_loss = self.debias_net_user_value(
            net_user_value=net_user_value,
            position=position,
            user_embedding=user_embedding,
        )  # [B], [1]

        # Floor by epsilon to only preserve positive net_user_value 
        net_user_value = torch.clamp(
            net_user_value,
            min=0.000001  # small epsilon to avoid divide by 0
        )  # [B]

        # Compute the product of loss and net_user_value
        loss = loss * net_user_value  # [B]
        loss = torch.mean(loss)  # ()

        # This loss helps us learn the debiasing archs
        loss = loss + additional_loss

        # Compute per task logits [T, B] and loss [B] for proxy ranker

        # Compute ranker logits [T, B, X] for impressed and unimpressed items.
        # Convert to ranker_vm_score [B, X] using user_value_weights
        # Using F.softmax convert ranker_vm_score into a probability of ranker
        # showing item at top.
        # Use torch.kl_div with ranker_top_probs as target and 
        # F.log_softmax(logits) as input.

        return loss  # ()

