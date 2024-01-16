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
        # Create a linear layer from 2*DI + 1 to T task logits
        self.light_ranker = nn.Linear(
            in_features=2*item_id_embedding_dim + 1,
            out_features=len(user_value_weights)
        )
    
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
        )  # [B, DI]
        # Query the knn module to get the top num_items items and their embeddings
        knn_items, knn_scores, knn_item_emb = self.knn_module(
            knn_user_embedding,
            self.num_knn_items
        )  # knn_items: [B, num_knn_items] knn_item_emb: [B, num_knn_items, DI]

        # TODO: Compute light ranker user embeddings. Reusing KNN ones for now.
        # Expand user embeddings to [B, NI, DI]
        expanded_user_embeddings = knn_user_embedding.unsqueeze(1).expand(
            knn_items.size(0), 
            knn_items.size(1),
            knn_user_embedding.size(1)
        )

        # Concatenate user embeddings, scores, and item embeddings
        concatenated_input = torch.cat(
            [
                expanded_user_embeddings,
                knn_scores,
                knn_item_emb
            ], dim=2
        )  # [B, num_knn_items, 2 * DI + 1]

        # 3. Compute pointwise immediate reward estimates for each of these items
        task_logits = self.light_ranker(concatenated_input)  # [B, num_knn_items, T]

        # 4. Combine these rewards into a single reward estimate for each item
        net_user_value = torch.sum(
            task_logits * torch.tensor(self.user_value_weights),
            dim=2
        )  # [B, num_knn_items]

        # 5. Select the top num_items items based on this reward estimate
        _, top_indices = torch.topk(
            net_user_value, 
            k=self.num_items, 
            dim=1
        )  # [B, num_items]

        # Gather the top items
        top_items = torch.gather(
            knn_items, 
            dim=1, 
            index=top_indices
        )

        return top_items

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
        knn_loss = super().train_forward(
            user_id, user_features, user_history, item_id, item_features, position, labels
        )

        # Unfortunatley, there is duplication in this code with the superclass method.
        # TODO: Refactor this to avoid duplication.

        # Compute the user embedding
        user_embedding = self.compute_user_embedding(
            user_id, user_features, user_history
        )  # [B, DI]
        # Compute item embeddings
        item_embeddings = self.compute_item_embeddings(
            item_id, item_features
        )  # [B, DI]
        # Compute the dot products between the user embedding and item embeddings
        dots = torch.bmm(user_embedding.unsqueeze(1), item_embeddings.unsqueeze(2)).squeeze(2)  # [B, 1]
        # Concatenate into input for light_ranker
        light_ranker_input = torch.cat(
            [user_embedding, item_embeddings, dots], dim=1
        )
        # Compute task logits
        task_logits = self.light_ranker(light_ranker_input)  # [B, T]
        # Compute binary cross entropy loss with labels
        light_ranker_loss = F.binary_cross_entropy_with_logits(
            task_logits, labels
        )
        return knn_loss + light_ranker_loss