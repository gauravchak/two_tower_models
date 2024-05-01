"""
TODO
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.two_tower_with_debiasing import TwoTowerWithDebiasing


class TwoTowerPlusLightRanker(TwoTowerWithDebiasing):
    """
    This extends TwoTowerWithDebiasing and adds a light ranker.
    During inference, adter retrieving the top num_mips_items items and their
    embeddings using the mips module, we will compute pointwise immediate
    reward estimates for each of these items. We will then use user_value_weights
    to combine these rewards into a single reward estimate for each item.
    We will then select the top num_items items based on this reward estimate.
    """

    def __init__(
        self,
        num_items: int,
        num_mips_items: int,
        num_ranker_user_embeddings: int,
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
            num_mips_items: the number of items to retrieve using the mips module
            num_ranker_user_embeddings: the number of user embeddings for light ranker
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
        self.num_mips_items: int = num_mips_items
        self.num_ranker_user_embeddings: int = num_ranker_user_embeddings

        num_user_history_dims: int = 2  # Keep in sync with user history encoder
        user_history_output_dim: int = num_user_history_dims * item_id_embedding_dim
        # Create a user tower arch to process the user_tower_input
        # and produce the light ranker user embeddings [B, NU, DI]
        # Input dimension =
        #   user_id_embedding_dim from get_user_embedding
        #   user_id_embedding_dim from user_features_arch
        #   user_history_output_dim from item_id_embedding_arch
        self.ranker_user_tower = nn.Linear(
            2 * user_id_embedding_dim + user_history_output_dim,
            num_ranker_user_embeddings * item_id_embedding_dim,
        )

        # Create a linear layer from ( 2*DI + NU + 1) to T task logits
        self.light_ranker = nn.Linear(
            in_features=2 * item_id_embedding_dim + self.num_ranker_user_embeddings + 1,
            out_features=len(user_value_weights),
        )

    def compute_user_embedding(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mips and light ranker user embeddings.

        Args:
            user_id: the user id
            user_features: the user features. We are assuming these are all dense features.
                In practice you will probably want to support sparse embedding features as well.
            user_history: for each user, the history of items they have interacted with.
                This is a tensor of item ids. Here we are assuming that the history is
                a fixed length, but in practice you will probably want to support variable
                length histories. jagged tensors are a good way to do this.
                This is NOT USED in this implementation. It is handled in a follow on derived class.

        Returns:
            torch.Tensor: Tensor containing mips query user embeddings. Shape: [B, DI]
            torch.Tensor: Tensor containing light ranker user embeddings. Shape: [B, NU, DI]
        """
        user_tower_input = self.process_user_features(
            user_id=user_id, user_features=user_features
        )

        # Compute the mips step user embedding
        mips_user_embedding = self.user_tower_arch(user_tower_input)  # [B, DI]

        # Compute the light ranker user embeddings
        light_ranker_user_embeddings = self.ranker_user_tower(
            user_tower_input
        )  # [B, NU * DI]
        light_ranker_user_embeddings = light_ranker_user_embeddings.view(
            light_ranker_user_embeddings.size(0),
            self.num_ranker_user_embeddings,
            self.item_id_embedding_dim,
        )  # [B, NU, DI]
        return mips_user_embedding, light_ranker_user_embeddings

    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        """This is used for inference.

        1. Compute the user embedding for mips
        2. Get num_mips_items items and their embeddings using the mips module
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
        mips_user_embedding, light_ranker_user_embeddings = self.compute_user_embedding(
            user_id, user_features, user_history
        )  # [B, DI], [B, NU, DI]
        # Query the mips module to get the top num_items items and their embeddings
        mips_items, mips_scores, mips_item_emb = self.mips_module(
            mips_user_embedding, self.num_mips_items
        )
        # mips_items: [B, num_mips_items=NI],
        # mips_scores: [B, NI]
        # mips_item_emb: [B, NI, DI]

        # Compute light ranker input
        # Dot product for each user embedding with each item embedding
        scores = torch.bmm(
            light_ranker_user_embeddings, mips_item_emb.permute(0, 2, 1)
        )  # [B, NU, NI]
        scores = scores.permute(0, 2, 1)  # [B, NI, NU]
        # Convert to probabilities
        probs = F.softmax(scores, dim=2)  # [B, NI, NU]
        # Compute weighted sum of user embeddings
        target_aware_user_embeddings = torch.bmm(
            probs, light_ranker_user_embeddings
        )  # [B, NI, DI]

        # Concatenate
        # mips_item_emb to capture item specific info
        # target_aware_user_embeddings captures importance weights of each user embedding
        # scores to capture mutual info of each user embedding with item (~ Factoization Machines)
        # mips_scores to capture the relevance of the mips user embedding.
        concatenated_input = torch.cat(
            [
                mips_item_emb,  # [B, NI, DI]
                target_aware_user_embeddings,  # [B, NI, DI]
                scores,  # [B, NI, NU]
                mips_scores.unsqueeze(2),  # [B, NI, 1]
            ],
            dim=2,
        )  # [B, NI, 2 * DI + NU + 1]

        # 3. Compute pointwise immediate reward estimates for each of these items
        task_logits = self.light_ranker(concatenated_input)  # [B, num_mips_items, T]

        # 4. Combine these rewards into a single reward estimate for each item
        net_user_value = torch.sum(
            task_logits * torch.tensor(self.user_value_weights), dim=2
        )  # [B, num_mips_items]

        # 5. Select the top num_items items based on this reward estimate
        _, top_indices = torch.topk(
            net_user_value, k=self.num_items, dim=1
        )  # [B, num_items]

        # Gather the top items
        top_items = torch.gather(mips_items, dim=1, index=top_indices)

        return top_items

    def train_forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        position: torch.Tensor,  # [B]
        labels: torch.Tensor,  # [B, T]
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
        # Compute user embeddings
        mips_user_embedding, light_ranker_user_embeddings = self.compute_user_embedding(
            user_id, user_features, user_history
        )  # [B, DI], [B, NU, DI]
        # Compute item embeddings
        item_embeddings = self.compute_item_embeddings(
            item_id, item_features
        )  # [B, DI]

        # Compute the mips step loss
        # Compute the scores for every pair of user and item
        mips_scores = torch.matmul(mips_user_embedding, item_embeddings.t())  # [B, B]

        # You should either try to handle the popularity bias
        # of in-batch negatives using log-Q correction or
        # use random negatives. Mixed Negative Sampling paper suggests
        # random negatives is a better approach.
        # Here we are not implementing either due to time constraints.

        # Compute softmax loss
        mips_step_target = torch.arange(
            mips_scores.shape[0], device=mips_scores.device
        )  # [B]
        # We are not reducing to mean since not every row in the batch is a
        # "positive" example. We are weighting the loss by the net_user_value
        # after this to give more weight to the positive examples and possibly
        # 0 weight to the hard-negative examples.
        mips_softmax_loss = F.cross_entropy(
            input=mips_scores, target=mips_step_target, reduction="none"
        )  # [B]

        # Compute the weighted average of the labels using user_value_weights
        # In the simplest case, assume you have a single label per item.
        # This label is either 1 or 0 depending on whether the user engaged
        # with this item when recommended. Then the net_user_value is non zero
        # actually exactly 1 when the user engaged with the item and 0 otherwise.
        net_user_value = torch.matmul(labels, self.user_value_weights)  # [B]

        # Optionally debias the net_user_value by the part explained purely
        # by position. Not implemented in this version. Hence net_user_value
        # is unchanged and additional_loss is 0.
        net_user_value, additional_loss = self.debias_net_user_value(
            net_user_value, position, mips_user_embedding
        )  # [B], [1]

        # Floor by epsilon to only preserve positive net_user_value
        net_user_value = torch.clamp(
            net_user_value, min=0.000001  # small epsilon to avoid divide by 0
        )  # [B]

        # Compute the product of loss and net_user_value, so that rows with
        # higher net_user_value get more signal.
        mips_softmax_loss = mips_softmax_loss * net_user_value  # [B]
        mips_softmax_loss = torch.mean(mips_softmax_loss)  # [1]
        # Optionally add the position bias loss to the loss
        mips_softmax_loss = mips_softmax_loss + additional_loss

        # Compute the light ranker loss

        # Compute light ranker input
        # Dot product for each user embedding with each item embedding
        ranker_scores = torch.bmm(
            light_ranker_user_embeddings,  # [B, NU, DI]
            item_embeddings.unsqueeze(2),  # [B, DI, 1]
        )  # [B, NU, 1]
        ranker_scores = ranker_scores.squeeze(2)  # [B, NU]
        # Convert to probabilities
        ranker_probs = F.softmax(ranker_scores, dim=1)  # [B, NU]
        # Compute weighted sum of user embeddings
        target_aware_user_embeddings = torch.bmm(
            ranker_probs.unsqueeze(1),  # [B, 1, NU]
            light_ranker_user_embeddings,  # [B, NU, DI]
        )  # [B, 1, DI]
        target_aware_user_embeddings = target_aware_user_embeddings.squeeze(
            1
        )  # [B, DI]

        # Take diagonal of mips_scores to get the relevance for impresed items
        mips_scores = torch.diag(mips_scores)  # [B]
        # Concatenate
        # mips_item_emb to capture item specific info
        # target_aware_user_embeddings captures importance weights of each user embedding
        # scores to capture mutual info of each user embedding with item (~ Factoization Machines)
        # mips_scores.diag to capture the relevance of the mips user embedding.
        concatenated_input = torch.cat(
            [
                item_embeddings,  # [B, DI]
                target_aware_user_embeddings,  # [B, DI]
                ranker_scores,  # [B, NU]
                mips_scores.unsqueeze(1),  # [B, 1]
            ],
            dim=2,
        )  # [B, 2 * DI + NU + 1]

        # 3. Compute pointwise immediate reward estimates for each of these items
        task_logits = self.light_ranker(concatenated_input)  # [B, T]

        # Compute binary cross entropy (BCE) loss with labels
        light_ranker_bce_loss = F.binary_cross_entropy_with_logits(task_logits, labels)
        return mips_softmax_loss + light_ranker_bce_loss
