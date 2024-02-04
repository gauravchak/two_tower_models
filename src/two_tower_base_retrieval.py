"""
This is a baseline implementation of a two-tower based candidate generator (retrieval)
in a recommender system.
Ref: https://recsysml.substack.com/p/two-tower-models-for-retrieval-of

In training we create a user embedding from user features. In this example, we are
ignoring user history features. They will be handled in a follow on derived class.
We compute item embeddings for the items in the batch and use the user embedding
and item embeddings to compute a softmax loss. We assume the training data comprises
of all items impressed by the user. Hence it includes both positive and hard negatives.
We weight the loss by the net_user_value, which is a linear combination of point-wise
immediate rewards. Hence the loss is effectively derived only from the "positives", as
is assumed in two-tower models.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baseline_mips_module import BaselineMIPSModule


class TwoTowerBaseRetrieval(nn.Module):
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
        mips_module: BaselineMIPSModule,
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
            mips_module: a module that computes the Maximum Inner Product Search (MIPS)
                over the item embeddings given the user embedding.
        """
        super().__init__()
        self.num_items = num_items
        self.user_value_weights = torch.tensor(user_value_weights)  # noqa TODO add device input.
        self.mips_module = mips_module

        # Create the archs fo user tower
        # Embedding layers for user id
        self.user_id_embedding_arch = nn.Embedding(
            user_id_hash_size, user_id_embedding_dim
        )
        # Create an arch to process the user_features
        self.user_features_arch = nn.Sequential(
            nn.Linear(user_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, user_id_embedding_dim)
        )
        # Create an arch to process the user_tower_input
        # Input dimension = 
        #   user_id_embedding_dim from user_id_embedding_arch
        #   user_id_embedding_dim from user_features_arch
        self.user_tower_arch = nn.Linear(
            2 * user_id_embedding_dim + item_id_embedding_dim, 
            item_id_embedding_dim
        )

        # Create the archs for item tower
        # Embedding layers for item id
        self.item_id_embedding_arch = nn.Embedding(
            item_id_hash_size, item_id_embedding_dim
        )
        # Create an arch to process the item_features
        self.item_features_arch = nn.Sequential(
            nn.Linear(item_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, item_id_embedding_dim)
        )
        # Create an arch to process the item_tower_input
        self.item_tower_arch = nn.Linear(
            in_features=2 * item_id_embedding_dim,  # concat id and features
            out_features=item_id_embedding_dim
        )

    def process_user_features(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
    ) -> torch.Tensor:
        """
        Process the user features to compute the user embedding.

        Args:
            user_id (torch.Tensor): Tensor containing the user IDs. Shape: [B]
            user_features (torch.Tensor): Tensor containing the user features. Shape: [B, IU]

        Returns:
            torch.Tensor: Tensor representing the user embedding. Shape: [B, DU]
        """
        # Process user id
        user_id_embedding = self.user_id_embedding_arch(user_id)  # [B, DU]
        # Process user features
        user_features_embedding = self.user_features_arch(user_features)  # [B, DU]
        # Concatenate the inputs and pass them through a linear layer to compute the user embedding
        user_tower_input = torch.cat(
            [user_id_embedding, user_features_embedding], dim=1
        )
        return user_tower_input

    def compute_user_embedding(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        """
        Compute the user embedding. This will be used to query mips.

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
            torch.Tensor: Tensor containing query user embeddings. Shape: [B, DI]
        """
        user_tower_input = self.process_user_features(
            user_id=user_id, user_features=user_features
        )
        # Compute the user embedding
        user_embedding = self.user_tower_arch(user_tower_input)  # [B, DI]
        return user_embedding

    def compute_item_embeddings(
        self,
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
    ) -> torch.Tensor:
        """
        Process item_id and item_features to compute item embeddings.

        Args:
            item_id (torch.Tensor): Tensor containing item IDs. Shape: [B]
            item_features (torch.Tensor): Tensor containing item features. Shape: [B, II]

        Returns:
            torch.Tensor: Tensor containing item embeddings. Shape: [B, DI]
        """
        # Process item id
        item_id_embedding = self.item_id_embedding_arch(item_id)
        # Process item features
        item_features_embedding = self.item_features_arch(item_features)
        # Concatenate the inputs and pass them through a linear layer to compute the item embedding
        item_tower_input = torch.cat(
            [item_id_embedding, item_features_embedding], dim=1
        )
        # Compute the item embedding
        item_embedding = self.item_tower_arch(item_tower_input)  # [B, DI]
        return item_embedding

    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        """This is used for inference.

        Compute the user embedding and return the top num_items items using the mips module.

        Args:
            user_id (torch.Tensor): Tensor representing the user ID. Shape: [B]
            user_features (torch.Tensor): Tensor representing the user features. Shape: [B, IU]
            user_history (torch.Tensor): Tensor representing the user history. Shape: [B, H]

        Returns:
            torch.Tensor: Tensor representing the top num_items items. Shape: [B, num_items]
        """
        # Compute the user embedding
        user_embedding = self.compute_user_embedding(
            user_id, user_features, user_history
        )
        # Query the mips module to get the top num_items items and their embeddings
        top_items, _, _ = self.mips_module(
            query_embedding=user_embedding, 
            num_items=self.num_items
        )  # Returns indices [B, num_items], scores, embeddings
        return top_items

    def debias_net_user_value(
        self,
        net_user_value: torch.Tensor,  # [B]
        position: torch.Tensor,  # [B]
        user_embedding: torch.Tensor,  # [B, DI]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the processed net_user_value and any losses to be added
        to the loss function.

        Args:
            net_user_value (torch.Tensor): The net user value tensor [B].
            position (torch.Tensor): The position tensor of shape [B].
            user_embedding: same as what is used in MIPS  # [B, DI]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the
            processed net_user_value tensor and any losses to be added
            to the loss function.

        This is written as a function and not in train_forward to make
        it easier to implement in a derived class.
        """
        return net_user_value, 0

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
        # Compute the scores for every pair of user and item
        scores = torch.matmul(user_embedding, item_embeddings.t())  # [B, B]

        # You should either try to handle the popularity bias 
        # of in-batch negatives using log-Q correction or
        # use random negatives. Mixed Negative Sampling paper suggests
        # random negatives is a better approach.
        # Here we are not implementing either due to time constraints.

        # Compute softmax loss
        # F.cross_entropy accepts target as 
        #   ground truth class indices or class probabilities;
        # Here we are using class indices
        target = torch.arange(scores.shape[0]).to(scores.device)  # [B]
        # We are not reducing to mean since not every row in the batch is a 
        # "positive" example. We are weighting the loss by the net_user_value
        # after this to give more weight to the positive examples and possibly
        # 0 weight to the hard-negative examples.
        loss = F.cross_entropy(
            input=scores,
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
        # Normalize net_user_value by the max value of it in batch.
        # This is to ensure that the net_user_value is between 0 and 1.
        net_user_value = net_user_value / torch.max(net_user_value)  # [B]

        # Compute the product of loss and net_user_value
        loss = loss * net_user_value  # [B]
        loss = torch.mean(loss)  # ()
        # Optionally add the position bias loss to the loss
        if self.enable_position_debiasing:
            loss = loss + additional_loss

        return loss  # ()

