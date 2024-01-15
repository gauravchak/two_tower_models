"""
This is a specific instance of a two-tower based candidate generator (retrieval)
in a recommender system.
Ref: https://recsysml.substack.com/p/two-tower-models-for-retrieval-of
"""

from typing import List
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class UserHistoryEncoder(nn.Module):
    """
    Given user history [B, H, DI], compute a summary using positional embeddings.
    """
    def __init__(
        self,
        item_id_embedding_dim: int,
        history_len: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.item_id_embedding_dim = item_id_embedding_dim
        self.history_len = history_len
        self.num_heads = num_heads

        # Create positional embeddings of shape [H, DI]
        self.positional_embeddings = self.positional_encoding(
            seq_len = history_len,
            d_model = item_id_embedding_dim
        )
        # Reverse the positional encodings since in the user history we 
        # using this for, the newest item is at the beginning of the sequence.
        self.positional_embeddings = self.positional_embeddings.flip([0])

        # Create the multi-head attention module
        # Note: PyTorch's MultiheadAttention expects input shape 
        # (seq_len=H, batch_size=B, d_model=DI) 
        # so we have to permute the dimensions when using this.
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=item_id_embedding_dim,
            num_heads=num_heads
        )

    def positional_encoding(seq_len, d_model):
        PE = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                PE[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                if i + 1 < d_model:
                    PE[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        return PE

    def forward(
        self,
        user_history: torch.Tensor
    ) -> torch.Tensor:
        """
        params:
            user_history: [B, H, DI] the newest item is assumed to be at
                the beginning of the sequence.

        returns [B, 2 DI] a summary of the user history
        """
        # Add positional encodings to history embeddings
        # Since positional encodings are [H, DI] and history embeddings are [B, H, DI]
        # we need to unsqueeze the positional embeddings to [1, H, DI] and add them
        user_history = user_history + self.positional_embeddings.unsqueeze(0)

        # Compute multi-head attention
        # Note: PyTorch's MultiheadAttention returns attn_output and 
        # attn_output_weights, we only keep attn_output.
        attn_output, _ = self.multihead_attn(
            query=user_history.permute(1, 0, 2),
            key=user_history.permute(1, 0, 2),
            value=user_history.permute(1, 0, 2)
        )

        # Convert attn_output back to (B, H, DI) format
        attn_output = attn_output.permute(1, 0, 2)

        # We will only take the first (most recent) item and the mean value
        first_item = attn_output[:, 0, :].squeeze(1)
        mean_value = torch.mean(attn_output, dim=1)
        # Stack the first item and the mean value [B, 2, DI]
        user_history_summary = torch.stack(
            [first_item, mean_value], dim=1
        )
        return user_history_summary


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
        knn_module: nn.Module,
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
        """
        super().__init__()
        self.num_items = num_items
        self.user_value_weights = torch.tensor(user_value_weights)  # noqa TODO add device input.

        # Embedding layers for user and item ids
        self.user_id_embedding_arch = nn.Embedding(
            user_id_hash_size, user_id_embedding_dim)
        self.item_id_embedding_arch = nn.Embedding(
            item_id_hash_size, item_id_embedding_dim)
        # Create an arch to process the user_features
        self.user_features_arch = nn.Linear(
            user_features_size, user_id_embedding_dim)
        # Create an arch to process the user_tower_input
        self.user_tower_arch = nn.Linear(
            2 * user_id_embedding_dim + item_id_embedding_dim, user_id_embedding_dim)
        # Create an arch to process the item_features
        self.item_features_arch = nn.Linear(
            item_features_size, item_id_embedding_dim)
        # Create an arch to process the item_tower_input
        self.item_tower_arch = nn.Linear(
            item_id_embedding_dim, item_id_embedding_dim)

    def compute_user_embedding(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        """
        Compute the user embedding .
        params:
            user_id: the user id
            user_features: the user features. We are assuming these are all dense features.
                In practice you will probably want to support sparse embedding features as well.
            user_history: for each user, the history of items they have interacted with.
                This is a tensor of item ids. Here we are assuming that the history is
                a fixed length, but in practice you will probably want to support variable
                length histories. jagged tensors are a good way to do this.
        """
        # Pass the user history through the item embedding layer
        user_history_embedding = self.item_id_embedding_arch(user_history)  # [B, H, DI]
        # Pass the user history through the user history encoder
        user_history_summary = self.user_history_encoder(user_history_embedding)  # [B, 2 DI]
        # Process user id
        user_id_embedding = self.user_id_embedding_arch(user_id)  # [B, DU]
        # Process user features
        user_features_embedding = self.user_features_arch(user_features)  # [B, DU]
        # Concatenate the inputs and pass them through a linear layer to compute the user embedding
        user_tower_input = torch.cat(
            [user_id_embedding, user_features_embedding, user_history_summary], dim=1
        )
        # Compute the user embedding
        user_embedding = self.user_tower_arch(user_tower_input)  # [B, DU]
        return user_embedding

    def compute_item_embeddings(
        self,
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
    ) -> torch.Tensor:
        """
        Process item_id and item_features to compute item embeddings.
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
        item_embedding = self.item_tower_arch(item_tower_input)
        return item_embedding

    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        """
        and return the top num_items items using the KNN module
        """
        # Compute the user embedding
        user_embedding = self.compute_user_embedding(
            user_id, user_features, user_history
        )
        # Query the knn module to get the top num_items items
        top_items = self.knn_module(user_embedding, self.num_items)  # [B, num_items]
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
        """Compute the loss during training"""
        # Compute the user embedding
        user_embedding = self.compute_user_embedding(
            user_id, user_features, user_history
        )  # [B, DU]
        # Compute item embeddings
        item_embeddings = self.compute_item_embeddings(
            item_id, item_features
        )  # [B, DI]
        # Compute the scores for every pair of user and item
        scores = torch.matmul(user_embedding, item_embeddings.t())

        # You should either try to handle the popularity bias 
        # of in batch negatives using log-Q correction or
        # use random negatives.
        # Here we are skipping this part.

        # Compute softmax loss
        target = torch.arange(scores.shape[0]).to(scores.device)
        loss = F.cross_entropy(
            input=scores,
            target=target,
            reduction="mean"
        )
        return loss

