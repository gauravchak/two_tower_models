"""
This is an example of how to encoder user history for two tower retrieval in recommender systems.
We use positional embeddings and target-indepedent multi-head attention.
"""

import torch
import torch.nn as nn
import math


class UserHistoryEncoder(nn.Module):
    """
    Given user history [B, H, DI], compute a summary
    using positional embeddings similar to
    Attention is all you need paper.
    """

    def __init__(
        self,
        item_id_embedding_dim: int,
        history_len: int,
        num_attention_heads: int,
        num_attention_layers: int,
        use_positional_encoding: bool,
    ) -> None:
        super().__init__()
        self.item_id_embedding_dim = item_id_embedding_dim
        self.history_len = history_len
        self.num_attention_heads = num_attention_heads
        self.num_attention_layers = num_attention_layers
        self.use_positional_encoding = use_positional_encoding

        # Create positional embeddings of shape [H, DI]
        if self.use_positional_encoding:
            self.positional_embeddings = self.positional_encoding(
                seq_len=history_len, d_model=item_id_embedding_dim
            )
            # The purpose of flipping the positional embeddings below is to match
            # the assumption made about the user history sequence. In the comments,
            # it's mentioned that the newest item in the user history sequence is
            # assumed to be at the beginning of the sequence. However, positional
            # encodings are typically designed with the assumption that the first
            # position corresponds to the earliest item in the sequence.
            # By flipping the positional embeddings, the model aligns the positional
            # encoding with the assumption about the user history sequence, ensuring
            # that the positional encoding reflects the correct position of each
            # item in the sequence. Without flipping, the positional encoding might
            # assign higher weights to the earliest positions, which would be
            # incorrect given the assumption about the sequence order.
            # In summary, flipping the positional embeddings ensures that the
            # positional encoding correctly reflects the position of each item in
            # the user history sequence, aligning with the assumption that the
            # newest item is at the beginning of the sequence.
            self.positional_embeddings = self.positional_embeddings.flip([0])

        # Create the multi-head attention module
        # Note: PyTorch's MultiheadAttention expects input shape
        # (seq_len=H, batch_size=B, d_model=DI)
        # so we have to permute the dimensions when using this.
        self.multihead_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=item_id_embedding_dim, num_heads=self.num_attention_heads
                )
                for _ in range(self.num_attention_layers)
            ]
        )

    def positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        PE = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                PE[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    PE[pos, i + 1] = math.cos(
                        pos / (10000 ** ((2 * (i + 1)) / d_model))
                    )
        return PE

    def forward(self, user_history: torch.Tensor) -> torch.Tensor:
        """
        params:
            user_history: [B, H, DI] the newest item is assumed to be at
                the beginning of the sequence. Here H is the history length.

        returns [B, 2, DI] a summary of the user history
        """
        # Get mean pooling of the history
        mean_pooled_history_encoding = torch.mean(user_history, dim=1)  # [B, DI]

        if self.use_positional_encoding:
            # Add positional encodings to history embeddings
            # Since positional encodings are [H, DI] and history embeddings are [B, H, DI]
            # we need to unsqueeze the positional embeddings to [1, H, DI] and add them
            user_history = user_history + self.positional_embeddings.unsqueeze(0)

        # Compute multi-head attention
        # Note: PyTorch's MultiheadAttention returns attn_output and
        # attn_output_weights, we only keep attn_output.
        # Since user_history : [B, H, DI]
        # user_history.permute(1, 0, 2) : [H, B, DI]
        user_history_permuted = user_history.permute(1, 0, 2)
        for layer in self.multihead_attn_layers:
            user_history_permuted, _ = layer(
                query=user_history_permuted,
                key=user_history_permuted,
                value=user_history_permuted,
            )

        # Convert user_history_permuted back to (B, H, DI) format
        # user_history_permuted : [H, B, DI]
        # Hence user_history_permuted.permute(1, 0, 2) : [B, H, DI]
        user_history = user_history_permuted.permute(1, 0, 2)

        # We will only take the first (most recent) item
        most_recent_item_encoding = user_history[:, 0, :].squeeze(1)  # [B, DI]
        # Stack the first item and the mean value
        user_history_summary = torch.stack(
            [most_recent_item_encoding, mean_pooled_history_encoding], dim=1
        )  # [B, 2, DI]
        return user_history_summary

    def get_output_dim(self) -> int:
        return self.item_id_embedding_dim * 2
