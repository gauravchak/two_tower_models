"""
This is an example of how to encoder user history for two tower retrieval in recommender systems.
We use positional embeddings and target-indepedent multi-head attention.
"""

import torch
import torch.nn as nn
import math


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

    def positional_encoding(
        self,
        seq_len:int, 
        d_model:int
    ) -> torch.Tensor:
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

        returns [B, 2, DI] a summary of the user history
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
        # Stack the first item and the mean value
        user_history_summary = torch.stack(
            [first_item, mean_value], dim=1
        )  # [B, 2, DI]
        return user_history_summary


