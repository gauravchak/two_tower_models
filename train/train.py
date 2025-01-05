#!/usr/bin/env python

"""
Example training script demonstrating how to call model.train_forward(...)
which returns the training loss directly.
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.two_tower_base_retrieval import TwoTowerBaseRetrieval
from src.baseline_mips_module import BaselineMIPSModule


# -------------------------------------------------
# Example dataset, generating random dummy data
# -------------------------------------------------
class DummyRecDataset(Dataset):
    """
    This dummy dataset simulates records that each contain:
      - user_ids
      - user_features
      - user_history
      - item_ids
      - item_features
      - position
      - labels
    """

    def __init__(
        self,
        num_samples: int,
        num_users: int,
        num_items: int,
        feature_dim: int,
        user_history_seqlen: int,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_users = num_users
        self.num_items = num_items
        self.feature_dim = feature_dim

        # Randomly generate data
        self.user_ids = torch.randint(
            0, num_users, (num_samples,)
        )  # [num_samples] values: 0 to num_users-1
        self.item_ids = torch.randint(
            0, num_items, (num_samples,)
        )  # [num_samples] values: 0 to num_items-1
        self.labels = torch.randint(
            0, 2, (num_samples,)
        ).float()  # [num_samples] values: 0 or 1

        # For demonstration, let's just generate random features, history, etc.
        self.user_features = torch.randn(num_samples, feature_dim)
        self.user_history = torch.randint(
            low=0, high=num_items, size=(num_samples, user_history_seqlen)
        )  # [num_samples, user_history_seqlen] values: 0 to num_items-1
        self.item_features = torch.randn(num_samples, feature_dim)

        # Positions (optional) if you have rank positions or something
        self.positions = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.user_features[idx],
            self.user_history[idx],
            self.item_ids[idx],
            self.item_features[idx],
            self.positions[idx],
            self.labels[idx],
        )


# -------------------------------------------------
# Training loop
# -------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()  # PyTorch training mode
    total_loss = 0.0

    for batch in dataloader:
        (
            user_ids,
            user_features,
            user_history,
            item_ids,
            item_features,
            positions,
            labels,
        ) = batch

        # Move to device
        user_ids = user_ids.to(device)
        user_features = user_features.to(device)
        user_history = user_history.to(device)
        item_ids = item_ids.to(device)
        item_features = item_features.to(device)
        positions = positions.to(device)
        labels = labels.to(device)

        # The train_forward function is where the model authors can define
        # the loss computation. They would typically call the forward function
        # and compute the loss based on the output and the labels.
        batch_loss = model.train_forward(
            user_ids,
            user_features,
            user_history,
            item_ids,
            item_features,
            positions,
            labels,
        )

        # Backprop & update
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Perform any update steps here that are separate from backprop.
        # For example, user_emb = beta * item_emb + (1-beta) * user_emb
        # from https://arxiv.org/abs/2403.18227

        # Accumulate loss
        total_loss += batch_loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Parameters for TwoTowerBaseRetrieval
    num_items_to_return: int = args.num_items_to_return
    user_id_hash_size: int = args.user_id_hash_size
    user_id_embedding_dim: int = args.embedding_dim
    user_features_size: int = args.feature_dim
    item_id_hash_size: int = args.item_id_hash_size
    item_id_embedding_dim: int = args.embedding_dim
    item_features_size: int = args.feature_dim
    # tasknum: int = 1 # Not used in this example
    mips_module = BaselineMIPSModule(
        corpus_size=args.num_items, embedding_dim=item_id_embedding_dim
    )

    # Instantiate model
    model = TwoTowerBaseRetrieval(
        num_items=num_items_to_return,
        user_id_hash_size=user_id_hash_size,
        user_id_embedding_dim=user_id_embedding_dim,
        user_features_size=user_features_size,
        item_id_hash_size=item_id_hash_size,
        item_id_embedding_dim=item_id_embedding_dim,
        item_features_size=item_features_size,
        user_value_weights=[1.0],
        mips_module=mips_module,
    ).to(device)

    # Create dummy dataset
    dataset = DummyRecDataset(
        num_samples=args.num_samples,
        num_users=args.num_users,
        num_items=args.num_items,
        feature_dim=args.feature_dim,
        user_history_seqlen=args.user_history_seqlen,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train two-tower retrieval model"
    )
    parser.add_argument(
        "--num_users",
        type=int,
        default=100,
        help="number of users in the dataset",
    )
    parser.add_argument(
        "--num_items_to_return",
        type=int,
        default=10,
        help="number of items to return in the retrieval task",
    )
    parser.add_argument(
        "--user_id_hash_size",
        type=int,
        default=1024,
        help="embedding table size for user_id",
    )
    parser.add_argument(
        "--item_id_hash_size",
        type=int,
        default=1024,
        help="embedding table size for item_id",
    )
    parser.add_argument(
        "--user_history_seqlen",
        type=int,
        default=10,
        help="length of user history sequence",
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=200,
        help="number of items in the corpus/dataset",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=32,
        help="Dimension of user/item embeddings",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=8,
        help="Dim of user_features, item_features, etc.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples in the dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size in training loop"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )

    args = parser.parse_args()
    main(args)
