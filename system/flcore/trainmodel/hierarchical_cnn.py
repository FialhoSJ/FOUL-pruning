"""Compact CNN with separate family and fine-grained breed predictions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalCNN(nn.Module):
    """Shared image features with independent family and breed output heads."""

    def __init__(
        self,
        num_families,
        num_breeds,
        image_size=128,
        channels=(32, 64, 128),
        hidden_dim=256,
        pool_grid=4,
    ):
        super().__init__()
        if num_families < 1 or num_breeds < 1:
            raise ValueError("num_families and num_breeds must both be positive")
        if image_size < 32:
            raise ValueError("image_size must be at least 32")
        if len(channels) != 3 or any(width < 1 for width in channels):
            raise ValueError("channels must contain three positive widths")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if pool_grid < 1:
            raise ValueError("pool_grid must be positive")

        self.num_families = int(num_families)
        self.num_breeds = int(num_breeds)
        self.image_size = int(image_size)
        self.channels = tuple(int(width) for width in channels)
        self.hidden_dim = int(hidden_dim)
        self.pool_grid = int(pool_grid)
        self.initialization = "random"

        self.conv1 = nn.Conv2d(3, self.channels[0], kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((self.pool_grid, self.pool_grid))
        self.fc1 = nn.Linear(self.channels[2] * self.pool_grid ** 2, self.hidden_dim)
        self.family_head = nn.Linear(self.hidden_dim, self.num_families)
        self.breed_head = nn.Linear(self.hidden_dim, self.num_breeds)

    def forward_features(self, images):
        features = self.pool(F.relu(self.conv1(images), inplace=False))
        features = self.pool(F.relu(self.conv2(features), inplace=False))
        features = F.relu(self.conv3(features), inplace=False)
        features = self.global_pool(features).flatten(1)
        return F.relu(self.fc1(features), inplace=False)

    def forward(self, images):
        features = self.forward_features(images)
        return self.family_head(features), self.breed_head(features)

    def feature_layers(self):
        """Return ordered layer names/modules used for activation scoring."""
        return (
            ("conv1", self.conv1),
            ("conv2", self.conv2),
            ("conv3", self.conv3),
            ("fc1", self.fc1),
        )

    def compact_config(self):
        """Return constructor settings for a shape-compatible compact copy."""
        return {
            "num_families": self.num_families,
            "num_breeds": self.num_breeds,
            "image_size": self.image_size,
            "channels": self.channels,
            "hidden_dim": self.hidden_dim,
            "pool_grid": getattr(self, "pool_grid", self.global_pool.output_size[0]),
        }
