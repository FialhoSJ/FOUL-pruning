"""Two-head pet classifier with an ImageNet-pretrained ResNet18 encoder."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class HierarchicalResNet18(nn.Module):
    """Configurable visual encoder and compact, prunable task-specific projection.

    ``forward`` also accepts cached 512-dimensional encoder features. This lets
    CPU-only federated experiments train and unlearn without repeatedly running
    the frozen ResNet for each local update.
    """

    def __init__(self, num_families, num_breeds, image_size=128, hidden_dim=256,
                 pretrained=True):
        super().__init__()
        if min(num_families, num_breeds, hidden_dim) < 1:
            raise ValueError("family, breed, and projection dimensions must be positive")
        self.num_families = int(num_families)
        self.num_breeds = int(num_breeds)
        self.image_size = int(image_size)
        self.hidden_dim = int(hidden_dim)
        self.initialization = "imagenet" if pretrained else "random"
        self.backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone.fc = nn.Identity()
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.encoder_frozen = True
        self.encoder_policy = "frozen"
        self.fc1 = nn.Linear(512, self.hidden_dim)
        self.family_head = nn.Linear(self.hidden_dim, self.num_families)
        self.breed_head = nn.Linear(self.hidden_dim, self.num_breeds)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # Frozen BatchNorm statistics must not drift per client.
        return self

    def encode(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("ResNet18 encoder expects NCHW RGB images")
        with torch.set_grad_enabled(self.training and not self.encoder_frozen):
            return self.backbone((images - self.image_mean) / self.image_std)

    def unfreeze_last_block(self):
        self.backbone.requires_grad_(False)
        self.backbone.layer4.requires_grad_(True)
        self.encoder_frozen = False
        self.encoder_policy = "last-block"

    def unfreeze_encoder(self):
        self.backbone.requires_grad_(True)
        self.encoder_frozen = False
        self.encoder_policy = "full"

    def freeze_encoder(self):
        self.backbone.requires_grad_(False)
        self.encoder_frozen = True
        self.encoder_policy = "frozen"

    def forward_features(self, images_or_features):
        if images_or_features.ndim == 4:
            features = self.encode(images_or_features)
        elif images_or_features.ndim == 2 and images_or_features.shape[1] == 512:
            if not self.encoder_frozen:
                raise ValueError("An updating encoder requires raw images, not cached features")
            features = images_or_features
        else:
            raise ValueError("Expected RGB images or 512-dimensional ResNet18 features")
        return F.relu(self.fc1(features))

    def forward(self, images_or_features):
        features = self.forward_features(images_or_features)
        return self.family_head(features), self.breed_head(features)

    def feature_layers(self):
        return (("fc1", self.fc1),)

    def compact_config(self):
        return {
            "num_families": self.num_families,
            "num_breeds": self.num_breeds,
            "image_size": self.image_size,
            "hidden_dim": self.hidden_dim,
            "pretrained": False,
        }
