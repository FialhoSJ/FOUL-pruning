"""Shared protocol for annotated-subclass experiments."""

import hashlib
import json

import numpy as np

from utils.hierarchical_data import read_hierarchical_client_data


def split_indices(breeds, fraction, seed):
    """Stratify within each client; keep at least one training item per breed."""
    if not 0 <= fraction < 1:
        raise ValueError("validation fraction must be in [0, 1)")
    breeds = np.asarray(breeds)
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for breed in np.unique(breeds):
        indices = np.flatnonzero(breeds == breed)
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, int(round(len(indices) * fraction)))) if fraction else 0
        validation.extend(indices[:count])
        training.extend(indices[count:])
    return np.asarray(sorted(training), dtype=int), np.asarray(sorted(validation), dtype=int)


def load_pet_split(dataset, client_id, root, split="train", fraction=0.0, seed=1):
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown pet split: {split}")
    data = read_hierarchical_client_data(dataset, client_id, split != "test", root)
    if split == "test" or (split == "train" and fraction == 0):
        return data
    train, val = split_indices(data["y_subclass"].numpy(), fraction, seed + client_id)
    indices = val if split == "validation" else train
    return {key: tensor[indices] for key, tensor in data.items()}


def dataset_signature(config):
    keys = ("dataset", "family_to_idx", "breed_to_idx", "parent_by_breed", "image_size")
    encoded = json.dumps({key: config.get(key) for key in keys}, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def apply_encoder_policy(model, policy):
    """CNNs always update all layers; ResNets have an explicit encoder policy."""
    if not hasattr(model, "backbone"):
        if policy not in ("preserve", "full"):
            raise ValueError("The CNN protocol requires a fully trainable encoder")
        model.requires_grad_(True)
        return "full"
    if policy == "preserve":
        return getattr(model, "encoder_policy", "frozen" if model.encoder_frozen else "last-block")
    if policy not in ("frozen", "last-block", "full"):
        raise ValueError(f"Unknown encoder policy: {policy}")
    model.freeze_encoder()
    if policy == "last-block":
        model.unfreeze_last_block()
    elif policy == "full":
        model.unfreeze_encoder()
    return policy
