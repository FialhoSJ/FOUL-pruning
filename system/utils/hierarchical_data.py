"""Reader for federated image datasets with parent and subclass labels."""

import json
import os

import numpy as np
import torch


def read_hierarchical_client_data(dataset, client_id, is_train=True, dataset_root="../dataset"):
    """Load one client's images and aligned parent/subclass labels.

    Files use the repository's ``data`` NPZ envelope. Its payload contains
    ``x`` (uint8 NHWC or NCHW), ``y`` (parent IDs), and ``y_subclass`` (breed
    IDs). The returned image tensor is normalized float32 NCHW.
    """
    split = "train" if is_train else "test"
    client_path = os.path.join(dataset_root, dataset, split, f"{client_id}.npz")
    config_path = os.path.join(dataset_root, dataset, "config.json")

    if not os.path.isfile(client_path):
        raise FileNotFoundError(
            f"Missing hierarchical data for client {client_id}: {client_path}"
        )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing dataset label map: {config_path}")

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    with np.load(client_path, allow_pickle=True) as archive:
        if "data" not in archive:
            raise ValueError(f"{client_path} is missing the 'data' field")
        payload = archive["data"].tolist()

    if not isinstance(payload, dict):
        raise ValueError(f"{client_path} must contain a dictionary in its 'data' field")

    missing = [key for key in ("x", "y", "y_subclass") if key not in payload]
    if missing:
        raise ValueError(f"{client_path} is missing required fields: {', '.join(missing)}")

    images = np.asarray(payload["x"])
    parents = np.asarray(payload["y"])
    subclasses = np.asarray(payload["y_subclass"])

    if images.ndim != 4:
        raise ValueError(f"{client_path} has image array with shape {images.shape}; expected 4D")
    if parents.ndim != 1 or subclasses.ndim != 1:
        raise ValueError(f"{client_path} labels must be one-dimensional arrays")
    if len(images) != len(parents) or len(images) != len(subclasses):
        raise ValueError(
            f"{client_path} has unaligned lengths: images={len(images)}, "
            f"family labels={len(parents)}, breed labels={len(subclasses)}"
        )
    if not np.issubdtype(parents.dtype, np.integer):
        raise ValueError(f"{client_path} contains non-integer family labels: {parents.dtype}")
    if not np.issubdtype(subclasses.dtype, np.integer):
        raise ValueError(f"{client_path} contains non-integer breed labels: {subclasses.dtype}")

    family_map = config.get("family_to_idx")
    breed_map = config.get("breed_to_idx")
    if not isinstance(family_map, dict) or not isinstance(breed_map, dict):
        raise ValueError(f"{config_path} must define family_to_idx and breed_to_idx mappings")

    family_ids = set(int(value) for value in family_map.values())
    breed_ids = set(int(value) for value in breed_map.values())
    invalid_families = sorted(set(parents.astype(np.int64).tolist()) - family_ids)
    invalid_breeds = sorted(set(subclasses.astype(np.int64).tolist()) - breed_ids)
    if invalid_families:
        raise ValueError(f"{client_path} contains unknown family IDs: {invalid_families}")
    if invalid_breeds:
        raise ValueError(f"{client_path} contains unknown breed IDs: {invalid_breeds}")

    if images.shape[-1] in (1, 3, 4):
        images = np.transpose(images, (0, 3, 1, 2))
    elif images.shape[1] not in (1, 3, 4):
        raise ValueError(
            f"{client_path} image array shape {images.shape} is neither NHWC nor NCHW"
        )

    image_tensor = torch.as_tensor(np.ascontiguousarray(images))
    if image_tensor.dtype == torch.uint8:
        image_tensor = image_tensor.to(torch.float32).div_(255.0)
    else:
        image_tensor = image_tensor.to(torch.float32)

    return {
        "x": image_tensor,
        "y": torch.as_tensor(parents, dtype=torch.int64),
        "y_subclass": torch.as_tensor(subclasses, dtype=torch.int64),
    }
