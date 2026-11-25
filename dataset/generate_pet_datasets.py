"""Prepare Oxford-IIIT Pet or Stanford Dogs for hierarchical federated FOUL."""

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
from PIL import Image


def _resize_rgb(image, image_size):
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.convert("RGB")
    width, height = image.size
    short_side = max(image_size, int(round(image_size * 256 / 224)))
    scale = short_side / min(width, height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    image = image.resize((resized_width, resized_height), resampling)
    left = (resized_width - image_size) // 2
    top = (resized_height - image_size) // 2
    image = image.crop((left, top, left + image_size, top + image_size))
    return np.asarray(image, dtype=np.uint8)


def _class_maps(family_names, breed_names, breed_families):
    family_to_idx = {name: index for index, name in enumerate(sorted(set(family_names)))}
    breed_to_idx = {name: index for index, name in enumerate(sorted(set(breed_names)))}
    breed_parent = {
        str(breed_to_idx[breed]): family_to_idx[family]
        for breed, family in breed_families.items()
    }
    return family_to_idx, breed_to_idx, breed_parent


def _partition_by_breed(records, num_clients, seed, mode="iid", alpha=0.5):
    """Return a deterministic client -> record-index allocation."""
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")
    if mode not in ("iid", "dirichlet"):
        raise ValueError("partition must be 'iid' or 'dirichlet'")
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0")

    rng = np.random.default_rng(seed)
    allocation = [[] for _ in range(num_clients)]
    by_breed = {}
    for index, record in enumerate(records):
        by_breed.setdefault(record["breed"], []).append(index)

    for breed in sorted(by_breed):
        indices = np.asarray(by_breed[breed], dtype=np.int64)
        rng.shuffle(indices)
        if mode == "iid":
            for offset, index in enumerate(indices):
                allocation[offset % num_clients].append(int(index))
        else:
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            boundaries = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            for client_id, client_indices in enumerate(np.split(indices, boundaries)):
                allocation[client_id].extend(int(index) for index in client_indices)

    for indices in allocation:
        rng.shuffle(indices)
    return allocation


def _pack_records(records, family_to_idx, breed_to_idx, breed_parent, image_size):
    images, families, breeds = [], [], []
    for record in records:
        family, breed, image = record["family"], record["breed"], record["image"]
        if family not in family_to_idx or breed not in breed_to_idx:
            raise ValueError(f"Unknown family/breed label in sample: {family!r}/{breed!r}")
        images.append(_resize_rgb(image, image_size))
        families.append(family_to_idx[family])
        breeds.append(breed_to_idx[breed])
    return (
        np.asarray(images, dtype=np.uint8),
        np.asarray(families, dtype=np.int64),
        np.asarray(breeds, dtype=np.int64),
    )


def _save_split(output_dir, split, images, families, breeds, allocation):
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    counts = []
    for client_id, record_indices in enumerate(allocation):
        indices = np.asarray(record_indices, dtype=np.int64)
        data = {
            "x": images[indices],
            "y": families[indices],
            "y_subclass": breeds[indices],
        }
        np.savez_compressed(os.path.join(split_dir, f"{client_id}.npz"), data=data)
        unique, frequencies = np.unique(data["y_subclass"], return_counts=True)
        counts.append({str(int(label)): int(count) for label, count in zip(unique, frequencies)})
    return counts


def _write_dataset(output_dir, dataset_name, train_records, test_records,
                   family_names, breed_names, breed_families, num_clients,
                   seed, partition, alpha, image_size):
    family_to_idx, breed_to_idx, breed_parent = _class_maps(
        family_names, breed_names, breed_families
    )
    unknown_breeds = set(breed_names) - set(breed_families)
    if unknown_breeds:
        raise ValueError(f"Breeds are missing parent-family mappings: {sorted(unknown_breeds)}")

    train_images, train_families, train_breeds = _pack_records(
        train_records, family_to_idx, breed_to_idx, breed_parent, image_size
    )
    test_images, test_families, test_breeds = _pack_records(
        test_records, family_to_idx, breed_to_idx, breed_parent, image_size
    )
    train_allocation = _partition_by_breed(train_records, num_clients, seed, partition, alpha)
    test_allocation = _partition_by_breed(test_records, num_clients, seed + 1, partition, alpha)

    parent_by_breed = {
        breed: family_to_idx[breed_families[breed]] for breed in sorted(breed_to_idx)
    }
    config = {
        "dataset": dataset_name,
        "num_clients": num_clients,
        "num_classes": len(family_to_idx),
        "num_breeds": len(breed_to_idx),
        "family_to_idx": family_to_idx,
        "breed_to_idx": breed_to_idx,
        "parent_by_breed": parent_by_breed,
        "partition": partition,
        "alpha": alpha if partition == "dirichlet" else None,
        "seed": seed,
        "image_size": image_size,
        "train_samples_per_client_by_breed": None,
        "test_samples_per_client_by_breed": None,
    }

    parent = os.path.dirname(os.path.abspath(output_dir))
    os.makedirs(parent, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=f".{dataset_name}-", dir=parent)
    try:
        train_counts = _save_split(
            staging_dir, "train", train_images, train_families, train_breeds, train_allocation
        )
        test_counts = _save_split(
            staging_dir, "test", test_images, test_families, test_breeds, test_allocation
        )
        config["train_samples_per_client_by_breed"] = train_counts
        config["test_samples_per_client_by_breed"] = test_counts
        with open(os.path.join(staging_dir, "config.json"), "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        if os.path.exists(output_dir):
            raise FileExistsError(
                f"Output already exists: {output_dir}. Choose a new --output directory."
            )
        shutil.move(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(f"Wrote {dataset_name} data for {num_clients} clients to {output_dir}")
    for client_id, per_breed in enumerate(train_counts):
        print(f"client {client_id}: {per_breed}")


def _load_oxford(root, download):
    from torchvision.datasets import OxfordIIITPet

    try:
        train_dataset = OxfordIIITPet(
            root=root, split="trainval", target_types=["category", "binary-category"], download=download
        )
        test_dataset = OxfordIIITPet(
            root=root, split="test", target_types=["category", "binary-category"], download=download
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load Oxford-IIIT Pet under {root}: {exc}") from exc

    def make_records(dataset):
        records = []
        for index in range(len(dataset)):
            image, labels = dataset[index]
            breed_id, family_id = (int(label) for label in labels)
            records.append({
                "image": image,
                "family": dataset.bin_classes[family_id].lower(),
                "breed": dataset.classes[breed_id],
            })
        return records

    train_records, test_records = make_records(train_dataset), make_records(test_dataset)
    breed_families = {}
    for record in train_records + test_records:
        existing = breed_families.setdefault(record["breed"], record["family"])
        if existing != record["family"]:
            raise ValueError(f"Breed {record['breed']} has conflicting family labels")
    return train_records, test_records, breed_families


def _mat_strings(values):
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, np.ndarray):
            value = "".join(str(part) for part in value.reshape(-1))
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value).strip())
    return result


def _load_stanford(root):
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("Stanford Dogs split loading requires scipy; install scikit-learn dependencies") from exc

    image_root = os.path.join(root, "Images")
    train_path = os.path.join(root, "lists", "train_list.mat")
    test_path = os.path.join(root, "lists", "test_list.mat")
    for required_path in (image_root, train_path, test_path):
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Stanford Dogs source path not found: {required_path}")

    breed_dirs = sorted(
        name for name in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, name))
    )
    if not breed_dirs:
        raise ValueError(f"No breed folders found under {image_root}")

    breed_names = {
        folder: folder.split("-", 1)[-1].replace("_", " ") for folder in breed_dirs
    }

    def load_split(mat_path):
        payload = loadmat(mat_path)
        if "file_list" not in payload or "labels" not in payload:
            raise ValueError(f"{mat_path} must contain file_list and labels")
        relative_paths = _mat_strings(payload["file_list"])
        labels = np.asarray(payload["labels"]).reshape(-1).astype(np.int64)
        if len(relative_paths) != len(labels):
            raise ValueError(f"{mat_path} has unaligned file_list and labels")
        records = []
        for relative_path, raw_label in zip(relative_paths, labels):
            relative_path = relative_path.replace("\\", "/")
            folder = relative_path.split("/", 1)[0]
            if folder not in breed_names:
                raise ValueError(f"{mat_path} names unknown breed folder {folder!r}")
            expected_label = breed_dirs.index(folder) + 1
            if int(raw_label) != expected_label:
                raise ValueError(
                    f"{mat_path} has label {int(raw_label)} for {relative_path}; expected {expected_label}"
                )
            image_path = os.path.join(image_root, *relative_path.split("/"))
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Stanford Dogs image listed in {mat_path} is missing: {image_path}")
            records.append({
                "image": Image.open(image_path).convert("RGB"),
                "family": "dog",
                "breed": breed_names[folder],
            })
        return records

    train_records, test_records = load_split(train_path), load_split(test_path)
    breed_families = {name: "dog" for name in breed_names.values()}
    return train_records, test_records, breed_families


def generate(args):
    if args.dataset == "OxfordPet":
        train_records, test_records, breed_families = _load_oxford(args.root, args.download)
    elif args.dataset == "StanfordDogs":
        train_records, test_records, breed_families = _load_stanford(args.root)
    else:
        raise ValueError("dataset must be OxfordPet or StanfordDogs")

    all_records = train_records + test_records
    family_names = sorted({record["family"] for record in all_records})
    breed_names = sorted({record["breed"] for record in all_records})
    if set(breed_names) - set(breed_families):
        raise ValueError("Some source breeds are missing family annotations")

    _write_dataset(
        args.output,
        args.dataset,
        train_records,
        test_records,
        family_names,
        breed_names,
        breed_families,
        args.num_clients,
        args.seed,
        args.partition,
        args.alpha,
        args.image_size,
    )


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("OxfordPet", "StanfordDogs"), required=True)
    parser.add_argument("--root", required=True, help="Source data root; Oxford downloads here when requested")
    parser.add_argument("--output", default=None, help="Output directory under dataset/")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--partition", choices=("iid", "dirichlet"), default="iid")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--download", action="store_true", help="Download Oxford-IIIT Pet if missing")
    return parser


if __name__ == "__main__":
    arguments = make_parser().parse_args()
    if arguments.output is None:
        arguments.output = os.path.join(os.path.dirname(os.path.abspath(__file__)), arguments.dataset)
    generate(arguments)
