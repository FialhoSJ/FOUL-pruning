"""Optional comparison between CLIP-selected visual clusters and breed labels."""

import numpy as np
import torch
import torch.nn.functional as F


def compare_prompt_cluster(
    images,
    breed_labels,
    target_breed_id,
    prompt,
    num_clusters=5,
    batch_size=32,
    seed=1,
    device="cpu",
):
    """Select the CLIP-nearest visual cluster and compare it with breed labels.

    ``images`` must be a 4D tensor in NCHW or NHWC format. Integer images are
    interpreted as uint8; floating-point images must be in [0, 1]. This helper
    is optional and never changes the annotated target set used by FOUL.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A non-empty CLIP prompt is required")
    if num_clusters < 2:
        raise ValueError("num_clusters must be at least 2")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    try:
        import open_clip
        from sklearn.cluster import KMeans
        from torchvision.transforms.functional import to_pil_image
    except ImportError as exc:
        raise RuntimeError(
            "CLIP comparison requires optional packages: install open_clip_torch and scikit-learn"
        ) from exc

    if not torch.is_tensor(images):
        images = torch.as_tensor(images)
    if images.ndim != 4:
        raise ValueError(f"images must be 4D, received shape {tuple(images.shape)}")
    if images.shape[-1] in (1, 3, 4):
        images = images.permute(0, 3, 1, 2)
    elif images.shape[1] not in (1, 3, 4):
        raise ValueError(f"cannot determine image channel axis for shape {tuple(images.shape)}")
    if len(images) < num_clusters:
        raise ValueError(
            f"need at least {num_clusters} images for clustering; received {len(images)}"
        )

    breed_labels = np.asarray(breed_labels, dtype=np.int64).reshape(-1)
    if len(breed_labels) != len(images):
        raise ValueError("breed_labels length must match the number of images")
    if not np.any(breed_labels == int(target_breed_id)):
        raise ValueError(f"target breed ID {target_breed_id} is absent from the supplied images")

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    except Exception as exc:
        raise RuntimeError(
            "Could not load CLIP ViT-B/32. Check open_clip_torch and the pretrained model cache/network."
        ) from exc

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    model = model.to(requested_device).eval()

    embeddings = []
    try:
        with torch.inference_mode():
            text = tokenizer([prompt.strip()]).to(requested_device)
            text_embedding = F.normalize(model.encode_text(text), dim=-1)[0]
            for start in range(0, len(images), batch_size):
                batch = images[start:start + batch_size]
                processed = []
                for image in batch:
                    if image.dtype == torch.uint8:
                        pil_image = to_pil_image(image.cpu())
                    else:
                        image = image.detach().cpu().to(torch.float32)
                        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                            raise ValueError("floating-point images must contain finite values in [0, 1]")
                        pil_image = to_pil_image(image)
                    processed.append(preprocess(pil_image))
                image_batch = torch.stack(processed).to(requested_device)
                embeddings.append(F.normalize(model.encode_image(image_batch), dim=-1).cpu())
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"CLIP could not encode the images/prompt: {exc}") from exc

    image_embeddings = torch.cat(embeddings, dim=0).numpy()
    clusterer = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    cluster_ids = clusterer.fit_predict(image_embeddings)

    centroids = clusterer.cluster_centers_
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(centroid_norms, 1e-12)
    similarities = centroids @ text_embedding.detach().cpu().numpy()
    selected_cluster = int(np.argmax(similarities))

    selected = cluster_ids == selected_cluster
    target = breed_labels == int(target_breed_id)
    true_positive = int(np.sum(selected & target))
    selected_count = int(np.sum(selected))
    target_count = int(np.sum(target))
    precision = true_positive / selected_count if selected_count else 0.0
    recall = true_positive / target_count if target_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "prompt": prompt.strip(),
        "selected_cluster": selected_cluster,
        "cluster_size": selected_count,
        "target_breed_id": int(target_breed_id),
        "target_breed_size": target_count,
        "cosine_similarity": float(similarities[selected_cluster]),
        "agreement_precision": precision,
        "agreement_recall": recall,
        "agreement_f1": f1,
        "selected_indices": np.flatnonzero(selected).tolist(),
    }
