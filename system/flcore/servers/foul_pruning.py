"""Federated class-discriminative channel scoring and compact CNN rebuilding."""

import numpy as np
import torch
import torch.nn as nn
import copy


def aggregate_activation_summaries(client_summaries):
    """Combine client per-breed activation sums and counts without raw images."""
    totals = {}
    for summary in client_summaries:
        for layer_name, layer_data in summary.items():
            layer_total = totals.setdefault(layer_name, {"sums": {}, "counts": {}})
            for breed_id, channel_sum in layer_data.get("sums", {}).items():
                breed_id = int(breed_id)
                channel_sum = torch.as_tensor(channel_sum, dtype=torch.float64).cpu()
                if channel_sum.ndim != 1 or not torch.isfinite(channel_sum).all():
                    raise ValueError(f"invalid activation summary for {layer_name}, breed {breed_id}")
                previous = layer_total["sums"].get(breed_id)
                if previous is not None and previous.shape != channel_sum.shape:
                    raise ValueError(f"client activation widths disagree for layer {layer_name}")
                layer_total["sums"][breed_id] = channel_sum if previous is None else previous + channel_sum
            for breed_id, count in layer_data.get("counts", {}).items():
                breed_id = int(breed_id)
                count = int(count)
                if count < 0:
                    raise ValueError(f"negative activation count for {layer_name}, breed {breed_id}")
                layer_total["counts"][breed_id] = layer_total["counts"].get(breed_id, 0) + count

    for layer_name, layer_data in totals.items():
        for breed_id, channel_sum in layer_data["sums"].items():
            count = layer_data["counts"].get(breed_id, 0)
            if count < 1:
                raise ValueError(f"missing activation count for {layer_name}, breed {breed_id}")
            layer_data["sums"][breed_id] = channel_sum / count
    return totals


def tfidf_channel_scores(activation_summary, target_breed_id):
    """Compute FCUP's class-discriminative TF-IDF score for each feature layer."""
    target_breed_id = int(target_breed_id)
    scores = {}
    for layer_name, layer_data in activation_summary.items():
        class_means = layer_data.get("sums", {})
        if target_breed_id not in class_means:
            raise ValueError(f"target breed {target_breed_id} has no activation summary for {layer_name}")

        breed_ids = sorted(class_means)
        matrix = torch.stack([
            torch.as_tensor(class_means[breed_id], dtype=torch.float64).cpu()
            for breed_id in breed_ids
        ]).numpy()
        if matrix.ndim != 2 or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
            raise ValueError(f"invalid class activation matrix for {layer_name}")
        if np.any(matrix < 0):
            raise ValueError(f"ReLU activation summaries for {layer_name} must be non-negative")

        target_row = matrix[breed_ids.index(target_breed_id)]
        term_frequency = target_row / max(float(target_row.sum()), 1e-12)
        document_frequency = np.sum(matrix >= matrix.mean(axis=1, keepdims=True), axis=0)
        inverse_document_frequency = np.log1p(len(breed_ids) / (1.0 + document_frequency))
        scores[layer_name] = term_frequency * inverse_document_frequency
    return scores


def _selected_channels(scores, width, pruning_rate):
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(score_values) != width:
        raise ValueError(f"activation score width {len(score_values)} does not match layer width {width}")
    if not np.isfinite(score_values).all():
        raise ValueError("activation scores must be finite")
    remove_count = min(width - 1, int(round(pruning_rate * width)))
    if remove_count <= 0:
        return torch.arange(width, dtype=torch.long)
    indices = np.arange(width)
    ranked = np.lexsort((indices, -score_values))
    removed = set(ranked[:remove_count].tolist())
    kept = [index for index in range(width) if index not in removed]
    return torch.tensor(kept, dtype=torch.long)


def _copy_conv(old_layer, in_indices, out_indices):
    in_indices = in_indices.to(old_layer.weight.device)
    out_indices = out_indices.to(old_layer.weight.device)
    new_layer = nn.Conv2d(
        in_channels=len(in_indices),
        out_channels=len(out_indices),
        kernel_size=old_layer.kernel_size,
        stride=old_layer.stride,
        padding=old_layer.padding,
        dilation=old_layer.dilation,
        groups=1,
        bias=old_layer.bias is not None,
        padding_mode=old_layer.padding_mode,
    )
    weight = old_layer.weight.detach().index_select(0, out_indices).index_select(1, in_indices)
    new_layer.weight = nn.Parameter(weight.clone())
    if old_layer.bias is not None:
        new_layer.bias = nn.Parameter(old_layer.bias.detach().index_select(0, out_indices).clone())
    return new_layer


def _copy_linear(old_layer, in_indices, out_indices=None):
    if out_indices is None:
        out_indices = torch.arange(old_layer.out_features)
    in_indices = in_indices.to(old_layer.weight.device)
    out_indices = out_indices.to(old_layer.weight.device)
    new_layer = nn.Linear(
        in_features=len(in_indices),
        out_features=len(out_indices),
        bias=old_layer.bias is not None,
    )
    weight = old_layer.weight.detach().index_select(0, out_indices).index_select(1, in_indices)
    new_layer.weight = nn.Parameter(weight.clone())
    if old_layer.bias is not None:
        new_layer.bias = nn.Parameter(old_layer.bias.detach().index_select(0, out_indices).clone())
    return new_layer


def prune_model(model, class_activation_summaries, target_breed_id, pruning_rate=0.5,
                strategy="discriminative", seed=1):
    """Physically shrink eligible target-discriminative channels."""
    if not 0 < pruning_rate < 1:
        raise ValueError("pruning_rate must be between 0 and 1")
    if strategy == "random":
        rng = np.random.default_rng(seed)
        score_map = {
            name: rng.random(layer.out_channels if isinstance(layer, nn.Conv2d) else layer.out_features)
            for name, layer in model.feature_layers()
        }
    elif strategy == "magnitude":
        # Existing selector removes the highest scores; negate the channel norm
        # so compression removes low-magnitude channels, independent of breed.
        score_map = {
            name: -layer.weight.detach().abs().flatten(1).mean(1).cpu().numpy()
            for name, layer in model.feature_layers()
        }
    elif strategy == "discriminative":
        score_map = tfidf_channel_scores(class_activation_summaries, target_breed_id)
    else:
        raise ValueError(f"Unknown pruning strategy: {strategy}")
    if hasattr(model, "backbone") and hasattr(model, "fc1"):
        return _prune_resnet_projection(
            model, score_map, pruning_rate
        )
    required_layers = ("conv1", "conv2", "conv3", "fc1")
    model_layers = dict(model.feature_layers())
    if any(name not in model_layers for name in required_layers):
        raise TypeError("model does not expose the supported hierarchical CNN feature layers")

    keep = {
        name: _selected_channels(score_map[name], layer.out_channels if isinstance(layer, nn.Conv2d)
                                 else layer.out_features, pruning_rate)
        for name, layer in model.feature_layers()
    }
    grid = model.global_pool.output_size
    if isinstance(grid, int):
        grid = (grid, grid)
    spatial_size = int(grid[0]) * int(grid[1])
    if model.fc1.in_features != model.conv3.out_channels * spatial_size:
        raise ValueError("fc1 input width does not match pooled conv3 channels")
    pooled_indices = torch.cat([
        torch.arange(int(channel) * spatial_size, (int(channel) + 1) * spatial_size)
        for channel in keep["conv3"]
    ])
    compact = model.__class__(**model.compact_config()).to(next(model.parameters()).device)
    compact.conv1 = _copy_conv(model.conv1, torch.arange(3), keep["conv1"])
    compact.conv2 = _copy_conv(model.conv2, keep["conv1"], keep["conv2"])
    compact.conv3 = _copy_conv(model.conv3, keep["conv2"], keep["conv3"])
    compact.fc1 = _copy_linear(model.fc1, pooled_indices, keep["fc1"])
    compact.family_head = _copy_linear(
        model.family_head, keep["fc1"], torch.arange(model.family_head.out_features)
    )
    compact.breed_head = _copy_linear(
        model.breed_head, keep["fc1"], torch.arange(model.breed_head.out_features)
    )
    compact.channels = (len(keep["conv1"]), len(keep["conv2"]), len(keep["conv3"]))
    compact.hidden_dim = len(keep["fc1"])
    compact.train(model.training)

    counts = {}
    for name, layer in model.feature_layers():
        before = layer.out_channels if isinstance(layer, nn.Conv2d) else layer.out_features
        counts[name] = {"before": before, "kept": len(keep[name]), "removed": before - len(keep[name])}

    if compact.family_head.out_features != model.family_head.out_features:
        raise ValueError("compact model changed the family output dimension")
    if compact.breed_head.out_features != model.breed_head.out_features:
        raise ValueError("compact model changed the breed output dimension")
    if compact.conv2.in_channels != compact.conv1.out_channels:
        raise ValueError("compact conv1/conv2 channel dimensions do not match")
    if compact.conv3.in_channels != compact.conv2.out_channels:
        raise ValueError("compact conv2/conv3 channel dimensions do not match")
    if compact.fc1.in_features != compact.conv3.out_channels * spatial_size:
        raise ValueError("compact conv3/fc1 dimensions do not match")
    if compact.family_head.in_features != compact.fc1.out_features:
        raise ValueError("compact family head input width does not match fc1")
    if compact.breed_head.in_features != compact.fc1.out_features:
        raise ValueError("compact breed head input width does not match fc1")
    for name, layer in compact.feature_layers():
        width = layer.out_channels if isinstance(layer, nn.Conv2d) else layer.out_features
        if width < 1:
            raise ValueError(f"pruning removed every feature from {name}")
    return compact, counts


def _prune_resnet_projection(model, score_map, pruning_rate):
    """Prune the trainable projection, leaving residual connections intact."""
    if tuple(name for name, _ in model.feature_layers()) != ("fc1",):
        raise TypeError("Unsupported pretrained backbone feature layers")
    scores = score_map["fc1"]
    keep = _selected_channels(scores, model.fc1.out_features, pruning_rate)
    compact = copy.deepcopy(model)
    compact.fc1 = _copy_linear(model.fc1, torch.arange(model.fc1.in_features), keep)
    compact.family_head = _copy_linear(
        model.family_head, keep, torch.arange(model.family_head.out_features)
    )
    compact.breed_head = _copy_linear(
        model.breed_head, keep, torch.arange(model.breed_head.out_features)
    )
    compact.hidden_dim = len(keep)
    compact.train(model.training)
    if compact.fc1.out_features != compact.family_head.in_features or \
            compact.fc1.out_features != compact.breed_head.in_features:
        raise ValueError("Compact projection and output heads have incompatible shapes")
    return compact, {"fc1": {
        "before": model.fc1.out_features,
        "kept": len(keep),
        "removed": model.fc1.out_features - len(keep),
    }}
