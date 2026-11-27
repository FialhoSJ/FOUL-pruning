"""Centralized diagnostics; these do not constitute a certified unlearning test."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.pet_protocol import load_pet_split, apply_encoder_policy, dataset_signature


def linear_probe(train_x, train_y, test_x, test_y, seed):
    if len(np.unique(train_y)) != 2 or len(np.unique(test_y)) != 2:
        raise ValueError('Probe needs target and other breeds of the same family in both splits')
    probe = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=2000, random_state=seed))
    probe.fit(train_x, train_y)
    probabilities = probe.predict_proba(test_x)[:, 1]
    return {'balanced_accuracy': float(balanced_accuracy_score(test_y, probabilities >= .5)),
            'auroc': float(roc_auc_score(test_y, probabilities)),
            'train_samples': len(train_y), 'test_samples': len(test_y)}


def load_split(args, config, split):
    parts = [load_pet_split(args.dataset, i, args.dataset_root, split,
                           args.validation_fraction, args.split_seed)
             for i in range(config['num_clients'])]
    return {key: torch.cat([part[key] for part in parts]) for key in ('x', 'y', 'y_subclass')}


def extract(model, images, batch_size, device, encoder=False):
    model.eval()
    with torch.no_grad():
        fn = model.encode if encoder else model.forward_features
        return torch.cat([fn(batch.to(device)).cpu() for batch in images.split(batch_size)]).numpy()


def accuracy(model, data, target, batch_size, device):
    model.eval()
    correct = family_correct = target_correct = target_n = other_correct = other_n = 0
    with torch.no_grad():
        for start in range(0, len(data['x']), batch_size):
            stop = start + batch_size
            family, breed = model(data['x'][start:stop].to(device))
            labels = data['y_subclass'][start:stop].to(device)
            mask = labels == target
            hits = breed.argmax(1) == labels
            correct += int(hits.sum())
            target_correct += int(hits[mask].sum())
            other_correct += int(hits[~mask].sum())
            target_n += int(mask.sum())
            other_n += int((~mask).sum())
            family_correct += int((family.argmax(1).cpu() == data['y'][start:stop]).sum())
    return {'target_accuracy': target_correct / target_n if target_n else None,
            'retained_accuracy': other_correct / other_n if other_n else None,
            'family_accuracy': family_correct / len(data['x'])}


def relearn(model, train, test, target, args):
    """Same labeled examples, order and optimizer for each checkpoint; never save it."""
    model = copy.deepcopy(model)
    apply_encoder_policy(model, 'full')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.relearn_lr)
    rng = np.random.default_rng(args.seed)
    indices = np.concatenate([rng.choice(np.flatnonzero(train['y_subclass'].numpy() == breed),
        size=min(args.relearn_per_breed, int((train['y_subclass'] == breed).sum())), replace=False)
        for breed in torch.unique(train['y_subclass']).tolist()])
    records = [{'step': 0, **accuracy(model, test, target, args.batch_size, args.device)}]
    for step in range(1, args.relearn_steps + 1):
        batch = rng.choice(indices, size=min(args.batch_size, len(indices)), replace=False)
        model.train()
        family, breed = model(train['x'][batch].to(args.device))
        loss = (torch.nn.functional.cross_entropy(family, train['y'][batch].to(args.device)) +
                2 * torch.nn.functional.cross_entropy(breed, train['y_subclass'][batch].to(args.device)))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == args.relearn_steps or step % args.relearn_eval_every == 0:
            records.append({'step': step, **accuracy(model, test, target, args.batch_size, args.device)})
    return records


def benchmark(model, images, args):
    batch = images[:args.batch_size].to(args.device)
    model.eval()
    def sync():
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(3):
            model(batch)
        sync()
        start = time.perf_counter()
        for _ in range(args.inference_repeats):
            model(batch)
        sync()
    return {'raw_image_batch_latency_ms': (time.perf_counter() - start) * 1000 / args.inference_repeats,
            'batch_size': len(batch), 'device': args.device,
            'parameters': sum(p.numel() for p in model.parameters()),
            'tensor_bytes': sum(p.numel() * p.element_size()
                                for p in list(model.parameters()) + list(model.buffers()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+', help='Trusted local model checkpoints, including reference')
    parser.add_argument('--dataset', default='OxfordPet')
    parser.add_argument('--dataset-root', default='../dataset/OxfordPet20')
    parser.add_argument('--target', default='Chihuahua')
    parser.add_argument('--validation-fraction', type=float, default=.2)
    parser.add_argument('--split-seed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--relearn-steps', type=int, default=0)
    parser.add_argument('--relearn-lr', type=float, default=.001)
    parser.add_argument('--relearn-per-breed', type=int, default=5)
    parser.add_argument('--relearn-eval-every', type=int, default=10)
    parser.add_argument('--inference-repeats', type=int, default=20)
    parser.add_argument('--output', default='../results/pet_representation_diagnostics.json')
    args = parser.parse_args()
    if min(args.batch_size, args.relearn_per_breed, args.relearn_eval_every, args.inference_repeats) < 1:
        parser.error('Batch size, sample counts and repeat intervals must be positive')
    torch.manual_seed(args.seed)
    config = json.loads((Path(args.dataset_root) / args.dataset / 'config.json').read_text())
    name = next(name for name in config['breed_to_idx'] if name.casefold() == args.target.casefold())
    target = config['breed_to_idx'][name]
    family = config['parent_by_breed'][name]
    train, test = load_split(args, config, 'train'), load_split(args, config, 'test')
    train_mask, test_mask = train['y'] == family, test['y'] == family
    results = []
    for path in args.checkpoints:
        model = torch.load(path, map_location=args.device, weights_only=False)
        signature = getattr(model, 'dataset_signature', None)
        if signature and signature != dataset_signature(config):
            raise ValueError(f'Label mapping differs: {path}')
        protocol = getattr(model, 'pet_protocol', {})
        if not protocol and args.validation_fraction:
            raise ValueError(f'Legacy checkpoint has no validation holdout; use fraction 0: {path}')
        if protocol and (protocol['pet_validation_fraction'] != args.validation_fraction or
                         protocol['pet_split_seed'] != args.split_seed):
            raise ValueError(f'Validation split differs: {path}')
        row = {'checkpoint': str(Path(path).resolve()),
               'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
               'checkpoint_bytes': Path(path).stat().st_size,
               'inference': benchmark(model, test['x'], args)}
        for level in (['projection', 'encoder'] if hasattr(model, 'encode') else ['projection']):
            row[level + '_probe'] = linear_probe(
                extract(model, train['x'][train_mask], args.batch_size, args.device, level == 'encoder'),
                (train['y_subclass'][train_mask] == target).numpy(),
                extract(model, test['x'][test_mask], args.batch_size, args.device, level == 'encoder'),
                (test['y_subclass'][test_mask] == target).numpy(), args.seed)
        if args.relearn_steps:
            row['centralized_relearning'] = relearn(model, train, test, target, args)
        results.append(row)
        print(f'Completed diagnostics: {path}', flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'protocol': vars(args), 'models': results}, indent=2), encoding='utf-8')
    print(output.resolve())


if __name__ == '__main__':
    main()
