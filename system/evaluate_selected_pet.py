"""Evaluate one selected pet checkpoint on the official test, once."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from flcore.servers.serverfoul import FOUL
from utils.pet_protocol import dataset_signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--dataset-root', default='../dataset/OxfordPet20')
    parser.add_argument('--target', default='Chihuahua')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((Path(args.dataset_root) / 'OxfordPet' / 'config.json').read_text(encoding='utf-8'))
    model = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if getattr(model, 'dataset_signature', None) != dataset_signature(config):
        raise ValueError('Checkpoint dataset signature is missing or differs from OxfordPet')
    if model.num_families != len(config['family_to_idx']) or model.num_breeds != len(config['breed_to_idx']):
        raise ValueError('Checkpoint output dimensions differ from OxfordPet')
    if hasattr(model, 'backbone') and model.encoder_frozen:
        # Raw-image evaluation avoids depending on client feature caches.
        model.unfreeze_encoder()
    server = object.__new__(FOUL)
    server.args = SimpleNamespace(dataset_root=args.dataset_root, forget_breed=args.target,
                                  pet_validation_fraction=.2, pet_split_seed=1, breed_loss_weight=2.)
    server.dataset = 'OxfordPet'
    server.device = args.device
    server.batch_size = args.batch_size
    server.global_model = model
    server.clients = [SimpleNamespace(id=index) for index in range(int(config['num_clients']))]
    server._target_breed_cache = None
    server.rs_test_acc, server.rs_train_loss, server.hierarchical_metrics = [], [], []
    server.current_round = 0
    metrics = server._evaluate_hierarchical('selected_test', split='test')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'checkpoint': str(args.checkpoint.resolve()), 'metrics': metrics}, indent=2),
                           encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
