"""Select FOUL strength from validation metrics without reading test results."""
import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('../results-2'))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--betas', nargs='+', type=int, default=[1, 2, 4])
    args = parser.parse_args()
    root = args.results_dir
    reference = json.loads((root / f'OxfordPet_FOUL_resnet_examples_reference_s{args.seed}_0_hierarchy.json')
                           .read_text(encoding='utf-8'))
    if (reference['experiment']['forget_scope'] != 'examples' or
            reference['experiment']['seed'] != args.seed or
            reference['experiment']['split_seed'] != 1 or
            reference['experiment']['validation_fraction'] != .2):
        raise ValueError('Reference protocol does not match the beta sweep')
    ref = next(row for row in reference['metrics'] if row['stage'] == 'after_learning'
               and row.get('split') == 'validation')
    candidates = []
    for beta in args.betas:
        goal = f'resnet_examples_magnitude_foul_beta{beta}_s{args.seed}'
        path = root / f'OxfordPet_FOUL_{goal}_0_hierarchy.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        if any(row.get('split') == 'test' for row in data['metrics']):
            raise ValueError(f'Test metrics found in tuning candidate: {path}')
        row = next(row for row in data['metrics'] if row['stage'] == 'after_unlearning'
                   and row.get('split') == 'validation')
        if (data['experiment']['forget_scope'] != 'examples' or
                data['experiment']['pruning_strategy'] != 'magnitude' or
                data['experiment']['forget_weight'] != beta or
                data['experiment']['pruning_rate'] != .25 or
                data['experiment']['unlearning_method'] != 'foul' or
                data['experiment']['seed'] != args.seed or
                data['experiment']['split_seed'] != 1 or
                data['experiment']['validation_fraction'] != .2 or
                data['experiment']['unlearn_rounds'] != 30 or
                data['experiment']['encoder_policy'] != 'full'):
            raise ValueError(f'Protocol mismatch: {path}')
        source_hash = data['experiment']['source_checkpoint_sha256']
        if not source_hash or (candidates and source_hash != candidates[0]['source_checkpoint_sha256']):
            raise ValueError(f'Source checkpoint mismatch: {path}')
        # Lower target accuracy/confidence, preserve utility, approach the
        # behavior of a model trained without target examples.
        score = (5 * abs(row['target_breed_accuracy'] - ref['target_breed_accuracy']) +
                 2 * abs(row['target_mean_confidence'] - ref['target_mean_confidence']) +
                 abs(row['non_target_breed_accuracy'] - ref['non_target_breed_accuracy']) +
                 abs(row['family_accuracy'] - ref['family_accuracy']) +
                 .25 * abs(row['groups']['target']['family_accuracy'] -
                           ref['groups']['target']['family_accuracy']))
        if not math.isfinite(score):
            raise ValueError(f'Non-finite validation score: {path}')
        checkpoint = root / 'models' / 'OxfordPet' / f'FOUL_{goal}_0_pruning_unlearning.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        diagnostics = data.get('update_diagnostics', [])
        def mean_alignment(key):
            values = [item[key] for item in diagnostics if item.get(key) is not None]
            return statistics.mean(values) if values else None
        candidates.append({'beta': beta, 'goal': goal, 'score': score,
                           'target_accuracy': row['target_breed_accuracy'],
                           'target_confidence': row['target_mean_confidence'],
                           'retained_accuracy': row['non_target_breed_accuracy'],
                           'family_accuracy': row['family_accuracy'],
                           'mean_server_vs_forget_cosine': mean_alignment('server_vs_forget_cosine'),
                           'mean_server_vs_retain_cosine': mean_alignment('server_vs_retain_cosine'),
                           'source_checkpoint_sha256': source_hash,
                           'checkpoint': str(checkpoint)})
    selected = min(candidates, key=lambda item: (item['score'], item['beta']))
    output = root / 'comparisons' / f'beta_selection_s{args.seed}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'rule': 'validation only; lower score is better',
                                  'score': '5*target_accuracy_gap + 2*target_confidence_gap + retained_accuracy_gap + family_accuracy_gap + 0.25*target_parent_accuracy_gap versus retraining reference',
                                  'seed': args.seed,
                                  'candidates': candidates, 'selected': selected}, indent=2), encoding='utf-8')
    print(f"Selected beta={selected['beta']} from validation; {output.resolve()}")


if __name__ == '__main__':
    main()
