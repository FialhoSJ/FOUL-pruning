"""Compare complete-example ablations on identical validation/test splits."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def read_row(root, goal, stage, split):
    path = root / f'OxfordPet_FOUL_{goal}_0_hierarchy.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if goal.startswith('resnet_examples') and data['experiment'].get('forget_scope') != 'examples':
        raise ValueError(f'Wrong forget scope: {path}')
    matches = [row for row in data['metrics'] if row['stage'] == stage and row.get('split') == split]
    if len(matches) != 1:
        raise ValueError(f'Expected one {stage}/{split} row in {path}')
    row = matches[0]
    costs = data.get('model_costs', [])
    cost = next((item for item in reversed(costs) if item['stage'] in ('after_unlearning', 'after_pruning', 'after_learning')), None)
    return {
        'goal': goal, 'split': split,
        'target_breed_accuracy': row['target_breed_accuracy'],
        'target_family_accuracy': row['groups']['target']['family_accuracy'],
        'other_dog_breed_accuracy': row['groups']['same_family_other_breeds']['breed_accuracy'],
        'cat_breed_accuracy': row['groups']['other_family']['breed_accuracy'],
        'retained_breed_accuracy': row['non_target_breed_accuracy'],
        'family_accuracy': row['family_accuracy'],
        'target_confidence': row.get('target_mean_confidence'),
        'retained_breed_loss': row.get('non_target_breed_loss'),
        'parameters': cost['total_parameters'] if cost else None,
        'treatment_seconds': data.get('times_seconds', {}).get('total_post_training'),
    }


def plot_breed_details(root, dataset_root, cases, output, seed):
    config = json.loads((dataset_root / 'OxfordPet' / 'config.json').read_text(encoding='utf-8'))
    names = [name for name, _ in sorted(config['breed_to_idx'].items(), key=lambda item: int(item[1]))]
    target = int(config['breed_to_idx']['Chihuahua'])
    accuracy, predictions, breed_rows, prediction_rows = [], [], [], []
    for label, goal, stage in cases:
        data = json.loads((root / f'OxfordPet_FOUL_{goal}_0_hierarchy.json').read_text(encoding='utf-8'))
        metric = next(row for row in data['metrics'] if row['stage'] == stage + '_test'
                      and row.get('split') == 'test')
        confusion = np.asarray(metric['breed_confusion_matrix'], dtype=int)
        if confusion.shape != (len(names), len(names)):
            raise ValueError(f'Invalid breed confusion matrix for {goal}')
        support = confusion.sum(axis=1)
        per_breed = np.divide(confusion.diagonal(), support, out=np.full(len(names), np.nan), where=support > 0)
        accuracy.append(per_breed)
        predictions.append(confusion[target])
        for index, name in enumerate(names):
            breed_rows.append({'method': label, 'breed': name, 'test_samples': int(support[index]),
                               'correct': int(confusion[index, index]), 'accuracy': float(per_breed[index])})
            prediction_rows.append({'method': label, 'predicted_breed': name,
                                    'chihuahua_images': int(confusion[target, index])})
    for filename, rows in [(f'breed_accuracy_s{seed}.csv', breed_rows),
                           (f'chihuahua_predictions_s{seed}.csv', prediction_rows)]:
        with (output / filename).open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    settings = [
        (np.asarray(accuracy) * 100, 'Acuracia por raca no teste (%)', 'viridis', 0, 100,
         f'breed_accuracy_s{seed}'),
        (np.asarray(predictions), 'Previsoes para imagens de Chihuahua (contagem)',
         'Blues', 0, max(1, int(np.asarray(predictions).max())),
         f'chihuahua_predictions_s{seed}'),
    ]
    for matrix, title, palette, minimum, maximum, filename in settings:
        fig, ax = plt.subplots(figsize=(21, 5.8), layout='constrained')
        graphic = ax.imshow(matrix, cmap=palette, vmin=minimum, vmax=maximum, aspect='auto')
        ax.set_xticks(range(len(names)), names, rotation=65, ha='right', fontsize=8)
        ax.set_yticks(range(len(cases)), [case[0] for case in cases])
        ax.add_patch(Rectangle((target - .5, -.5), 1, len(cases), fill=False,
                               edgecolor='red', linewidth=2.5))
        ax.set_title(title + ' - Chihuahua destacado em vermelho')
        fig.colorbar(graphic, ax=ax, shrink=.82)
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'{filename}.{suffix}', dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('../results-2'))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset-root', type=Path, default=Path('../dataset/OxfordPet20'))
    args = parser.parse_args()
    cases = [
        ('Original', f'resnet_train_s{args.seed}', 'after_learning'),
        ('Referencia', f'resnet_examples_reference_s{args.seed}', 'after_learning'),
        ('FOUL', f'resnet_examples_noprune_foul_s{args.seed}', 'after_unlearning'),
        ('Retencao', f'resnet_examples_noprune_retain_s{args.seed}', 'after_unlearning'),
        ('Poda apenas', f'resnet_examples_magnitude_only_s{args.seed}', 'after_pruning'),
        ('Poda + retencao', f'resnet_examples_magnitude_retain_s{args.seed}', 'after_unlearning'),
        ('Poda + FOUL', f'resnet_examples_magnitude_foul_s{args.seed}', 'after_unlearning'),
    ]
    rows = []
    for label, goal, stage in cases:
        for split in ('validation', 'test'):
            suffix = '_test' if split == 'test' else ''
            row = read_row(args.results_dir, goal, stage + suffix, split)
            rows.append({'method': label, **row})
    output = args.results_dir / 'comparisons'
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f'example_removal_s{args.seed}.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # The figure deliberately displays the final test only; validation is in CSV
    # for selecting a method without making the official test a tuning target.
    test = [row for row in rows if row['split'] == 'test']
    labels = [row['method'] for row in test]
    x = np.arange(len(test))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout='constrained')
    for ax, series, title in [
        (axes[0], [('target_breed_accuracy', 'Chihuahua: raca'),
                   ('target_family_accuracy', 'Chihuahua: cachorro')], 'Respostas para Chihuahua'),
        (axes[1], [('retained_breed_accuracy', 'Demais racas'),
                   ('family_accuracy', 'Cachorro/gato')], 'Tarefas preservadas'),
    ]:
        width = .36
        for index, (key, caption) in enumerate(series):
            values = [row[key] * 100 for row in test]
            bars = ax.bar(x + (index - .5) * width, values, width, label=caption)
            ax.bar_label(bars, fmt='%.1f', padding=2, fontsize=8)
        ax.set_xticks(x, labels, rotation=25, ha='right')
        ax.set_ylim(0, 110)
        ax.set_ylabel('Acuracia no teste (%)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False)
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'example_removal_s{args.seed}.{suffix}', dpi=220)
    plt.close(fig)
    plot_breed_details(args.results_dir, args.dataset_root, cases, output, args.seed)
    print(f'Comparison written to {output.resolve()}')


if __name__ == '__main__':
    main()
