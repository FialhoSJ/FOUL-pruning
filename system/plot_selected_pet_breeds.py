"""Plot per-breed test accuracy for the selected FOUL strength and controls."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def test_metric(root, goal):
    path = root / f'OxfordPet_FOUL_{goal}_0_hierarchy.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    return next(row for row in data['metrics'] if row.get('split') == 'test'
                and row['stage'] == 'after_learning_test')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('../results-2'))
    parser.add_argument('--dataset-root', type=Path, default=Path('../dataset/OxfordPet20'))
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    root = args.results_dir
    selection = json.loads((root / 'comparisons' / f'beta_selection_s{args.seed}.json')
                           .read_text(encoding='utf-8'))
    selected = json.loads((root / 'comparisons' / f'beta_selected_test_s{args.seed}.json')
                          .read_text(encoding='utf-8'))
    if Path(selected['checkpoint']).name != Path(selection['selected']['checkpoint']).name:
        raise ValueError('The selected test belongs to another checkpoint')
    config = json.loads((args.dataset_root / 'OxfordPet' / 'config.json').read_text(encoding='utf-8'))
    names = [name for name, _ in sorted(config['breed_to_idx'].items(), key=lambda item: int(item[1]))]
    target = int(config['breed_to_idx']['Chihuahua'])
    labels = ['Original', 'Sem Chihuahua no treino', f"Poda + FOUL beta={selection['selected']['beta']}"]
    metrics = [test_metric(root, f'resnet_train_s{args.seed}'),
               test_metric(root, f'resnet_examples_reference_s{args.seed}'),
               selected['metrics']]
    matrices = [np.asarray(row['breed_confusion_matrix'], dtype=int) for row in metrics]
    if any(matrix.shape != (len(names), len(names)) for matrix in matrices):
        raise ValueError('Unexpected breed confusion matrix dimensions')
    accuracy = []
    for matrix in matrices:
        support = matrix.sum(axis=1)
        accuracy.append(np.divide(matrix.diagonal(), support,
                                  out=np.full(len(names), np.nan), where=support > 0) * 100)
    charts = [
        (np.asarray(accuracy), 'Acuracia por raca no teste (%)', 'viridis', 100,
         f'beta_selected_breeds_s{args.seed}'),
        (np.asarray([matrix[target] for matrix in matrices]),
         'Previsoes para imagens de Chihuahua no teste', 'Blues',
         max(1, max(int(matrix[target].max()) for matrix in matrices)),
         f'beta_selected_chihuahua_predictions_s{args.seed}'),
    ]
    for values, title, palette, maximum, filename in charts:
        fig, ax = plt.subplots(figsize=(20, 4.8), layout='constrained')
        graphic = ax.imshow(values, cmap=palette, vmin=0, vmax=maximum, aspect='auto')
        ax.set_xticks(range(len(names)), names, rotation=65, ha='right', fontsize=8)
        ax.set_yticks(range(len(labels)), labels)
        ax.add_patch(Rectangle((target - .5, -.5), 1, len(labels), fill=False,
                               edgecolor='red', linewidth=2.5))
        ax.set_title(title + ' - Chihuahua em vermelho')
        fig.colorbar(graphic, ax=ax, shrink=.8)
        output = root / 'comparisons' / f'{filename}.png'
        fig.savefig(output, dpi=200)
        plt.close(fig)
        print(output.resolve())


if __name__ == '__main__':
    main()
