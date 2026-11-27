"""Compare pruning treatments on validation only; never select using test rows."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('../results-2'))
    parser.add_argument('--dataset-root', type=Path, default=Path('../dataset/OxfordPet20'))
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    root, seed = args.results_dir, args.seed
    out = root / 'comparisons' / f'pruning_sweep_s{seed}'
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.dataset_root / 'OxfordPet/config.json').read_text(encoding='utf-8'))
    names = [n for n, i in sorted(config['breed_to_idx'].items(), key=lambda x: int(x[1]))]
    target = int(config['breed_to_idx']['Chihuahua'])

    def read(goal, stage):
        payload = json.loads((root / f'OxfordPet_FOUL_{goal}_0_hierarchy.json').read_text(encoding='utf-8'))
        matches = [m for m in payload['metrics'] if m.get('split') == 'validation' and m['stage'] == stage]
        if len(matches) != 1:
            raise ValueError(f'Expected one validation/{stage} row: {goal}')
        return payload, matches[0]

    baseline, base = read(f'resnet_train_s{seed}', 'after_learning')
    _, reference = read(f'resnet_examples_reference_s{seed}', 'after_learning')
    cases = [('Original', 0, 'baseline', baseline, base),
             ('Referencia sem Chihuahua', 0, 'reference', {}, reference)]
    labels = {'only': 'Poda isolada', 'foul': 'FOUL', 'retain': 'Retencao'}
    hashes = set()
    for rate in (0, 5, 10, 15, 25):
        for method in ('only', 'foul', 'retain'):
            if rate == 0 and method == 'only':
                continue
            goal = (f'resnet_examples_noprune_{method}_s{seed}' if rate == 0 else
                    f'resnet_examples_magnitude_{method}_s{seed}' if rate == 25 else
                    f'resnet_examples_mag_p{rate:02d}_{method}_s{seed}')
            payload, metric = read(goal, 'after_pruning' if method == 'only' else 'after_unlearning')
            exp = payload['experiment']
            if (exp['forget_scope'] != 'examples' or exp['validation_fraction'] != .2 or
                    exp['split_seed'] != 1 or exp['seed'] != seed or exp['pruning_rate'] != rate / 100):
                raise ValueError(f'Incompatible protocol: {goal}')
            hashes.add(exp['source_checkpoint_sha256'])
            cases.append((f'{labels[method]} | poda {rate}%', rate, method, payload, metric))
    if len(hashes) != 1 or None in hashes:
        raise ValueError('Treatments use different source checkpoints')

    rows = []
    base_params = baseline['model_costs'][-1]['total_parameters']
    for label, rate, method, payload, m in cases:
        costs = payload.get('model_costs', [])
        params = costs[-1]['total_parameters'] if costs else base_params
        rows.append(dict(method=label, pruning_rate=rate, split='validation',
                         target_accuracy=m['target_breed_accuracy'], target_confidence=m['target_mean_confidence'],
                         retained_accuracy=m['non_target_breed_accuracy'], family_accuracy=m['family_accuracy'],
                         other_dogs_accuracy=m['groups']['same_family_other_breeds']['breed_accuracy'],
                         cats_accuracy=m['groups']['other_family']['breed_accuracy'],
                         target_parent_accuracy=m['groups']['target']['family_accuracy'],
                         target_samples=m['target_breed_samples'], parameters=params,
                         parameter_reduction_pct=100 * (1 - params / base_params),
                         treatment_minutes=payload.get('times_seconds', {}).get('total_post_training', 0) / 60))
    with (out / 'metrics_validation.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    def save(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'{name}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, key, title in zip(axes.flat,
            ['target_breed_accuracy', 'non_target_breed_accuracy', 'family_accuracy', 'target_mean_confidence'],
            ['Acerto em Chihuahua', 'Acerto nas demais racas', 'Acerto cachorro/gato', 'Probabilidade media de Chihuahua nas imagens alvo']):
        for method, color in [('only', '#777777'), ('foul', '#276bb2'), ('retain', '#198754')]:
            selected = [c for c in cases if c[2] == method]
            ax.plot([c[1] for c in selected], [100*c[4][key] for c in selected], 'o-', label=labels[method], color=color)
        ax.axhline(100*base[key], color='#e19418', ls='--', label='Original')
        ax.axhline(100*reference[key], color='#aa378b', ls=':', label='Referencia')
        ax.set(title=title, xlabel='Poda da projecao (%)', ylabel='Validacao (%)', xticks=[0, 5, 10, 15, 25])
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Comparacao na validacao | seed 1 | Chihuahua: 20 imagens; demais racas: 720')
    save(fig, 'comparison_validation')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for rate, color in zip((0, 5, 10, 15, 25), plt.cm.viridis(np.linspace(.1,.9,5))):
        c = next(c for c in cases if c[1] == rate and c[2] == 'foul')
        ms = [m for m in c[3]['metrics'] if m.get('split') == 'validation' and m['stage'] in ('unlearning', 'after_unlearning')]
        for ax, key in zip(axes, ('target_breed_accuracy', 'non_target_breed_accuracy')):
            ax.plot([m['round'] for m in ms], [m[key]*100 for m in ms], label=f'Poda {rate}%', color=color)
    for ax, title in zip(axes, ('Chihuahua', 'Demais racas')):
        ax.set(title=title, xlabel='Rodada de FOUL', ylabel='Acuracia na validacao (%)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    save(fig, 'foul_trajectories_validation')

    matrices = [np.asarray(c[4]['breed_confusion_matrix']) for c in cases]
    accuracies = [np.divide(m.diagonal(), m.sum(1), out=np.zeros(len(names), dtype=float), where=m.sum(1)>0)*100 for m in matrices]
    for values, title, cmap, vmax, filename in [
        (accuracies, 'Acuracia por raca na validacao (%)', 'viridis', 100, 'breeds_validation'),
        ([m[target] for m in matrices], 'Previsoes para as 20 imagens de Chihuahua da validacao', 'Blues', 20, 'chihuahua_predictions_validation')]:
        fig, ax = plt.subplots(figsize=(20, 9), layout='constrained')
        im=ax.imshow(values, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(names)), names, rotation=65, ha='right', fontsize=8)
        ax.set_yticks(range(len(cases)), [c[0] for c in cases], fontsize=9)
        ax.axvline(target-.5, color='red', lw=1.5); ax.axvline(target+.5, color='red', lw=1.5)
        ax.set_title(title + ' | coluna Chihuahua em vermelho'); fig.colorbar(im, ax=ax, shrink=.7)
        save(fig, filename)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    selected=[r for r in rows if r['method'].startswith('Poda isolada')]
    bars=axes[0].bar([str(r['pruning_rate'])+'%' for r in selected], [r['parameter_reduction_pct'] for r in selected])
    axes[0].bar_label(bars, fmt='%.3f%%'); axes[0].set(title='Reducao de parametros do modelo inteiro', xlabel='Poda da projecao', ylabel='Reducao (%)')
    for method in ('foul', 'retain'):
        chosen=[r for r,c in zip(rows,cases) if c[2]==method]
        axes[1].plot([r['pruning_rate'] for r in chosen], [r['treatment_minutes'] for r in chosen], 'o-', label=labels[method])
    axes[1].set(title='Tempo observado (inclui avaliacao; execucoes sequenciais)', xlabel='Poda da projecao (%)', ylabel='Minutos'); axes[1].legend()
    save(fig, 'costs')
    print(out.resolve())
    for r in rows:
        print(r['method'], f"target={r['target_accuracy']:.1%} retain={r['retained_accuracy']:.2%} family={r['family_accuracy']:.2%} confidence={r['target_confidence']:.6%} reduction={r['parameter_reduction_pct']:.3f}%")


if __name__ == '__main__':
    main()
