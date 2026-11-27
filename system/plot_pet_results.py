"""Create publication-ready plots from a hierarchical pet experiment JSON."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    metrics = payload.get("metrics", [])
    if not metrics:
        raise ValueError(f"No metrics found in {path}")
    return payload, metrics


def plot_learning_curves(metrics, output_dir, stem):
    by_round = {}
    for row in metrics:
        stage = row.get("stage")
        if stage not in ("evaluation", "after_learning"):
            continue
        round_id = int(row.get("round", 0))
        if round_id not in by_round or stage == "after_learning":
            by_round[round_id] = row
    rows = [by_round[key] for key in sorted(by_round)]
    if not rows:
        return
    rounds = [int(row.get("round", 0)) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    ax = axes[0]
    series = (
        ("target_breed_accuracy", "Raça alvo"),
        ("non_target_breed_accuracy", "Demais raças"),
        ("family_accuracy", "Cachorro/gato"),
        ("overall_breed_accuracy", "Raças: geral"),
    )
    for key, label in series:
        values = [row.get(key) for row in rows]
        if any(value is not None for value in values):
            ax.plot(rounds, values, linewidth=2, label=label)
    ax.set(title="Acurácia por rodada (split monitorado)", xlabel="Rodada", ylabel="Acurácia")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    for key, label in (("family_test_loss", "Perda: família"),
                       ("breed_test_loss", "Perda: raça")):
        values = [row.get(key) for row in rows]
        if any(value is not None for value in values):
            ax.plot(rounds, values, linewidth=2, label=label)
    ax.set(title="Perda por rodada (split monitorado)", xlabel="Rodada", ylabel="Cross-entropy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle("Treinamento federado hierárquico — Oxford-IIIT Pet", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}_curvas.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_final_groups(metrics, output_dir, stem):
    final = metrics[-1]
    groups = final.get("groups", {})
    ordered = (
        ("target", "Raça alvo"),
        ("same_family_other_breeds", "Outras raças da mesma família"),
        ("other_family", "Outra família"),
    )
    names = [label for key, label in ordered if key in groups]
    breed = [groups[key].get("breed_accuracy") for key, _ in ordered if key in groups]
    family = [groups[key].get("family_accuracy") for key, _ in ordered if key in groups]
    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    b1 = ax.bar(x - width / 2, breed, width, label="Acurácia da raça", color="#2878B5")
    valid_family = [value if value is not None else 0 for value in family]
    b2 = ax.bar(x + width / 2, valid_family, width, label="Acurácia cachorro/gato", color="#E07A3F")
    for bar, value in zip(list(b1) + list(b2), breed + valid_family):
        if value is not None:
            ax.annotate(f"{value:.1%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Acurácia")
    ax.set_title(f"Desempenho final por grupo — alvo: {final.get('target_breed', 'subclasse')}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}_grupos_finais.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_treatment_curves(metrics, output_dir, stem, baseline_metrics=None, pruning_rate=0):
    unlearning = [row for row in metrics if row.get("stage") == "unlearning"
                  and int(row.get("round", 0)) > 0]
    before = next((row for row in metrics if row.get("stage") == "before_pruning"), None)
    after_pruning = next((row for row in metrics if row.get("stage") == "after_pruning"), None)
    final = next((row for row in reversed(metrics) if row.get("stage") == "after_unlearning"), None)
    if before is None or after_pruning is None or final is None:
        return

    rows = []
    labels = []
    if baseline_metrics:
        trained = next((row for row in reversed(baseline_metrics)
                        if row.get("stage") == "after_learning"), baseline_metrics[-1])
        rows.append(trained)
        labels.append("Treino final")
    rows.extend((before, after_pruning))
    labels.extend(("Antes do\npruning", "Após o\npruning"))
    rows.extend(unlearning)
    labels.extend([f"{int(row['round'])}" for row in unlearning])
    if not unlearning or int(unlearning[-1].get("round", -1)) != int(final.get("round", -2)):
        rows.append(final)
        labels.append(f"{int(final.get('round', 0))}")
    else:
        rows[-1] = final

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x = np.arange(len(rows))
    series = (("target_breed_accuracy", "Raça alvo"),
              ("non_target_breed_accuracy", "Demais raças"),
              ("family_accuracy", "Família: geral"))
    for key, label in series:
        axes[0].plot(x, [row.get(key) for row in rows], marker="o", markersize=3,
                     linewidth=2, label=label)
    axes[0].set(title="Acurácia durante pruning e unlearning",
                xlabel="Fase e rodadas de unlearning", ylabel="Acurácia", ylim=(0, 1.03))
    tick_positions = sorted(set([0, len(rows) - 1] + list(range(0, len(rows), max(1, len(rows) // 8)))))
    tick_labels = [labels[i] for i in tick_positions]
    axes[0].set_xticks(tick_positions, tick_labels)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")

    for key, label in (("family_test_loss", "Perda: família"),
                       ("breed_test_loss", "Perda: raça")):
        axes[1].plot(x, [row.get(key) for row in rows], marker="o", markersize=3,
                     linewidth=2, label=label)
    axes[1].set(title="Perda durante o tratamento (split monitorado)",
                xlabel="Fase e rodadas de unlearning", ylabel="Cross-entropy")
    axes[1].set_xticks(tick_positions, tick_labels)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    title = "Pruning e recuperação" if pruning_rate else "Unlearning sem pruning"
    fig.suptitle(f"{title}: trajetória no split monitorado", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}_trajetoria_unlearning.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pruning_costs(payload, output_dir, stem):
    costs = payload.get("model_costs", [])
    before = next((row for row in costs if row.get("stage") == "before_pruning"), None)
    after = next((row for row in costs if row.get("stage") == "after_pruning"), None)
    if before is None or after is None:
        return
    labels = ["Antes do pruning", "Após pruning"]
    total = [before["total_parameters"] / 1e6, after["total_parameters"] / 1e6]
    trainable = [before["trainable_parameters"] / 1e6, after["trainable_parameters"] / 1e6]
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    x = np.arange(2)
    width = 0.34
    bars_total = ax.bar(x - width / 2, total, width, label="Parâmetros totais", color="#2878B5")
    bars_train = ax.bar(x + width / 2, trainable, width, label="Parâmetros treináveis", color="#E07A3F")
    for bars in (bars_total, bars_train):
        ax.bar_label(bars, fmt="%.3f M", padding=3, fontsize=9)
    total_drop = 100 * (1 - after["total_parameters"] / before["total_parameters"])
    train_drop = 100 * (1 - after["trainable_parameters"] / before["trainable_parameters"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Número de parâmetros (milhões)")
    ax.set_title(f"Redução pelo pruning — total: {total_drop:.2f}%; treináveis: {train_drop:.1f}%")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}_custo_pruning.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None,
                        help="Optional training-only JSON to include its final metrics")
    args = parser.parse_args()
    payload, metrics = load_metrics(args.results_json)
    baseline_metrics = load_metrics(args.baseline_json)[1] if args.baseline_json else None
    output_dir = args.output_dir or args.results_json.parent / "graficos"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.results_json.stem.replace("_hierarchy", "")
    plot_learning_curves(metrics, output_dir, stem)
    plot_final_groups(metrics, output_dir, stem)
    pruning_rate = float(payload.get("experiment", {}).get("pruning_rate", 0))
    plot_treatment_curves(metrics, output_dir, stem, baseline_metrics, pruning_rate)
    if pruning_rate:
        plot_pruning_costs(payload, output_dir, stem)
    print(f"Plots saved under {output_dir}")
    print(f"Target breed: {metrics[-1].get('target_breed')}; "
          f"pruning metadata: {payload.get('pruning_counts')}")


if __name__ == "__main__":
    main()
