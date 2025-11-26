from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt


def plot_backbone_comparison(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "auc",
    save_path: str | None = None,
    figsize: Tuple[int, int] = (8, 10),
) -> None:
    """Bar plot comparing backbones on a given metric."""

    if not all_results:
        raise ValueError("No results provided to plot_backbone_comparison.")

    data = []
    for backbone, metrics in all_results.items():
        if metrics and metrics.get("overall"):
            data.append((backbone, metrics["overall"].get(metric, 0.0)))

    if not data:
        raise ValueError("Overall metrics missing in all_results.")

    data.sort(key=lambda x: x[1], reverse=True)
    names, scores = zip(*data)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(names, scores, color="teal")
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Backbone comparison ({metric.upper()})")
    ax.invert_yaxis()

    for bar, score in zip(bars, scores):
        ax.text(score + 0.005, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

