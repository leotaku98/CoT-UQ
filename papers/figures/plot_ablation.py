# -*- coding: utf-8 -*-
"""Regenerate the band-depth hyperparameter ablation figures.

Reads the aggregated AUROC sweeps in ``output/ablation/*.json`` (produced by
``methods/bd_ensemble/run_ablation.py``) and renders one panel per swept
hyperparameter to ``papers/resources/abl_*.{pdf,png}``.

Each panel plots AUROC (x100) versus the swept value on SVAMP, one line per
backbone, for the main gBD method; a star marks each backbone's best setting.
"""

import json
import os

import matplotlib.pyplot as plt

# Repo root is two levels up from papers/figures/.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ABL_DIR = os.path.join(ROOT, "output", "ablation")
OUT_DIR = os.path.join(ROOT, "papers", "resources")

DATASET = "svamp"
METHOD = "gBD_v2"

# Backbone key in the JSON -> (legend label, colour, marker).
MODELS = [
    ("llama3-1_8B", "Llama-3.1-8B", "#2ca02c", "o"),
    ("qwen2.5-3b", "Qwen2.5-3B", "#1f77b4", "s"),
    ("llama2-13b", "Llama2-13B", "#ff7f0e", "^"),
]

# Per-panel config. Optional keys: log_x, y_ticks, x_ticks (display tick positions).
PANELS = [
    {"source": "sim_threshold.json", "basename": "abl_theta_c",
     "title": r"Cluster threshold $\theta_c$", "xlabel": r"$\theta_c$"},
    {"source": "cross_threshold.json", "basename": "abl_theta_e",
     "title": r"Edge threshold $\theta_e$", "xlabel": r"$\theta_e$"},
    {"source": "subset_size.json", "basename": "abl_J",
     "title": r"Band subset size $J$", "xlabel": r"$J$",
     "y_ticks": [60, 65, 70, 75, 80], "x_ticks": [5, 10, 15, 20]},
    {"source": "n_trials.json", "basename": "abl_T",
     "title": r"Monte-Carlo trials $T$", "xlabel": r"$T$", "log_x": True},
    {"source": "alpha.json", "basename": "abl_alpha",
     "title": r"Blend weight $\alpha$", "xlabel": r"$\alpha$"},
]


def _sorted_points(method_dict: dict) -> tuple:
    """Return (xs, ys) sorted by numeric x for one method's SVAMP sweep."""
    points = []
    for raw_value, per_dataset in method_dict.items():
        if DATASET not in per_dataset:
            continue
        points.append((float(raw_value), per_dataset[DATASET] * 100.0))
    points.sort(key=lambda pair: pair[0])
    xs = [pair[0] for pair in points]
    ys = [pair[1] for pair in points]
    return xs, ys


def render_panel(source: str, basename: str, title: str, xlabel: str,
                 log_x: bool = False, y_ticks: list = None, x_ticks: list = None) -> None:
    """Render a single ablation panel to PDF and PNG."""
    with open(os.path.join(ABL_DIR, source), encoding="utf-8") as handle:
        data = json.load(handle)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for model_key, label, colour, marker in MODELS:
        method_dict = data.get(model_key, {}).get(METHOD)
        if not method_dict:
            print(f"[skip] {source}: no {METHOD} for {model_key}")
            continue
        xs, ys = _sorted_points(method_dict)
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, color=colour, label=label, linewidth=1.8, markersize=6)
        best_index = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot(xs[best_index], ys[best_index], marker="*", color="black", markersize=15, zorder=5)

    if log_x:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(r"AUROC ($\times$100)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_ticks[0], y_ticks[-1])
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{basename}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {basename}.pdf / .png")


def main() -> None:
    """Render every ablation panel."""
    for panel in PANELS:
        render_panel(**panel)


if __name__ == "__main__":
    main()
