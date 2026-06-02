"""Plot per-epoch current-stage accuracy for permuted AR scheduler controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PATHS = [
    "results/class_incremental_ar_permuted_formal_global_cosine_formal20_permuted_20260523_231700.json",
    "results/class_incremental_ar_permuted_formal_stage_cosine_formal20_permuted_20260523_231700.json",
    "results/class_incremental_ar_permuted_formal_constant_formal20_permuted_20260523_231649.json",
]

MODE_ORDER = ["global_cosine", "stage_cosine", "constant"]
MODE_LABELS = {
    "global_cosine": "global cosine",
    "stage_cosine": "stage cosine",
    "constant": "constant LR",
}
MODE_COLORS = {
    "global_cosine": "#4C78A8",
    "stage_cosine": "#F58518",
    "constant": "#54A24B",
}
MODEL_ORDER = ["attention", "base_rmt_nmem16", "fmrmt_stable", "fmrmt_plastic"]
MODEL_LABELS = {
    "attention": "Transformer/MHA",
    "base_rmt_nmem16": "Base RMT n_mem=16",
    "fmrmt_stable": "FMRMT stable",
    "fmrmt_plastic": "FMRMT plastic",
}


def load_by_mode(paths: list[Path]) -> dict:
    by_mode = {}
    for path in paths:
        data = json.loads(path.read_text())
        mode = data["setup"]["lr_scheduler_modes"][0]
        by_mode[mode] = data
    return by_mode


def epoch_points(run: dict) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for stage_key, curve in sorted(
        run.get("current_stage_epoch_curves", {}).items(),
        key=lambda item: int(item[0]),
    ):
        for entry in curve:
            if entry.get("global_epoch") is None or entry.get("accuracy") is None:
                continue
            xs.append(int(entry["global_epoch"]) + 1)
            ys.append(float(entry["accuracy"]))
    return np.array(xs), np.array(ys)


def aggregate(data: dict, model_key: str):
    seed_series = []
    for run in data["runs"]:
        if run["model_key"] != model_key:
            continue
        x, y = epoch_points(run)
        if len(x):
            seed_series.append((run["seed"], x, y))
    if not seed_series:
        return [], np.array([]), np.array([]), np.array([])

    common_x = sorted(set.intersection(*(set(x.tolist()) for _, x, _ in seed_series)))
    means, stds = [], []
    for x_value in common_x:
        values = []
        for _, x, y in seed_series:
            idx = np.where(x == x_value)[0]
            if len(idx):
                values.append(float(y[idx[0]]))
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
    return seed_series, np.array(common_x), np.array(means), np.array(stds)


def decorate_epoch_axis(ax):
    ax.set_xlabel("Global epoch")
    ax.set_ylabel("Current-stage accuracy")
    ax.set_xlim(1, 160)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    for boundary in range(8, 160, 8):
        ax.axvline(boundary + 0.5, color="black", alpha=0.07, linewidth=0.7)


def plot_model_figures(by_mode: dict, outdir: Path):
    for model_key in MODEL_ORDER:
        fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
        for mode in MODE_ORDER:
            if mode not in by_mode:
                continue
            seed_series, x, mean, std = aggregate(by_mode[mode], model_key)
            for _, sx, sy in seed_series:
                ax.plot(sx, sy, color=MODE_COLORS[mode], alpha=0.18, linewidth=0.9)
            if len(x):
                ax.plot(
                    x,
                    mean,
                    color=MODE_COLORS[mode],
                    linewidth=2.4,
                    label=MODE_LABELS[mode],
                )
                ax.fill_between(
                    x,
                    np.clip(mean - std, 0, 1),
                    np.clip(mean + std, 0, 1),
                    color=MODE_COLORS[mode],
                    alpha=0.13,
                    linewidth=0,
                )
        ax.set_title(f"Current-stage accuracy per epoch: {MODEL_LABELS[model_key]}")
        decorate_epoch_axis(ax)
        ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.27))
        fig.savefig(outdir / f"current_stage_epoch_accuracy_{model_key}.png", dpi=180)
        fig.savefig(outdir / f"current_stage_epoch_accuracy_{model_key}.pdf")
        plt.close(fig)


def plot_grid(by_mode: dict, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for ax, model_key in zip(axes.ravel(), MODEL_ORDER):
        for mode in MODE_ORDER:
            if mode not in by_mode:
                continue
            _, x, mean, std = aggregate(by_mode[mode], model_key)
            if not len(x):
                continue
            ax.plot(x, mean, color=MODE_COLORS[mode], linewidth=2.0, label=MODE_LABELS[mode])
            ax.fill_between(
                x,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                color=MODE_COLORS[mode],
                alpha=0.12,
                linewidth=0,
            )
        ax.set_title(MODEL_LABELS[model_key])
        decorate_epoch_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
    fig.suptitle("20-stage permuted AR: current-stage accuracy per epoch")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(outdir / "current_stage_epoch_accuracy_grid.png", dpi=180)
    fig.savefig(outdir / "current_stage_epoch_accuracy_grid.pdf")
    plt.close(fig)


def plot_late_stage_figures(by_mode: dict, outdir: Path):
    for mode in MODE_ORDER:
        if mode not in by_mode:
            continue
        data = by_mode[mode]
        for model_key in MODEL_ORDER:
            fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
            stage_colors = plt.cm.viridis(np.linspace(0.15, 0.95, 5))
            for color, stage in zip(stage_colors, range(15, 20)):
                values = []
                for run in data["runs"]:
                    if run["model_key"] != model_key:
                        continue
                    curve = run.get("current_stage_epoch_curves", {}).get(str(stage), [])
                    vals = [float(e["accuracy"]) for e in curve if e.get("accuracy") is not None]
                    if vals:
                        values.append(vals)
                if not values:
                    continue
                arr = np.array(values, dtype=float)
                x = np.arange(1, arr.shape[1] + 1)
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                ax.plot(x, mean, color=color, linewidth=2.0, label=f"stage {stage}")
                ax.fill_between(
                    x,
                    np.clip(mean - std, 0, 1),
                    np.clip(mean + std, 0, 1),
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
            ax.set_title(f"Late-stage learning: {MODEL_LABELS[model_key]} ({MODE_LABELS[mode]})")
            ax.set_xlabel("Epoch within stage")
            ax.set_ylabel("Current-stage accuracy")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks(range(1, 9))
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.28))
            fig.savefig(outdir / f"late_stage_epoch_accuracy_{mode}_{model_key}.png", dpi=180)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", default=DEFAULT_PATHS)
    parser.add_argument("--outdir", default="results/figures/permuted_ar_epoch_accuracy_20260523")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    by_mode = load_by_mode([Path(path) for path in args.paths])

    plot_model_figures(by_mode, outdir)
    plot_grid(by_mode, outdir)
    plot_late_stage_figures(by_mode, outdir)

    print(outdir)
    for path in sorted(outdir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
