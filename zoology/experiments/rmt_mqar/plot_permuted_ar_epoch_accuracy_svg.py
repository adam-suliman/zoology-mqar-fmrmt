"""Dependency-free SVG plots for permuted AR per-epoch accuracy."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


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


def epoch_points(run: dict):
    points = []
    for _, curve in sorted(
        run.get("current_stage_epoch_curves", {}).items(),
        key=lambda item: int(item[0]),
    ):
        for entry in curve:
            epoch = entry.get("global_epoch")
            accuracy = entry.get("accuracy")
            if epoch is not None and accuracy is not None:
                points.append((int(epoch) + 1, float(accuracy)))
    return points


def aggregate(data: dict, model_key: str):
    by_epoch: dict[int, list[float]] = {}
    seed_series = []
    for run in data["runs"]:
        if run["model_key"] != model_key:
            continue
        points = epoch_points(run)
        seed_series.append(points)
        for epoch, accuracy in points:
            by_epoch.setdefault(epoch, []).append(accuracy)

    mean_points = []
    for epoch in sorted(by_epoch):
        values = by_epoch[epoch]
        mean_points.append((epoch, sum(values) / len(values)))
    return seed_series, mean_points


def aggregate_late_stage(data: dict, model_key: str, stage: int):
    by_epoch: dict[int, list[float]] = {}
    for run in data["runs"]:
        if run["model_key"] != model_key:
            continue
        curve = run.get("current_stage_epoch_curves", {}).get(str(stage), [])
        for entry in curve:
            stage_epoch = entry.get("stage_epoch")
            accuracy = entry.get("accuracy")
            if stage_epoch is not None and accuracy is not None:
                by_epoch.setdefault(int(stage_epoch), []).append(float(accuracy))

    return [
        (epoch, sum(values) / len(values))
        for epoch, values in sorted(by_epoch.items())
    ]


def path_data(points, x_min, x_max, y_min, y_max, left, top, width, height):
    coords = []
    for x, y in points:
        px = left + (x - x_min) / (x_max - x_min) * width
        py = top + (1.0 - (y - y_min) / (y_max - y_min)) * height
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def svg_header(width: int, height: int):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222} .small{font-size:12px} .label{font-size:14px} .title{font-size:18px;font-weight:600}</style>',
    ]


def draw_axes(lines, left, top, width, height, title, x_label, y_label):
    lines.append(f'<text class="title" x="{left}" y="30">{html.escape(title)}</text>')
    for y in [0.0, 0.25, 0.5, 0.75, 1.0]:
        py = top + (1.0 - y) * height
        lines.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left + width}" y2="{py:.2f}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text class="small" x="{left - 35}" y="{py + 4:.2f}">{y:.2f}</text>')
    for x in range(8, 160, 8):
        px = left + (x - 1) / 159 * width
        lines.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + height}" stroke="#eee" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#222" stroke-width="1.2"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#222" stroke-width="1.2"/>')
    lines.append(f'<text class="label" x="{left + width / 2 - 40}" y="{top + height + 45}">{html.escape(x_label)}</text>')
    lines.append(f'<text class="label" transform="translate(18 {top + height / 2 + 50}) rotate(-90)">{html.escape(y_label)}</text>')


def write_model_figure(by_mode: dict, model_key: str, outdir: Path):
    width, height = 1100, 470
    left, top, plot_w, plot_h = 75, 55, 960, 320
    lines = svg_header(width, height)
    draw_axes(
        lines,
        left,
        top,
        plot_w,
        plot_h,
        f"Current-stage accuracy per epoch: {MODEL_LABELS[model_key]}",
        "Global epoch",
        "Accuracy",
    )

    for mode in MODE_ORDER:
        if mode not in by_mode:
            continue
        seed_series, mean_points = aggregate(by_mode[mode], model_key)
        for points in seed_series:
            if not points:
                continue
            d = path_data(points, 1, 160, 0, 1, left, top, plot_w, plot_h)
            lines.append(f'<polyline points="{d}" fill="none" stroke="{MODE_COLORS[mode]}" stroke-opacity="0.18" stroke-width="1"/>')
        if mean_points:
            d = path_data(mean_points, 1, 160, 0, 1, left, top, plot_w, plot_h)
            lines.append(f'<polyline points="{d}" fill="none" stroke="{MODE_COLORS[mode]}" stroke-width="3"/>')

    legend_x = left
    legend_y = height - 35
    for i, mode in enumerate(MODE_ORDER):
        x = legend_x + i * 180
        lines.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 30}" y2="{legend_y}" stroke="{MODE_COLORS[mode]}" stroke-width="4"/>')
        lines.append(f'<text class="small" x="{x + 38}" y="{legend_y + 4}">{html.escape(MODE_LABELS[mode])}</text>')

    lines.append("</svg>")
    (outdir / f"current_stage_epoch_accuracy_{model_key}.svg").write_text("\n".join(lines))


def write_late_stage_figure(by_mode: dict, model_key: str, mode: str, outdir: Path):
    width, height = 900, 460
    left, top, plot_w, plot_h = 75, 55, 760, 300
    colors = ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"]
    lines = svg_header(width, height)
    draw_axes(
        lines,
        left,
        top,
        plot_w,
        plot_h,
        f"Late-stage learning: {MODEL_LABELS[model_key]} ({MODE_LABELS[mode]})",
        "Epoch within stage",
        "Accuracy",
    )
    for color, stage in zip(colors, range(15, 20)):
        points = aggregate_late_stage(by_mode[mode], model_key, stage)
        if not points:
            continue
        d = path_data(points, 1, 8, 0, 1, left, top, plot_w, plot_h)
        lines.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="3"/>')
    legend_y = height - 35
    for i, (color, stage) in enumerate(zip(colors, range(15, 20))):
        x = left + i * 135
        lines.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<text class="small" x="{x + 36}" y="{legend_y + 4}">stage {stage}</text>')
    lines.append("</svg>")
    (outdir / f"late_stage_epoch_accuracy_{mode}_{model_key}.svg").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", default=DEFAULT_PATHS)
    parser.add_argument("--outdir", default="results/figures/permuted_ar_epoch_accuracy_20260523")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    by_mode = load_by_mode([Path(path) for path in args.paths])

    for model_key in MODEL_ORDER:
        write_model_figure(by_mode, model_key, outdir)
    for mode in MODE_ORDER:
        for model_key in MODEL_ORDER:
            write_late_stage_figure(by_mode, model_key, mode, outdir)

    print(outdir)
    for path in sorted(outdir.glob("*.svg")):
        print(path)


if __name__ == "__main__":
    main()
