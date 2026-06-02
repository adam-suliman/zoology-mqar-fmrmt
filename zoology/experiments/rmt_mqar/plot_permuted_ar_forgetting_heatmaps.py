"""Plot forgetting heatmaps for formal permuted class-incremental AR runs."""

from __future__ import annotations

import argparse
import glob
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_GLOB = (
    "results/"
    "class_incremental_ar_permuted_formal_stage_onecycle_accumulate_"
    "formal20_permuted_*.json"
)
DEFAULT_MODELS = [
    "attention",
    "base_rmt_nmem16",
    "fmrmt_fast0_slow2",
    "fmrmt_lr0p005_slow2",
    "fmrmt_stable",
]
MODEL_LABELS = {
    "attention": "Transformer/MHA",
    "base_rmt_nmem16": "Base RMT n_mem=16",
    "fmrmt_fast0_slow2": "FMRMT no-fast slow2",
    "fmrmt_lr0p005_slow2": "FMRMT lr=0.005 slow2",
    "fmrmt_stable": "FMRMT stable slow4",
    "fmrmt_plastic": "FMRMT plastic slow1",
}
METRIC_LABELS = {
    "forgetting_from_learning": "Forgetting from learning",
    "forgetting": "Forgetting",
    "accuracy": "Accuracy",
    "bwt": "Backward transfer",
}


def _load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_10 = _load_font(10)
FONT_11 = _load_font(11)
FONT_12 = _load_font(12)
FONT_13_BOLD = _load_font(13, bold=True)
FONT_16_BOLD = _load_font(16, bold=True)


def _lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _interpolate_colors(stops: list[str], t: float) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    if t >= 1.0:
        return _hex_to_rgb(stops[-1])
    position = t * (len(stops) - 1)
    idx = int(position)
    frac = position - idx
    c0 = _hex_to_rgb(stops[idx])
    c1 = _hex_to_rgb(stops[idx + 1])
    return tuple(_lerp(a, b, frac) for a, b in zip(c0, c1))


def _value_to_color(value: float, metric: str) -> tuple[int, int, int]:
    if metric == "bwt":
        vmin, vmax = -1.0, 0.25
        t = (value - vmin) / (vmax - vmin)
        return _interpolate_colors(["#b2182b", "#f7f7f7", "#2166ac"], t)
    if metric == "accuracy":
        return _interpolate_colors(["#440154", "#31688e", "#35b779", "#fde725"], value)
    return _interpolate_colors(["#ffffcc", "#ffeda0", "#feb24c", "#f03b20", "#bd0026"], value)


def _text(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=(31, 31, 31), anchor=None):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _jsonable_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    rows = []
    for row in matrix.tolist():
        rows.append([None if value is None or np.isnan(value) else float(value) for value in row])
    return rows


def _matrix_to_array(matrix: list[list[Any]]) -> np.ndarray:
    return np.array(
        [
            [np.nan if value is None else float(value) for value in row]
            for row in matrix
        ],
        dtype=float,
    )


def load_runs(paths: list[Path], models: list[str]) -> dict[tuple[str, int], dict[str, Any]]:
    model_set = set(models)
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(paths):
        data = json.loads(path.read_text())
        for run in data.get("runs", []):
            model_key = run.get("model_key")
            seed = run.get("seed")
            if model_key not in model_set or seed is None:
                continue
            if "stage_end_matrices" not in run:
                continue
            key = (model_key, int(seed))
            selected[key] = {
                "run": run,
                "source_path": str(path),
            }
    return selected


def aggregate_matrices(
    selected: dict[tuple[str, int], dict[str, Any]],
    models: list[str],
    metric: str,
) -> dict[str, dict[str, Any]]:
    aggregates = {}
    for model_key in models:
        matrices = []
        seeds = []
        sources = []
        summaries = []
        for (candidate_model, seed), payload in sorted(selected.items()):
            if candidate_model != model_key:
                continue
            run = payload["run"]
            matrix = run.get("stage_end_matrices", {}).get(metric)
            if matrix is None:
                continue
            matrices.append(_matrix_to_array(matrix))
            seeds.append(seed)
            sources.append(payload["source_path"])
            summaries.append(run.get("summary", {}))
        if not matrices:
            continue
        stack = np.stack(matrices, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_matrix = np.nanmean(stack, axis=0)
            std_matrix = np.nanstd(stack, axis=0)
        aggregates[model_key] = {
            "mean": mean_matrix,
            "std": std_matrix,
            "n": len(matrices),
            "seeds": seeds,
            "sources": sources,
            "summary": {
                name: float(np.mean([
                    summary[name]
                    for summary in summaries
                    if isinstance(summary.get(name), (int, float))
                ]))
                for name in [
                    "continual/seen_avg_accuracy",
                    "continual/avg_learning_accuracy",
                    "continual/plasticity",
                    "continual/avg_bwt",
                    "continual/avg_forgetting_from_learning",
                ]
                if any(isinstance(summary.get(name), (int, float)) for summary in summaries)
            },
        }
    return aggregates


def _draw_colorbar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    height: int,
    metric: str,
    label: str,
):
    width = 18
    for offset in range(height):
        frac = 1.0 - offset / max(1, height - 1)
        if metric == "bwt":
            value = -1.0 + frac * 1.25
        else:
            value = frac
        draw.rectangle(
            [x, y + offset, x + width, y + offset],
            fill=_value_to_color(value, metric),
        )
    draw.rectangle([x, y, x + width, y + height], outline=(80, 80, 80), width=1)

    if metric == "bwt":
        ticks = [0.25, 0.0, -0.5, -1.0]
        tick_to_y = lambda value: y + int(round((1.0 - ((value + 1.0) / 1.25)) * height))
    else:
        ticks = [1.0, 0.75, 0.5, 0.25, 0.0]
        tick_to_y = lambda value: y + int(round((1.0 - value) * height))
    for tick in ticks:
        tick_y = tick_to_y(tick)
        draw.line([x + width, tick_y, x + width + 5, tick_y], fill=(80, 80, 80))
        _text(draw, (x + width + 8, tick_y - 6), f"{tick:g}", FONT_10)
    _text(draw, (x - 2, y - 24), label, FONT_11)


def _render_panel(
    model_key: str,
    aggregate: dict[str, Any],
    metric: str,
    individual: bool = False,
) -> Image.Image:
    matrix = aggregate["mean"]
    num_stages = matrix.shape[0]
    cell = 21 if individual else 18
    left = 86
    top = 92
    grid = num_stages * cell
    colorbar_space = 96 if individual else 0
    width = left + grid + 56 + colorbar_space
    height = top + grid + 54
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title = MODEL_LABELS.get(model_key, model_key)
    _text(draw, (18, 14), title, FONT_13_BOLD)
    summary = aggregate["summary"]
    seen = summary.get("continual/seen_avg_accuracy")
    bwt = summary.get("continual/avg_bwt")
    forget = summary.get("continual/avg_forgetting_from_learning")
    summary_bits = [f"n={aggregate['n']}", f"seeds={','.join(map(str, aggregate['seeds']))}"]
    if seen is not None:
        summary_bits.append(f"seen={seen:.3f}")
    if bwt is not None:
        summary_bits.append(f"BWT={bwt:.3f}")
    if forget is not None:
        summary_bits.append(f"forget={forget:.3f}")
    _text(draw, (18, 36), "  ".join(summary_bits), FONT_11, fill=(70, 70, 70))
    _text(
        draw,
        (18, 56),
        METRIC_LABELS.get(metric, metric),
        FONT_11,
        fill=(70, 70, 70),
    )

    x0 = left
    y0 = top
    for train_stage in range(num_stages):
        for eval_stage in range(num_stages):
            x = x0 + eval_stage * cell
            y = y0 + train_stage * cell
            value = matrix[train_stage, eval_stage]
            if np.isnan(value):
                fill = (244, 244, 244)
            else:
                fill = _value_to_color(float(value), metric)
            draw.rectangle([x, y, x + cell, y + cell], fill=fill)
            draw.rectangle([x, y, x + cell, y + cell], outline=(230, 230, 230), width=1)
    draw.rectangle([x0, y0, x0 + grid, y0 + grid], outline=(35, 35, 35), width=1)

    ticks = list(range(num_stages)) if individual else list(range(0, num_stages, 2))
    if num_stages - 1 not in ticks:
        ticks.append(num_stages - 1)
    for tick in ticks:
        center = x0 + tick * cell + cell / 2
        _text(draw, (center, y0 + grid + 8), str(tick), FONT_10, anchor="ma")
        center_y = y0 + tick * cell + cell / 2
        _text(draw, (x0 - 8, center_y - 6), str(tick), FONT_10, anchor="ra")

    _text(draw, (x0 + grid / 2, y0 + grid + 30), "Eval stage", FONT_11, anchor="ma")
    _text(draw, (14, y0 - 24), "After training stage", FONT_11)

    if individual:
        _draw_colorbar(
            draw,
            x=x0 + grid + 28,
            y=y0 + 8,
            height=min(330, grid - 16),
            metric=metric,
            label=METRIC_LABELS.get(metric, metric),
        )
    return image


def plot_grid(
    aggregates: dict[str, dict[str, Any]],
    models: list[str],
    metric: str,
    outdir: Path,
    prefix: str,
):
    plotted_models = [model for model in models if model in aggregates]
    if not plotted_models:
        raise RuntimeError("No model matrices found to plot")

    ncols = 3
    nrows = int(np.ceil(len(plotted_models) / ncols))
    panel_width = 520
    panel_height = 500
    header = 54
    colorbar_width = 170
    image = Image.new(
        "RGB",
        (ncols * panel_width + colorbar_width, nrows * panel_height + header),
        "white",
    )
    draw = ImageDraw.Draw(image)
    _text(
        draw,
        (24, 18),
        f"Formal20 permuted AR: mean {METRIC_LABELS.get(metric, metric).lower()} heatmaps",
        FONT_16_BOLD,
    )

    for index, model_key in enumerate(plotted_models):
        row = index // ncols
        col = index % ncols
        panel = _render_panel(model_key, aggregates[model_key], metric)
        image.paste(panel, (col * panel_width, header + row * panel_height))

    _draw_colorbar(
        draw,
        x=ncols * panel_width + 12,
        y=header + 54,
        height=330,
        metric=metric,
        label=METRIC_LABELS.get(metric, metric),
    )
    image.save(outdir / f"{prefix}_{metric}_grid.png")


def plot_individual(
    aggregates: dict[str, dict[str, Any]],
    models: list[str],
    metric: str,
    outdir: Path,
    prefix: str,
):
    for model_key in models:
        if model_key not in aggregates:
            continue
        panel = _render_panel(model_key, aggregates[model_key], metric, individual=True)
        panel.save(outdir / f"{prefix}_{metric}_{model_key}.png")


def write_aggregate_json(
    aggregates: dict[str, dict[str, Any]],
    metric: str,
    outdir: Path,
    prefix: str,
):
    payload = {
        "metric": metric,
        "models": {
            model_key: {
                "n": aggregate["n"],
                "seeds": aggregate["seeds"],
                "sources": aggregate["sources"],
                "summary": aggregate["summary"],
                "mean": _jsonable_matrix(aggregate["mean"]),
                "std": _jsonable_matrix(aggregate["std"]),
            }
            for model_key, aggregate in aggregates.items()
        },
    }
    (outdir / f"{prefix}_{metric}_matrices.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="*", default=None)
    parser.add_argument("--glob", default=DEFAULT_GLOB)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--metric", default="forgetting_from_learning")
    parser.add_argument(
        "--outdir",
        default="results/figures/permuted_ar_forgetting_heatmaps_20260524",
    )
    parser.add_argument("--prefix", default="formal20_stage_onecycle_accumulate")
    args = parser.parse_args()

    paths = [Path(path) for path in args.paths] if args.paths else [
        Path(path) for path in glob.glob(args.glob)
    ]
    if not paths:
        raise RuntimeError(f"No input files matched {args.glob!r}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected = load_runs(paths, args.models)
    aggregates = aggregate_matrices(selected, args.models, args.metric)
    plot_grid(aggregates, args.models, args.metric, outdir, args.prefix)
    plot_individual(aggregates, args.models, args.metric, outdir, args.prefix)
    write_aggregate_json(aggregates, args.metric, outdir, args.prefix)

    print(outdir)
    for model_key in args.models:
        if model_key not in aggregates:
            print(f"missing {model_key}")
            continue
        summary = aggregates[model_key]["summary"]
        print(
            f"{model_key}: n={aggregates[model_key]['n']} "
            f"seeds={aggregates[model_key]['seeds']} "
            f"seen={summary.get('continual/seen_avg_accuracy', float('nan')):.4f} "
            f"bwt={summary.get('continual/avg_bwt', float('nan')):.4f} "
            f"forget={summary.get('continual/avg_forgetting_from_learning', float('nan')):.4f}"
        )
    for path in sorted(outdir.glob(f"{args.prefix}_{args.metric}*")):
        print(path)


if __name__ == "__main__":
    main()
