"""Paper-style figures for the formal20 permuted AR core comparison."""

from __future__ import annotations

import glob
import json
import math
import warnings
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUTDIR = Path("results/figures/permuted_ar_core_20260525")

PATTERNS = [
    "results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_core_*.json",
    "results/class_incremental_ar_permuted_online_ewc_core_*.json",
    "results/class_incremental_ar_permuted_si_core_*.json",
]

MODEL_ORDER = [
    "attention",
    "attention_online_ewc_lam100000",
    "attention_si_lam300",
    "base_rmt_nmem8",
    "fmrmt_lr0p005_slow2",
]

LABELS = {
    "attention": "Transformer",
    "attention_online_ewc_lam100000": "EWC",
    "attention_si_lam300": "SI",
    "base_rmt_nmem8": "Base RMT",
    "fmrmt_lr0p005_slow2": "FastMem RMT",
}

COLORS = {
    "attention": "#000000",
    "attention_online_ewc_lam100000": "#7f7f7f",
    "attention_si_lam300": "#009E73",
    "base_rmt_nmem8": "#CC79A7",
    "fmrmt_lr0p005_slow2": "#0072B2",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = font(44)
FONT_LABEL = font(34)
FONT_TICK = font(26)
FONT_LEGEND = font(25)
FONT_PANEL = font(28)
FONT_SMALL = font(22)


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def blend(color: str, alpha: float, background: str = "#ffffff") -> tuple[int, int, int]:
    fg = hex_rgb(color)
    bg = hex_rgb(background)
    return tuple(round(alpha * f + (1.0 - alpha) * b) for f, b in zip(fg, bg))


def text(draw: ImageDraw.ImageDraw, xy, value: str, font_obj, fill="#202124", anchor=None):
    draw.text(xy, value, font=font_obj, fill=fill, anchor=anchor)


def normalize_model_key(model_key: str) -> str:
    if model_key == "attention_no_ewc":
        return "attention"
    if model_key == "attention_no_si":
        return "attention"
    return model_key


def load_runs() -> dict[str, list[dict[str, Any]]]:
    runs_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODEL_ORDER}
    for pattern in PATTERNS:
        for path in sorted(glob.glob(pattern)):
            if "20260525_04" not in path:
                continue
            payload = json.loads(Path(path).read_text())
            setup = payload.get("setup", {})
            if setup.get("tasks") != ["formal20_permuted"]:
                continue
            for run in payload.get("runs", []):
                model_key = normalize_model_key(run.get("model_key", ""))
                if model_key in runs_by_model and run.get("stage_end_matrices"):
                    copied = dict(run)
                    copied["_source"] = path
                    runs_by_model[model_key].append(copied)
    return runs_by_model


def matrix(run: dict[str, Any], key: str) -> np.ndarray:
    values = run["stage_end_matrices"][key]
    return np.array(
        [[np.nan if value is None else float(value) for value in row] for row in values],
        dtype=float,
    )


def old_stage_curve(run: dict[str, Any], key: str) -> list[float]:
    values = matrix(run, key)
    curve = []
    for stage_idx in range(values.shape[0]):
        if stage_idx == 0:
            curve.append(np.nan)
            continue
        row = values[stage_idx, :stage_idx]
        curve.append(float(np.nanmean(row)))
    return curve


def aggregate_curve(runs: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    curves = np.array([old_stage_curve(run, key) for run in runs], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (
            np.nanmean(curves, axis=0).tolist(),
            np.nanstd(curves, axis=0).tolist(),
        )


def aggregate_heatmap(runs: list[dict[str, Any]], key: str) -> np.ndarray:
    matrices = np.array([matrix(run, key) for run in runs], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(matrices, axis=0)


def canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def y_coord(value: float, top: int, height: int, ymin: float, ymax: float) -> float:
    return top + height - ((value - ymin) / (ymax - ymin)) * height


def x_coord(value: float, left: int, width: int, xmin: float, xmax: float) -> float:
    return left + ((value - xmin) / (xmax - xmin)) * width


def draw_axes(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xticks: list[int],
    yticks: list[float],
    xlabel: str,
    ylabel: str,
):
    axis = "#202124"
    grid = "#dddddd"
    draw.line((left, top + height, left + width, top + height), fill=axis, width=3)
    draw.line((left, top, left, top + height), fill=axis, width=3)
    for tick in yticks:
        y = y_coord(tick, top, height, ymin, ymax)
        draw.line((left, y, left + width, y), fill=grid, width=2)
        draw.line((left - 10, y, left, y), fill=axis, width=3)
        text(draw, (left - 18, y), f"{tick:g}", FONT_TICK, fill="#202124", anchor="rm")
    for tick in xticks:
        x = x_coord(tick, left, width, xmin, xmax)
        draw.line((x, top + height, x, top + height + 10), fill=axis, width=3)
        text(draw, (x, top + height + 24), str(tick), FONT_TICK, fill="#202124", anchor="mt")
    text(draw, (left + width / 2, top + height + 82), xlabel, FONT_LABEL, anchor="mm")


def paste_vertical_label(
    image: Image.Image,
    label: str,
    *,
    x: int,
    center_y: float,
):
    layer = Image.new("RGBA", (900, 90), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((450, 45), label, font=FONT_LABEL, fill="#202124", anchor="mm")
    rotated = layer.rotate(90, expand=True)
    image.paste(rotated, (x, round(center_y - rotated.height / 2)), rotated)


def draw_curve_figure(
    path: Path,
    *,
    title: str,
    ylabel: str,
    y_range: tuple[float, float],
    yticks: list[float],
    curves: dict[str, tuple[list[float], list[float]]],
):
    image, draw = canvas(1900, 1180)
    left, top, width, height = 230, 120, 1540, 760
    text(draw, (left + width / 2, 40), title, FONT_TITLE, anchor="mt")
    draw_axes(
        draw,
        left=left,
        top=top,
        width=width,
        height=height,
        xmin=1,
        xmax=20,
        ymin=y_range[0],
        ymax=y_range[1],
        xticks=[1, 5, 10, 15, 20],
        yticks=yticks,
        xlabel="Training stage",
        ylabel=ylabel,
    )
    paste_vertical_label(image, ylabel, x=24, center_y=top + height / 2)

    band_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    band_draw = ImageDraw.Draw(band_layer)

    for model_key in MODEL_ORDER:
        if model_key not in curves:
            continue
        means, stds = curves[model_key]
        xs = [idx + 1 for idx, value in enumerate(means) if not math.isnan(value)]
        ys = [value for value in means if not math.isnan(value)]
        es = [stds[idx - 1] for idx in xs]
        color = COLORS[model_key]
        upper = [
            (x_coord(x, left, width, 1, 20), y_coord(min(y_range[1], y + e), top, height, *y_range))
            for x, y, e in zip(xs, ys, es)
        ]
        lower = [
            (x_coord(x, left, width, 1, 20), y_coord(max(y_range[0], y - e), top, height, *y_range))
            for x, y, e in reversed(list(zip(xs, ys, es)))
        ]
        if len(upper) >= 2:
            band_draw.polygon(
                upper + lower,
                fill=(*hex_rgb(color), 34),
            )

    image.alpha_composite(band_layer.convert("RGBA")) if image.mode == "RGBA" else None
    if image.mode != "RGBA":
        image_rgba = image.convert("RGBA")
        image_rgba.alpha_composite(band_layer)
        image.paste(image_rgba.convert("RGB"))

    for model_key in MODEL_ORDER:
        if model_key not in curves:
            continue
        means, stds = curves[model_key]
        xs = [idx + 1 for idx, value in enumerate(means) if not math.isnan(value)]
        ys = [value for value in means if not math.isnan(value)]
        color = COLORS[model_key]
        points = [
            (x_coord(x, left, width, 1, 20), y_coord(y, top, height, *y_range))
            for x, y in zip(xs, ys)
        ]
        if len(points) >= 2:
            draw.line(points, fill=color, width=5, joint="curve")
        for point in points:
            x, y = point
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)

    legend_y = 1010
    legend_items = [(model, LABELS[model]) for model in MODEL_ORDER if model in curves]
    column_widths = [390, 300, 300, 340, 360]
    x0 = 190
    for idx, (model_key, label) in enumerate(legend_items):
        row = idx // 5
        col = idx % 5
        x = x0 + sum(column_widths[:col])
        y = legend_y + row * 54
        color = COLORS[model_key]
        draw.line((x, y, x + 48, y), fill=color, width=7)
        draw.ellipse((x + 18, y - 8, x + 34, y + 8), fill=color)
        text(draw, (x + 66, y), label, FONT_LEGEND, anchor="lm")

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def interpolate(stops: list[str], t: float) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    if t >= 1.0:
        return hex_rgb(stops[-1])
    scaled = t * (len(stops) - 1)
    idx = int(scaled)
    frac = scaled - idx
    c0 = hex_rgb(stops[idx])
    c1 = hex_rgb(stops[idx + 1])
    return tuple(lerp(a, b, frac) for a, b in zip(c0, c1))


def heat_color(value: float) -> tuple[int, int, int]:
    return interpolate(["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#f03b20", "#bd0026"], value)


def draw_heatmap_panel(
    draw: ImageDraw.ImageDraw,
    matrix_values: np.ndarray,
    *,
    x: int,
    y: int,
    cell: int,
    title: str,
):
    n = matrix_values.shape[0]
    text(draw, (x + n * cell / 2, y - 48), title, FONT_PANEL, anchor="mm")
    for row in range(n):
        for col in range(n):
            value = matrix_values[row, col]
            if np.isnan(value):
                fill = "#ffffff"
            else:
                fill = heat_color(float(min(max(value, 0.0), 1.0)))
            x0 = x + col * cell
            y0 = y + row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=fill, outline="#eeeeee")
    draw.rectangle((x, y, x + n * cell, y + n * cell), outline="#202124", width=2)
    for tick in [0, 5, 10, 15, 19]:
        tx = x + tick * cell + cell / 2
        ty = y + tick * cell + cell / 2
        text(draw, (tx, y + n * cell + 16), str(tick), FONT_SMALL, anchor="mt")
        text(draw, (x - 14, ty), str(tick), FONT_SMALL, anchor="rm")


def draw_heatmap_figure(path: Path, heatmaps: dict[str, np.ndarray]):
    image, draw = canvas(2300, 1660)
    text(draw, (1150, 38), "Forgetting From Learning", FONT_TITLE, anchor="mt")

    cell = 31
    panel_w = 20 * cell
    x_positions = [130, 850, 1570]
    y_positions = [170, 920]
    models = [
        "attention",
        "attention_online_ewc_lam100000",
        "attention_si_lam300",
        "base_rmt_nmem8",
        "fmrmt_lr0p005_slow2",
    ]
    for idx, model_key in enumerate(models):
        row = idx // 3
        col = idx % 3
        draw_heatmap_panel(
            draw,
            heatmaps[model_key],
            x=x_positions[col],
            y=y_positions[row],
            cell=cell,
            title=LABELS[model_key],
        )

    text(draw, (1150, 1600), "Eval stage", FONT_LABEL, anchor="mm")
    ylabel_layer = Image.new("RGBA", (700, 90), (255, 255, 255, 0))
    ylabel_draw = ImageDraw.Draw(ylabel_layer)
    ylabel_draw.text((350, 45), "After training stage", font=FONT_LABEL, fill="#202124", anchor="mm")
    rotated = ylabel_layer.rotate(90, expand=True)
    image.paste(rotated, (16, 520), rotated)

    cbar_x = 1720
    cbar_y = 1020
    cbar_h = 380
    cbar_w = 34
    for offset in range(cbar_h):
        frac = 1.0 - offset / max(1, cbar_h - 1)
        draw.rectangle(
            (cbar_x, cbar_y + offset, cbar_x + cbar_w, cbar_y + offset),
            fill=heat_color(frac),
        )
    draw.rectangle((cbar_x, cbar_y, cbar_x + cbar_w, cbar_y + cbar_h), outline="#202124", width=2)
    for tick in [1.0, 0.75, 0.5, 0.25, 0.0]:
        ty = cbar_y + (1.0 - tick) * cbar_h
        draw.line((cbar_x + cbar_w, ty, cbar_x + cbar_w + 9, ty), fill="#202124", width=2)
        text(draw, (cbar_x + cbar_w + 16, ty), f"{tick:g}", FONT_SMALL, anchor="lm")

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    runs_by_model = load_runs()
    missing = [model for model in MODEL_ORDER if not runs_by_model.get(model)]
    if missing:
        raise RuntimeError(f"Missing runs for: {missing}")

    retained_curves = {
        model: aggregate_curve(runs_by_model[model], "accuracy")
        for model in MODEL_ORDER
    }
    heatmaps = {
        model: aggregate_heatmap(runs_by_model[model], "forgetting_from_learning")
        for model in MODEL_ORDER
    }

    retained_path = OUTDIR / "core_retained_accuracy_vs_stage.png"
    heatmap_path = OUTDIR / "core_forgetting_heatmaps.png"

    draw_curve_figure(
        retained_path,
        title="Retained Accuracy",
        ylabel="Mean old-stage accuracy",
        y_range=(0.0, 1.03),
        yticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        curves=retained_curves,
    )
    draw_heatmap_figure(heatmap_path, heatmaps)

    manifest = {
        "outdir": str(OUTDIR),
        "figures": [str(retained_path), str(heatmap_path)],
        "models": MODEL_ORDER,
        "sources": sorted({
            run["_source"]
            for runs in runs_by_model.values()
            for run in runs
        }),
        "notes": {
            "retained_accuracy": "Mean accuracy over old seen stages only; current-stage learning accuracy is excluded.",
            "heatmap": "Mean forgetting-from-learning matrix across seeds.",
        },
    }
    (OUTDIR / "core_figure_sources.json").write_text(json.dumps(manifest, indent=2))
    for figure in manifest["figures"]:
        print(figure)


if __name__ == "__main__":
    main()
