"""Generate stage-wise paper figures for formal permuted AR runs.

This intentionally uses only PIL so the figures can be regenerated in the
project venv without adding matplotlib as a dependency.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


OUTDIR = Path("results/figures/permuted_ar_stage_paper_20260525")
RESULTS = Path("results")

MODEL_LABELS = {
    "attention": "Transformer",
    "base_rmt_nmem16": "Base RMT",
    "fmrmt_lr0p005_slow2": "FastMem RMT",
    "fmrmt_fast0_slow2": "No-fast RMT control",
    "attention_online_ewc_lam100000": "Online EWC",
    "attention_si_lam300": "Synaptic Intelligence",
}

COLORS = {
    "attention": "#4C78A8",
    "base_rmt_nmem16": "#F58518",
    "fmrmt_lr0p005_slow2": "#54A24B",
    "fmrmt_fast0_slow2": "#B279A2",
    "attention_online_ewc_lam100000": "#E45756",
    "attention_si_lam300": "#72B7B2",
    "reservoir": "#4C78A8",
    "stored": "#F58518",
    "base_stored": "#54A24B",
    "fastmem_stored": "#B279A2",
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


TITLE = _font(54, bold=True)
SUBTITLE = _font(31)
AXIS = _font(34)
TICK = _font(28)
LEGEND = _font(30)
PANEL_TITLE = _font(34, bold=True)
PANEL_SUB = _font(25)
SMALL = _font(23)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def text_size(draw: ImageDraw.ImageDraw, value: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), value, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    font: ImageFont.ImageFont,
    fill: str = "#202124",
    anchor: str | None = None,
) -> None:
    draw.text(xy, value, font=font, fill=fill, anchor=anchor)


def read_runs(patterns: Iterable[str], *, segment_len: int | None = None) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            data = load_json(path)
            setup = data.get("setup", {})
            if setup.get("tasks") != ["formal20_permuted"]:
                continue
            if segment_len is not None and setup.get("segment_len") != segment_len:
                continue
            for run in data.get("runs", []):
                copied = dict(run)
                copied["_source"] = path
                copied["_created_at"] = data.get("created_at", "")
                copied["_setup"] = setup
                runs.append(copied)
    return runs


def latest_by_model_seed(runs: Iterable[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for run in sorted(runs, key=lambda item: (item.get("_source", ""), item.get("run_id", ""))):
        selected[(run["model_key"], int(run["seed"]))] = run
    return selected


def formal64_runs() -> dict[tuple[str, int], dict[str, Any]]:
    return latest_by_model_seed(
        read_runs(
            [
                str(
                    RESULTS
                    / "class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_*.json"
                )
            ],
            segment_len=64,
        )
    )


def formal128_runs() -> dict[tuple[str, int], dict[str, Any]]:
    return latest_by_model_seed(
        read_runs(
            [
                str(
                    RESULTS
                    / "class_incremental_ar_permuted_formal_stage_onecycle_accumulate_segment_len128_formal20_permuted_20260524_*.json"
                )
            ],
            segment_len=128,
        )
    )


def si_runs() -> dict[tuple[str, int], dict[str, Any]]:
    return latest_by_model_seed(
        read_runs([str(RESULTS / "class_incremental_ar_permuted_si_formal20_permuted_*.json")])
    )


def ewc_runs() -> dict[tuple[str, int], dict[str, Any]]:
    return latest_by_model_seed(
        read_runs([str(RESULTS / "class_incremental_ar_permuted_online_ewc_formal20_permuted_*.json")])
    )


def summary_metric(run: dict[str, Any], key: str) -> float | None:
    value = run.get("summary", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def per_stage_metric(
    selected: dict[tuple[str, int], dict[str, Any]],
    model_key: str,
    metric_key: str,
) -> tuple[list[int], list[float], list[float], int]:
    by_stage: dict[int, list[float]] = {}
    seeds = set()
    for (candidate, seed), run in selected.items():
        if candidate != model_key:
            continue
        seeds.add(seed)
        for entry in run.get("stage_end_history", []):
            stage = int(entry["continual/stage"]) + 1
            value = entry.get(metric_key)
            if isinstance(value, (int, float)):
                by_stage.setdefault(stage, []).append(float(value))
    xs, ys, errs = [], [], []
    for stage in sorted(by_stage):
        values = by_stage[stage]
        xs.append(stage)
        ys.append(mean(values))
        errs.append(pstdev(values) if len(values) > 1 else 0.0)
    return xs, ys, errs, len(seeds)


def final_stats(
    selected: dict[tuple[str, int], dict[str, Any]],
    model_key: str,
    metric_key: str,
) -> tuple[float, float, int]:
    values = [
        value
        for (candidate, _), run in selected.items()
        if candidate == model_key
        for value in [summary_metric(run, metric_key)]
        if value is not None
    ]
    if not values:
        return math.nan, math.nan, 0
    return mean(values), pstdev(values) if len(values) > 1 else 0.0, len(values)


def y_pos(value: float, top: int, height: int, ymin: float, ymax: float) -> float:
    return top + height - ((value - ymin) / (ymax - ymin)) * height


def x_pos(value: float, left: int, width: int, xmin: float, xmax: float) -> float:
    return left + ((value - xmin) / (xmax - xmin)) * width


def draw_header(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str,
    *,
    x: int = 90,
    y: int = 62,
) -> None:
    draw_text(draw, (x, y), title, TITLE)
    draw_text(draw, (x, y + 70), subtitle, SUBTITLE, fill="#5f6368")


def draw_legend(
    draw: ImageDraw.ImageDraw,
    items: list[dict[str, Any]],
    *,
    left: int,
    top: int,
    max_width: int,
) -> int:
    x = left
    y = top
    row_height = 48
    for item in items:
        label = item["label"]
        label_w, _ = text_size(draw, label, LEGEND)
        item_w = label_w + 92
        if x + item_w > left + max_width and x > left:
            x = left
            y += row_height
        color = item["color"]
        draw.line((x, y + 19, x + 46, y + 19), fill=color, width=8)
        draw.ellipse((x + 16, y + 8, x + 30, y + 22), fill=color, outline="white", width=2)
        draw_text(draw, (x + 60, y + 18), label, LEGEND, fill="#202124", anchor="lm")
        x += item_w
    return y + row_height


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
    xticks: list[float],
    yticks: list[float],
    x_label: str,
    y_label: str,
) -> None:
    axis_color = "#202124"
    grid = "#e8eaed"
    for tick in yticks:
        y = y_pos(tick, top, height, ymin, ymax)
        draw.line((left, y, left + width, y), fill=grid, width=2)
        draw_text(draw, (left - 18, y), f"{tick:g}", TICK, fill="#3c4043", anchor="rm")
    for tick in xticks:
        x = x_pos(tick, left, width, xmin, xmax)
        draw.line((x, top, x, top + height), fill="#f1f3f4", width=2)
        draw_text(draw, (x, top + height + 44), f"{int(tick)}", TICK, fill="#3c4043", anchor="mm")
    draw.line((left, top, left, top + height), fill=axis_color, width=3)
    draw.line((left, top + height, left + width, top + height), fill=axis_color, width=3)
    draw_text(draw, (left, top - 56), y_label, AXIS, fill="#202124")
    draw_text(draw, (left + width / 2, top + height + 105), x_label, AXIS, fill="#202124", anchor="mm")


def line_points(
    xs: list[int],
    ys: list[float],
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> list[tuple[float, float]]:
    return [
        (x_pos(x, left, width, xmin, xmax), y_pos(y, top, height, ymin, ymax))
        for x, y in zip(xs, ys)
    ]


def make_line_figure(
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    series: list[dict[str, Any]],
    y_label: str,
    x_label: str = "Training stage",
    xmin: float = 1,
    xmax: float = 20,
    ymin: float = 0,
    ymax: float = 1.02,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
) -> None:
    image = Image.new("RGBA", (2500, 1550), "white")
    draw = ImageDraw.Draw(image)
    draw_header(draw, title, subtitle)
    left, top, width, height = 190, 330, 2180, 900
    legend_bottom = draw_legend(draw, series, left=left, top=190, max_width=width)
    if legend_bottom > top - 25:
        top = legend_bottom + 40
        height = 1230 - top
    draw_axes(
        draw,
        left=left,
        top=top,
        width=width,
        height=height,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        xticks=xticks or [1, 5, 10, 15, 20],
        yticks=yticks or [0, 0.25, 0.5, 0.75, 1.0],
        x_label=x_label,
        y_label=y_label,
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for item in series:
        xs = item["x"]
        ys = item["y"]
        errs = item.get("err", [0.0] * len(xs))
        color = item["color"]
        rgb = hex_to_rgb(color)
        if any(err > 0 for err in errs):
            upper = [
                (
                    x_pos(x, left, width, xmin, xmax),
                    y_pos(min(ymax, y + err), top, height, ymin, ymax),
                )
                for x, y, err in zip(xs, ys, errs)
            ]
            lower = [
                (
                    x_pos(x, left, width, xmin, xmax),
                    y_pos(max(ymin, y - err), top, height, ymin, ymax),
                )
                for x, y, err in zip(xs, ys, errs)
            ]
            overlay_draw.polygon(upper + lower[::-1], fill=rgb + (35,))
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)
    for item in series:
        points = line_points(
            item["x"],
            item["y"],
            left=left,
            top=top,
            width=width,
            height=height,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
        if len(points) > 1:
            draw.line(points, fill=item["color"], width=8, joint="curve")
        for x, y in points:
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=item["color"], outline="white", width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path, dpi=(300, 300))


def make_stage_figures(formal64, ewc, si) -> None:
    core_models = [
        "attention",
        "base_rmt_nmem16",
        "fmrmt_lr0p005_slow2",
        "fmrmt_fast0_slow2",
    ]
    core_series = []
    for model_key in core_models:
        xs, ys, errs, _ = per_stage_metric(formal64, model_key, "continual/seen_avg_accuracy")
        core_series.append(
            {
                "label": MODEL_LABELS[model_key],
                "color": COLORS[model_key],
                "x": xs,
                "y": ys,
                "err": errs,
            }
        )
    make_line_figure(
        OUTDIR / "fig1_seen_accuracy_by_stage_core.png",
        title="Permuted AR Retention Across Stages",
        subtitle="Stage-local seen accuracy, mean +/- std across 3 seeds",
        series=core_series,
        y_label="Seen accuracy",
    )

    cl_selected = dict(formal64)
    cl_selected.update(
        {
            key: value
            for key, value in ewc.items()
            if key[0] == "attention_online_ewc_lam100000"
        }
    )
    cl_selected.update({key: value for key, value in si.items() if key[0] == "attention_si_lam300"})
    cl_models = [
        "attention",
        "attention_online_ewc_lam100000",
        "attention_si_lam300",
        "fmrmt_lr0p005_slow2",
    ]
    cl_series = []
    for model_key in cl_models:
        xs, ys, errs, _ = per_stage_metric(cl_selected, model_key, "continual/seen_avg_accuracy")
        cl_series.append(
            {
                "label": MODEL_LABELS[model_key],
                "color": COLORS[model_key],
                "x": xs,
                "y": ys,
                "err": errs,
            }
        )
    make_line_figure(
        OUTDIR / "fig2_seen_accuracy_by_stage_cl_baselines.png",
        title="No-Replay Baselines Across Stages",
        subtitle="Stage-local seen accuracy, mean +/- std across 3 seeds",
        series=cl_series,
        y_label="Seen accuracy",
    )

    learning_models = [
        "attention",
        "base_rmt_nmem16",
        "attention_si_lam300",
        "fmrmt_lr0p005_slow2",
    ]
    learning_selected = dict(formal64)
    learning_selected.update({key: value for key, value in si.items() if key[0] == "attention_si_lam300"})
    learning_series = []
    for model_key in learning_models:
        xs, ys, errs, _ = per_stage_metric(learning_selected, model_key, "continual/current_stage_accuracy")
        learning_series.append(
            {
                "label": MODEL_LABELS[model_key],
                "color": COLORS[model_key],
                "x": xs,
                "y": ys,
                "err": errs,
            }
        )
    make_line_figure(
        OUTDIR / "fig3_current_task_learning_by_stage.png",
        title="Current-Task Learning Across Stages",
        subtitle="Current-stage test accuracy after each stage, mean +/- std across 3 seeds",
        series=learning_series,
        y_label="Current-task accuracy",
    )


def matrix_for_model(
    selected: dict[tuple[str, int], dict[str, Any]],
    model_key: str,
    matrix_key: str = "forgetting_from_learning",
) -> tuple[list[list[float | None]], int]:
    matrices = []
    for (candidate, _), run in selected.items():
        if candidate != model_key:
            continue
        matrix = run.get("stage_end_matrices", {}).get(matrix_key)
        if matrix:
            matrices.append(matrix)
    if not matrices:
        raise ValueError(f"No {matrix_key} matrices for {model_key}")
    n_rows = len(matrices[0])
    n_cols = len(matrices[0][0])
    averaged: list[list[float | None]] = []
    for row_idx in range(n_rows):
        row: list[float | None] = []
        for col_idx in range(n_cols):
            values = [
                matrix[row_idx][col_idx]
                for matrix in matrices
                if isinstance(matrix[row_idx][col_idx], (int, float))
            ]
            row.append(mean(values) if values else None)
        averaged.append(row)
    return averaged, len(matrices)


def heat_color(value: float | None) -> tuple[int, int, int]:
    if value is None:
        return (245, 247, 250)
    value = max(0.0, min(1.0, value))
    if value < 0.5:
        frac = value / 0.5
        c0 = (255, 250, 205)
        c1 = (253, 174, 97)
    else:
        frac = (value - 0.5) / 0.5
        c0 = (253, 174, 97)
        c1 = (215, 25, 28)
    return tuple(int(c0[idx] + (c1[idx] - c0[idx]) * frac) for idx in range(3))


def draw_heatmap_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    matrix: list[list[float | None]],
    x0: int,
    y0: int,
    cell: int,
    title: str,
    subtitle: str,
) -> None:
    draw_text(draw, (x0, y0 - 76), title, PANEL_TITLE)
    draw_text(draw, (x0, y0 - 35), subtitle, PANEL_SUB, fill="#5f6368")
    n = len(matrix)
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            x = x0 + col_idx * cell
            y = y0 + row_idx * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=heat_color(value))
            draw.rectangle((x, y, x + cell, y + cell), outline="#e0e0e0", width=1)
    for tick in [0, 5, 10, 15, 19]:
        x = x0 + tick * cell + cell / 2
        y = y0 + tick * cell + cell / 2
        draw_text(draw, (x, y0 + n * cell + 34), str(tick), SMALL, fill="#3c4043", anchor="mm")
        draw_text(draw, (x0 - 20, y), str(tick), SMALL, fill="#3c4043", anchor="rm")
    draw_text(draw, (x0 + n * cell / 2, y0 + n * cell + 74), "Eval stage", SMALL, fill="#202124", anchor="mm")


def make_heatmap(formal64) -> None:
    image = Image.new("RGB", (2700, 1900), "white")
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        "Forgetting Heatmaps",
        "Rows are training checkpoints, columns are eval stages; mean forgetting from learning across 3 seeds",
    )
    draw_text(draw, (120, 196), "Stage-local formal20 permuted AR", SUBTITLE, fill="#5f6368")
    models = [
        "attention",
        "base_rmt_nmem16",
        "fmrmt_lr0p005_slow2",
        "fmrmt_fast0_slow2",
    ]
    positions = [(180, 360), (1325, 360), (180, 1120), (1325, 1120)]
    cell = 30
    for model_key, (x, y) in zip(models, positions):
        matrix, n = matrix_for_model(formal64, model_key)
        seen, _, _ = final_stats(formal64, model_key, "continual/seen_avg_accuracy")
        bwt, _, _ = final_stats(formal64, model_key, "continual/avg_bwt")
        subtitle = f"Seen {seen:.3f}, BWT {bwt:.3f}, across {n} seeds"
        draw_heatmap_panel(
            image,
            draw,
            matrix=matrix,
            x0=x,
            y0=y,
            cell=cell,
            title=MODEL_LABELS[model_key],
            subtitle=subtitle,
        )
    # Shared colorbar.
    bar_x, bar_y, bar_w, bar_h = 2495, 520, 46, 760
    for idx in range(bar_h):
        value = 1.0 - idx / (bar_h - 1)
        draw.line((bar_x, bar_y + idx, bar_x + bar_w, bar_y + idx), fill=heat_color(value), width=1)
    draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), outline="#202124", width=2)
    draw_text(draw, (bar_x - 15, bar_y - 42), "Forgetting", PANEL_SUB, fill="#202124")
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = bar_y + bar_h - tick * bar_h
        draw.line((bar_x + bar_w, y, bar_x + bar_w + 12, y), fill="#202124", width=2)
        draw_text(draw, (bar_x + bar_w + 20, y), f"{tick:g}", SMALL, fill="#3c4043", anchor="lm")
    out_path = OUTDIR / "fig4_forgetting_heatmaps_core.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, dpi=(300, 300))


def replay_stats(source: str) -> dict[int, dict[str, dict[str, float]]]:
    if source == "reservoir":
        pattern = (
            RESULTS
            / "class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_reservoir_replay*_buffer*_formal20_permuted_20260524_*.json"
        )
    elif source == "stored":
        pattern = (
            RESULTS
            / "class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay*_formal20_permuted_20260524_*.json"
        )
    else:
        raise ValueError(source)
    grouped: dict[int, dict[str, list[float]]] = {}
    seen_paths = sorted(glob.glob(str(pattern)))
    for path in seen_paths:
        data = load_json(path)
        if data.get("setup", {}).get("tasks") != ["formal20_permuted"]:
            continue
        for run in data.get("runs", []):
            value = summary_metric(run, "continual/seen_avg_accuracy")
            if value is None:
                continue
            budget = int(run["replay_examples_per_old_stage"])
            grouped.setdefault(budget, {}).setdefault(run["model_key"], []).append(value)
    stats: dict[int, dict[str, dict[str, float]]] = {}
    for budget, by_model in grouped.items():
        stats[budget] = {}
        for model_key, values in by_model.items():
            stats[budget][model_key] = {
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "n": float(len(values)),
            }
    return stats


def make_replay_figure() -> None:
    reservoir = replay_stats("reservoir")
    stored = replay_stats("stored")
    budgets = [1, 2, 4, 8, 16]
    series = [
        {
            "label": "Transformer, reservoir replay",
            "color": COLORS["reservoir"],
            "x": budgets,
            "y": [reservoir[b]["attention"]["mean"] for b in budgets],
            "err": [reservoir[b]["attention"]["std"] for b in budgets],
        },
        {
            "label": "Transformer, balanced replay",
            "color": COLORS["stored"],
            "x": budgets,
            "y": [stored[b]["attention"]["mean"] for b in budgets],
            "err": [stored[b]["attention"]["std"] for b in budgets],
        },
        {
            "label": "Base RMT, balanced replay",
            "color": COLORS["base_stored"],
            "x": budgets,
            "y": [stored[b]["base_rmt_nmem16"]["mean"] for b in budgets],
            "err": [stored[b]["base_rmt_nmem16"]["std"] for b in budgets],
        },
        {
            "label": "FastMem RMT, balanced replay",
            "color": COLORS["fastmem_stored"],
            "x": budgets,
            "y": [stored[b]["fmrmt_lr0p005_slow2"]["mean"] for b in budgets],
            "err": [stored[b]["fmrmt_lr0p005_slow2"]["std"] for b in budgets],
        },
    ]
    make_line_figure(
        OUTDIR / "fig5_replay_budget_curve.png",
        title="Replay Baselines",
        subtitle="Final seen accuracy vs old examples retained per prior stage, mean +/- std across 3 seeds",
        series=series,
        y_label="Final seen accuracy",
        x_label="Replay examples per old stage",
        xmin=1,
        xmax=16,
        xticks=budgets,
    )


def make_segment_figure(formal64, formal128) -> None:
    series = []
    for model_key, selected, suffix in [
        ("base_rmt_nmem16", formal64, "segment 64"),
        ("base_rmt_nmem16", formal128, "segment 128"),
        ("fmrmt_lr0p005_slow2", formal64, "segment 64"),
        ("fmrmt_lr0p005_slow2", formal128, "segment 128"),
    ]:
        xs, ys, errs, _ = per_stage_metric(selected, model_key, "continual/seen_avg_accuracy")
        base_color = COLORS[model_key]
        label = f"{MODEL_LABELS[model_key]}, {suffix}"
        color = base_color if suffix == "segment 64" else ("#D95F02" if model_key == "base_rmt_nmem16" else "#1B9E77")
        series.append({"label": label, "color": color, "x": xs, "y": ys, "err": errs})
    make_line_figure(
        OUTDIR / "fig6_segment_length_seen_accuracy.png",
        title="Segment-Length Control",
        subtitle="Stage-local seen accuracy with and without cross-segment recurrence, mean +/- std across 3 seeds",
        series=series,
        y_label="Seen accuracy",
    )


def write_sources(formal64, formal128, ewc, si) -> None:
    payload = {
        "figures": sorted(str(path) for path in OUTDIR.glob("*.png")),
        "sources": {
            "formal20_segment64": sorted({run["_source"] for run in formal64.values()}),
            "formal20_segment128": sorted({run["_source"] for run in formal128.values()}),
            "online_ewc": sorted({run["_source"] for run in ewc.values()}),
            "synaptic_intelligence": sorted({run["_source"] for run in si.values()}),
            "existing_heatmap_directory": "results/figures/permuted_ar_forgetting_heatmaps_20260524",
        },
        "notes": [
            "Existing heatmaps cover the canonical formal20 segment_len=64 RMT/Transformer comparison.",
            "This script regenerates a cleaner core heatmap grid with paper labels.",
        ],
    }
    (OUTDIR / "paper_figure_sources.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def make_contact_sheet() -> None:
    figure_paths = [
        OUTDIR / "fig1_seen_accuracy_by_stage_core.png",
        OUTDIR / "fig2_seen_accuracy_by_stage_cl_baselines.png",
        OUTDIR / "fig3_current_task_learning_by_stage.png",
        OUTDIR / "fig4_forgetting_heatmaps_core.png",
        OUTDIR / "fig5_replay_budget_curve.png",
        OUTDIR / "fig6_segment_length_seen_accuracy.png",
    ]
    thumb_w, thumb_h = 900, 558
    pad = 40
    title_h = 82
    sheet = Image.new("RGB", (pad * 3 + thumb_w * 2, title_h + pad * 4 + thumb_h * 3), "white")
    draw = ImageDraw.Draw(sheet)
    draw_text(draw, (pad, 26), "Permuted AR Paper Figure Drafts", _font(38, bold=True))
    for idx, path in enumerate(figure_paths):
        if not path.exists():
            continue
        row, col = divmod(idx, 2)
        x = pad + col * (thumb_w + pad)
        y = title_h + pad + row * (thumb_h + pad)
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            frame = Image.new("RGB", (thumb_w, thumb_h), "white")
            frame.paste(thumb, ((thumb_w - thumb.width) // 2, (thumb_h - thumb.height) // 2))
        sheet.paste(frame, (x, y))
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline="#dadce0", width=2)
    sheet.save(OUTDIR / "paper_figures_contact_sheet.png", dpi=(220, 220))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    formal64 = formal64_runs()
    formal128 = formal128_runs()
    ewc = ewc_runs()
    si = si_runs()
    make_stage_figures(formal64, ewc, si)
    make_heatmap(formal64)
    make_replay_figure()
    make_segment_figure(formal64, formal128)
    make_contact_sheet()
    write_sources(formal64, formal128, ewc, si)
    print(OUTDIR)
    for path in sorted(OUTDIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
