"""Create paper-oriented figures for formal permuted AR results.

The script is intentionally dependency-light and uses PIL instead of
matplotlib, because the project venv does not currently include matplotlib.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import Image, ImageDraw, ImageFont


OUTDIR = Path("results/figures/permuted_ar_paper_20260524")

MODEL_LABELS = {
    "attention": "Transformer",
    "base_rmt_nmem16": "Base RMT",
    "fmrmt_lr0p005_slow2": "FMRMT lr=.005",
    "fmrmt_fast0_slow2": "FMRMT no-fast",
    "attention_online_ewc_lam100000": "EWC",
    "attention_si_lam300": "SI",
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
    "fmrmt_stored": "#B279A2",
}


def _font(size: int, bold: bool = False):
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
            continue
    return ImageFont.load_default()


FONT_11 = _font(11)
FONT_12 = _font(12)
FONT_13 = _font(13)
FONT_14 = _font(14)
FONT_16_BOLD = _font(16, bold=True)
FONT_20_BOLD = _font(20, bold=True)
FONT_24_BOLD = _font(24, bold=True)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _metric(run: dict[str, Any], key: str) -> float | None:
    value = run.get("summary", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def collect_runs(pattern: str, *, segment_len: int | None = None) -> list[dict[str, Any]]:
    runs = []
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
            copied["_setup"] = setup
            runs.append(copied)
    return runs


def latest_by_model_seed(runs: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    selected = {}
    for run in runs:
        key = (run["model_key"], int(run["seed"]))
        selected[key] = run
    return selected


def stat_for_model(
    selected: dict[tuple[str, int], dict[str, Any]],
    model_key: str,
    metric: str,
):
    values = [
        _metric(run, metric)
        for (candidate, _), run in selected.items()
        if candidate == model_key
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "values": values,
        "n": len(values),
    }


def collect_model_stats(
    selected: dict[tuple[str, int], dict[str, Any]],
    model_keys: list[str],
):
    metrics = [
        "continual/seen_avg_accuracy",
        "continual/avg_learning_accuracy",
        "continual/plasticity",
        "continual/avg_bwt",
        "continual/avg_forgetting_from_learning",
        "continual/cumulative_wall_seconds",
    ]
    return {
        model: {
            metric: stat_for_model(selected, model, metric)
            for metric in metrics
        }
        for model in model_keys
    }


def paper_canvas(width=1600, height=950):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    return image, draw


def text(draw, xy, value, font=FONT_12, fill="#202124", anchor=None):
    draw.text(xy, str(value), font=font, fill=fill, anchor=anchor)


def draw_title(draw, title: str, subtitle: str | None = None):
    text(draw, (60, 42), title, FONT_24_BOLD)
    if subtitle:
        text(draw, (60, 75), subtitle, FONT_13, fill="#5f6368")


def axis_y(draw, left, top, height, ymin, ymax, ticks, label: str):
    draw.line((left, top, left, top + height), fill="#202124", width=2)
    draw.line((left, top + height, left + 1, top + height), fill="#202124", width=2)
    for tick in ticks:
        frac = (tick - ymin) / (ymax - ymin)
        y = top + height - frac * height
        draw.line((left - 6, y, left, y), fill="#202124", width=1)
        draw.line((left, y, left + 1330, y), fill="#e8eaed", width=1)
        text(draw, (left - 12, y), f"{tick:g}", FONT_11, fill="#5f6368", anchor="rm")
    text(draw, (left - 58, top + height / 2), label, FONT_12, fill="#3c4043", anchor="mm")


def value_y(value: float, top: int, height: int, ymin: float, ymax: float) -> float:
    return top + height - ((value - ymin) / (ymax - ymin)) * height


def draw_grouped_bars(
    image: Image.Image,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    groups: list[str],
    series: list[dict[str, Any]],
    ymin: float,
    ymax: float,
    ticks: list[float],
    y_label: str,
):
    draw = ImageDraw.Draw(image)
    draw_title(draw, title, subtitle)
    left, top, width, height = 130, 135, 1330, 600
    axis_y(draw, left, top, height, ymin, ymax, ticks, y_label)
    draw.line((left, top + height, left + width, top + height), fill="#202124", width=2)

    group_w = width / len(groups)
    bar_w = min(46, group_w / (len(series) + 1.6))
    for group_idx, group in enumerate(groups):
        center = left + group_w * (group_idx + 0.5)
        for series_idx, item in enumerate(series):
            value = item["values"][group_idx]
            err = item.get("errors", [0.0] * len(groups))[group_idx]
            x0 = center - (len(series) * bar_w) / 2 + series_idx * bar_w
            x1 = x0 + bar_w * 0.82
            y0 = value_y(value, top, height, ymin, ymax)
            y_base = value_y(0.0, top, height, ymin, ymax)
            draw.rectangle(
                (x0, min(y0, y_base), x1, max(y0, y_base)),
                fill=item["color"],
            )
            if err:
                y_hi = value_y(min(ymax, value + err), top, height, ymin, ymax)
                y_lo = value_y(max(ymin, value - err), top, height, ymin, ymax)
                x_mid = (x0 + x1) / 2
                draw.line((x_mid, y_hi, x_mid, y_lo), fill="#202124", width=2)
                draw.line((x_mid - 7, y_hi, x_mid + 7, y_hi), fill="#202124", width=2)
                draw.line((x_mid - 7, y_lo, x_mid + 7, y_lo), fill="#202124", width=2)
        label = group.replace(" ", "\n")
        text(draw, (center, top + height + 36), label, FONT_12, fill="#3c4043", anchor="mm")

    legend_x = left
    legend_y = 835
    for idx, item in enumerate(series):
        x = legend_x + idx * 230
        draw.rectangle((x, legend_y, x + 22, legend_y + 14), fill=item["color"])
        text(draw, (x + 30, legend_y + 9), item["label"], FONT_13, fill="#3c4043", anchor="lm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def line_points(xs, ys, left, top, width, height, xmin, xmax, ymin, ymax):
    points = []
    for x, y in zip(xs, ys):
        px = left + ((x - xmin) / (xmax - xmin)) * width
        py = value_y(y, top, height, ymin, ymax)
        points.append((px, py))
    return points


def draw_line_chart(
    image: Image.Image,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xticks: list[float],
    yticks: list[float],
    series: list[dict[str, Any]],
):
    draw = ImageDraw.Draw(image)
    draw_title(draw, title, subtitle)
    left, top, width, height = 130, 135, 1330, 600
    axis_y(draw, left, top, height, ymin, ymax, yticks, y_label)
    draw.line((left, top + height, left + width, top + height), fill="#202124", width=2)
    for tick in xticks:
        x = left + ((tick - xmin) / (xmax - xmin)) * width
        draw.line((x, top + height, x, top + height + 6), fill="#202124", width=1)
        text(draw, (x, top + height + 26), f"{tick:g}", FONT_11, fill="#5f6368", anchor="mm")
    text(draw, (left + width / 2, top + height + 66), x_label, FONT_13, fill="#3c4043", anchor="mm")

    for item in series:
        points = line_points(
            item["x"],
            item["y"],
            left,
            top,
            width,
            height,
            xmin,
            xmax,
            ymin,
            ymax,
        )
        if len(points) > 1:
            draw.line(points, fill=item["color"], width=4, joint="curve")
        for point in points:
            x, y = point
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=item["color"], outline="white", width=2)
        for x_val, y_val, err in zip(item["x"], item["y"], item.get("err", [0.0] * len(item["x"]))):
            if not err:
                continue
            x = left + ((x_val - xmin) / (xmax - xmin)) * width
            y_hi = value_y(min(ymax, y_val + err), top, height, ymin, ymax)
            y_lo = value_y(max(ymin, y_val - err), top, height, ymin, ymax)
            draw.line((x, y_hi, x, y_lo), fill=item["color"], width=2)
            draw.line((x - 6, y_hi, x + 6, y_hi), fill=item["color"], width=2)
            draw.line((x - 6, y_lo, x + 6, y_lo), fill=item["color"], width=2)

    legend_x = left
    legend_y = 835
    for idx, item in enumerate(series):
        x = legend_x + idx * 300
        draw.line((x, legend_y + 7, x + 28, legend_y + 7), fill=item["color"], width=4)
        draw.ellipse((x + 9, legend_y - 1, x + 19, legend_y + 9), fill=item["color"], outline="white")
        text(draw, (x + 38, legend_y + 7), item["label"], FONT_13, fill="#3c4043", anchor="lm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main_formal_selected():
    runs = collect_runs(
        "results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_*.json",
        segment_len=64,
    )
    return latest_by_model_seed(runs)


def segment128_selected():
    runs = collect_runs(
        "results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_segment_len128_formal20_permuted_20260524_*.json",
        segment_len=128,
    )
    return latest_by_model_seed(runs)


def ewc_selected():
    runs = collect_runs("results/class_incremental_ar_permuted_online_ewc_formal20_permuted_20260524_*.json")
    return latest_by_model_seed(runs)


def si_selected():
    runs = collect_runs("results/class_incremental_ar_permuted_si_formal20_permuted_20260524_*.json")
    return latest_by_model_seed(runs)


def replay_stats(source: str, model_key: str | None = None):
    if source == "reservoir":
        pattern = (
            "results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_"
            "fixed_total_reservoir_replay*_buffer*_formal20_permuted_20260524_*.json"
        )
    elif source == "stored":
        pattern = (
            "results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_"
            "fixed_total_stored_replay*_formal20_permuted_20260524_0*.json"
        )
    else:
        raise ValueError(source)
    grouped: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
    for path in sorted(glob.glob(pattern)):
        data = load_json(path)
        if data.get("setup", {}).get("tasks") != ["formal20_permuted"]:
            continue
        for run in data.get("runs", []):
            if model_key is not None and run["model_key"] != model_key:
                continue
            budget = int(run["replay_examples_per_old_stage"])
            grouped.setdefault(budget, {})[(run["model_key"], int(run["seed"]))] = run
    out = {}
    for budget, selected in grouped.items():
        values_by_model: dict[str, list[float]] = {}
        for (candidate_model, _), run in selected.items():
            value = _metric(run, "continual/seen_avg_accuracy")
            if value is not None:
                values_by_model.setdefault(candidate_model, []).append(value)
        out[budget] = {
            model: {
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
            for model, values in values_by_model.items()
        }
    return out


def curve_for_model(selected: dict[tuple[str, int], dict[str, Any]], model_key: str, stages: list[int]):
    by_epoch: dict[int, list[float]] = {}
    for (candidate, _), run in selected.items():
        if candidate != model_key:
            continue
        curves = run.get("current_stage_epoch_curves", {})
        for stage in stages:
            for entry in curves.get(str(stage), []):
                epoch = entry.get("stage_epoch")
                acc = entry.get("accuracy")
                if epoch is not None and acc is not None:
                    by_epoch.setdefault(int(epoch), []).append(float(acc))
    xs, ys, errs = [], [], []
    for epoch, values in sorted(by_epoch.items()):
        xs.append(epoch)
        ys.append(mean(values))
        errs.append(pstdev(values) if len(values) > 1 else 0.0)
    return xs, ys, errs


def make_main_comparison(formal):
    model_keys = ["attention", "base_rmt_nmem16", "fmrmt_fast0_slow2", "fmrmt_lr0p005_slow2"]
    stats = collect_model_stats(formal, model_keys)
    groups = [MODEL_LABELS[key] for key in model_keys]
    image, _ = paper_canvas()
    draw_grouped_bars(
        image,
        OUTDIR / "fig1_formal20_main_retention.png",
        title="Formal20 Permuted AR: Main Model Comparison",
        subtitle="Stage-local continual evaluation, mean +/- std over seeds 123/456/789",
        groups=groups,
        series=[
            {
                "label": "Final seen accuracy",
                "color": "#4C78A8",
                "values": [stats[key]["continual/seen_avg_accuracy"]["mean"] for key in model_keys],
                "errors": [stats[key]["continual/seen_avg_accuracy"]["std"] for key in model_keys],
            },
            {
                "label": "Avg learning accuracy",
                "color": "#54A24B",
                "values": [stats[key]["continual/avg_learning_accuracy"]["mean"] for key in model_keys],
                "errors": [stats[key]["continual/avg_learning_accuracy"]["std"] for key in model_keys],
            },
        ],
        ymin=0,
        ymax=1.05,
        ticks=[0, 0.25, 0.5, 0.75, 1.0],
        y_label="Accuracy",
    )
    image, _ = paper_canvas()
    draw_grouped_bars(
        image,
        OUTDIR / "fig2_formal20_forgetting_bwt.png",
        title="Formal20 Permuted AR: Forgetting and Backward Transfer",
        subtitle="Lower forgetting is better; BWT closer to zero is better",
        groups=groups,
        series=[
            {
                "label": "Forgetting from learning",
                "color": "#E45756",
                "values": [stats[key]["continual/avg_forgetting_from_learning"]["mean"] for key in model_keys],
                "errors": [stats[key]["continual/avg_forgetting_from_learning"]["std"] for key in model_keys],
            },
            {
                "label": "-Avg BWT",
                "color": "#F58518",
                "values": [-stats[key]["continual/avg_bwt"]["mean"] for key in model_keys],
                "errors": [stats[key]["continual/avg_bwt"]["std"] for key in model_keys],
            },
        ],
        ymin=0,
        ymax=1.0,
        ticks=[0, 0.25, 0.5, 0.75, 1.0],
        y_label="Magnitude",
    )


def make_cl_baselines(formal, ewc, si):
    selected = {}
    for key in [("attention", 123), ("attention", 456), ("attention", 789)]:
        selected[key] = formal[key]
    for key in [("base_rmt_nmem16", 123), ("base_rmt_nmem16", 456), ("base_rmt_nmem16", 789)]:
        selected[key] = formal[key]
    for key in [
        ("fmrmt_lr0p005_slow2", 123),
        ("fmrmt_lr0p005_slow2", 456),
        ("fmrmt_lr0p005_slow2", 789),
    ]:
        selected[key] = formal[key]
    for seed in [123, 456, 789]:
        selected[("attention_online_ewc_lam100000", seed)] = ewc[
            ("attention_online_ewc_lam100000", seed)
        ]
        selected[("attention_si_lam300", seed)] = si[("attention_si_lam300", seed)]
    model_keys = [
        "attention",
        "attention_online_ewc_lam100000",
        "attention_si_lam300",
        "base_rmt_nmem16",
        "fmrmt_lr0p005_slow2",
    ]
    stats = collect_model_stats(selected, model_keys)
    groups = [MODEL_LABELS[key] for key in model_keys]
    image, _ = paper_canvas()
    draw_grouped_bars(
        image,
        OUTDIR / "fig3_no_replay_cl_baselines.png",
        title="No-Replay Continual-Learning Baselines",
        subtitle="SI is strong but trades off late-stage plasticity; FMRMT keeps learning accuracy at 1.0",
        groups=groups,
        series=[
            {
                "label": "Final seen accuracy",
                "color": "#4C78A8",
                "values": [stats[key]["continual/seen_avg_accuracy"]["mean"] for key in model_keys],
                "errors": [stats[key]["continual/seen_avg_accuracy"]["std"] for key in model_keys],
            },
            {
                "label": "Final plasticity",
                "color": "#72B7B2",
                "values": [stats[key]["continual/plasticity"]["mean"] for key in model_keys],
                "errors": [stats[key]["continual/plasticity"]["std"] for key in model_keys],
            },
        ],
        ymin=0,
        ymax=1.05,
        ticks=[0, 0.25, 0.5, 0.75, 1.0],
        y_label="Accuracy",
    )


def make_replay_curve():
    reservoir = replay_stats("reservoir", "attention")
    stored = replay_stats("stored")
    budgets = [1, 2, 4, 8, 16]
    series = [
        {
            "label": "Transformer reservoir",
            "color": COLORS["reservoir"],
            "x": budgets,
            "y": [reservoir[b]["attention"]["mean"] for b in budgets],
            "err": [reservoir[b]["attention"]["std"] for b in budgets],
        },
        {
            "label": "Transformer balanced stored",
            "color": COLORS["stored"],
            "x": budgets,
            "y": [stored[b]["attention"]["mean"] for b in budgets],
            "err": [stored[b]["attention"]["std"] for b in budgets],
        },
        {
            "label": "Base RMT balanced stored",
            "color": COLORS["base_stored"],
            "x": budgets,
            "y": [stored[b]["base_rmt_nmem16"]["mean"] for b in budgets],
            "err": [stored[b]["base_rmt_nmem16"]["std"] for b in budgets],
        },
        {
            "label": "FMRMT balanced stored",
            "color": COLORS["fmrmt_stored"],
            "x": budgets,
            "y": [stored[b]["fmrmt_lr0p005_slow2"]["mean"] for b in budgets],
            "err": [stored[b]["fmrmt_lr0p005_slow2"]["std"] for b in budgets],
        },
    ]
    image, _ = paper_canvas()
    draw_line_chart(
        image,
        OUTDIR / "fig4_replay_buffer_curve.png",
        title="Formal20 Permuted AR: Replay Buffer Curve",
        subtitle="Final seen accuracy vs old examples per prior-stage equivalent",
        x_label="Replay examples per old stage equivalent",
        y_label="Final seen accuracy",
        xmin=1,
        xmax=16,
        ymin=0,
        ymax=1.05,
        xticks=budgets,
        yticks=[0, 0.25, 0.5, 0.75, 1.0],
        series=series,
    )


def make_segment_control(formal64, seg128):
    model_keys = ["attention", "base_rmt_nmem16", "fmrmt_fast0_slow2", "fmrmt_lr0p005_slow2"]
    stats64 = collect_model_stats(formal64, model_keys)
    stats128 = collect_model_stats(seg128, model_keys)
    groups = [MODEL_LABELS[key] for key in model_keys]
    image, _ = paper_canvas()
    draw_grouped_bars(
        image,
        OUTDIR / "fig5_segment_len_control.png",
        title="Segment-Length Control",
        subtitle="segment_len=128 removes cross-segment recurrence but preserves memory-token architecture",
        groups=groups,
        series=[
            {
                "label": "segment_len=64",
                "color": "#4C78A8",
                "values": [stats64[key]["continual/seen_avg_accuracy"]["mean"] for key in model_keys],
                "errors": [stats64[key]["continual/seen_avg_accuracy"]["std"] for key in model_keys],
            },
            {
                "label": "segment_len=128",
                "color": "#F58518",
                "values": [stats128[key]["continual/seen_avg_accuracy"]["mean"] for key in model_keys],
                "errors": [stats128[key]["continual/seen_avg_accuracy"]["std"] for key in model_keys],
            },
        ],
        ymin=0,
        ymax=1.05,
        ticks=[0, 0.25, 0.5, 0.75, 1.0],
        y_label="Final seen accuracy",
    )


def make_late_learning(formal, si):
    stages = [15, 16, 17, 18, 19]
    series = []
    for model_key, label, color, selected in [
        ("attention", "Transformer", "#4C78A8", formal),
        ("base_rmt_nmem16", "Base RMT", "#F58518", formal),
        ("fmrmt_lr0p005_slow2", "FMRMT lr=.005", "#54A24B", formal),
        ("attention_si_lam300", "SI lambda=300", "#72B7B2", si),
    ]:
        xs, ys, errs = curve_for_model(selected, model_key, stages)
        series.append({"label": label, "color": color, "x": xs, "y": ys, "err": errs})
    image, _ = paper_canvas()
    draw_line_chart(
        image,
        OUTDIR / "fig6_late_stage_learning_curves.png",
        title="Late-Stage Current-Task Learning",
        subtitle="Mean current-stage accuracy across stages 15-19 and seeds 123/456/789",
        x_label="Epoch within stage",
        y_label="Current-stage accuracy",
        xmin=1,
        xmax=8,
        ymin=0,
        ymax=1.05,
        xticks=[1, 2, 3, 4, 5, 6, 7, 8],
        yticks=[0, 0.25, 0.5, 0.75, 1.0],
        series=series,
    )


def write_summary_json(formal, seg128, ewc, si):
    summary = {
        "created_figures": sorted(str(path) for path in OUTDIR.glob("*.png")),
        "sources": {
            "formal20_segment64": sorted({run["_source"] for run in formal.values()}),
            "formal20_segment128": sorted({run["_source"] for run in seg128.values()}),
            "ewc": sorted({run["_source"] for run in ewc.values()}),
            "si": sorted({run["_source"] for run in si.values()}),
        },
    }
    (OUTDIR / "paper_figure_sources.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    formal = main_formal_selected()
    seg128 = segment128_selected()
    ewc = ewc_selected()
    si = si_selected()
    make_main_comparison(formal)
    make_cl_baselines(formal, ewc, si)
    make_replay_curve()
    make_segment_control(formal, seg128)
    make_late_learning(formal, si)
    write_summary_json(formal, seg128, ewc, si)
    print(OUTDIR)
    for path in sorted(OUTDIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
