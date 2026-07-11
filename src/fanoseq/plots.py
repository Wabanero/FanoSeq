"""Pillow-based multipanel plots for FanoSeq output directories."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PlotMode = Literal["auto", "window", "codon"]

BACKGROUND = (255, 255, 255)
INK = (25, 35, 45)
MUTED = (110, 120, 130)
GRID = (225, 230, 235)
PANEL = (248, 250, 252)
COLORS = [
    (20, 74, 124),
    (217, 5, 85),
    (46, 125, 50),
    (239, 124, 0),
    (106, 61, 154),
    (0, 137, 123),
    (120, 86, 50),
    (70, 70, 70),
]


def plot_multipanel(
    input_dir: str | Path,
    output_path: str | Path,
    mode: PlotMode = "auto",
    sequence_id: str | None = None,
    frame: int | None = None,
    max_points: int = 500,
) -> Path:
    """Create a compact multipanel PNG summary from FanoSeq output tables."""
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"Input directory does not exist: {base}")
    if max_points <= 0:
        raise ValueError("max_points must be > 0.")

    tables = _load_available_tables(base)
    selected_mode = _select_mode(tables, mode)
    if selected_mode == "window":
        image = _plot_window_multipanel(tables, sequence_id, max_points)
    else:
        image = _plot_codon_multipanel(tables, sequence_id, frame, max_points)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def _load_available_tables(base: Path) -> dict[str, pd.DataFrame]:
    stems = [
        "window_octonions",
        "octonion_products",
        "octonion_triplets",
        "fano_interactions",
        "codon_octonions",
        "codon_transition_products",
        "codon_usage_sequence_summary",
        "codon_usage_fano_features",
        "sequence_fingerprints",
    ]
    tables: dict[str, pd.DataFrame] = {}
    for stem in stems:
        table = _read_table_if_exists(base, stem)
        if table is not None:
            tables[stem] = table
    return tables


def _read_table_if_exists(base: Path, stem: str) -> pd.DataFrame | None:
    tsv_path = base / f"{stem}.tsv"
    if tsv_path.exists():
        return pd.read_csv(tsv_path, sep="\t")
    parquet_path = base / f"{stem}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return None


def _select_mode(tables: dict[str, pd.DataFrame], mode: PlotMode) -> Literal["window", "codon"]:
    if mode not in {"auto", "window", "codon"}:
        raise ValueError("mode must be one of: auto, window, codon.")
    if mode == "window":
        if "window_octonions" not in tables:
            raise ValueError("Window mode plot requires window_octonions output.")
        return "window"
    if mode == "codon":
        if "codon_octonions" not in tables:
            raise ValueError("Codon mode plot requires codon_octonions output.")
        return "codon"
    if "window_octonions" in tables:
        return "window"
    if "codon_octonions" in tables:
        return "codon"
    raise ValueError("No window_octonions or codon_octonions table found.")


def _plot_window_multipanel(
    tables: dict[str, pd.DataFrame], sequence_id: str | None, max_points: int
) -> Image.Image:
    windows = _select_sequence(tables["window_octonions"], sequence_id)
    sequence = str(windows["sequence_id"].iloc[0])
    windows = windows.sort_values("position").head(max_points)
    products = _filter_sequence(tables.get("octonion_products"), sequence)
    triplets = _filter_sequence(tables.get("octonion_triplets"), sequence)
    fano = _filter_sequence(tables.get("fano_interactions"), sequence)
    if products is not None:
        products = products.head(max_points)
    if triplets is not None:
        triplets = triplets.head(max_points)
    if fano is not None and "mode" in fano.columns:
        fano = fano[fano["mode"] == "window"]

    image, draw, panels, fonts = _base_canvas(f"FanoSeq window multipanel: {sequence}")
    _draw_component_panel(draw, panels[0], fonts, windows, "position", "Window octonion components")
    _draw_series_panel(
        draw,
        panels[1],
        fonts,
        windows,
        "position",
        ["gc_content", "mono_entropy", "valid_fraction", "e7"],
        "Window descriptors",
    )
    _draw_score_panel(draw, panels[2], fonts, products, triplets, "Window transition scores")
    _draw_fano_heatmap(draw, panels[3], fonts, fano, "Window Fano-line contributions")
    _draw_top_fano_panel(draw, panels[4], fonts, fano)
    _draw_text_panel(
        draw,
        panels[5],
        fonts,
        "Run summary",
        _window_summary_lines(windows, products, triplets, fano),
    )
    return image


def _plot_codon_multipanel(
    tables: dict[str, pd.DataFrame],
    sequence_id: str | None,
    frame: int | None,
    max_points: int,
) -> Image.Image:
    codons = _select_sequence(tables["codon_octonions"], sequence_id)
    sequence = str(codons["sequence_id"].iloc[0])
    if frame is None:
        frame = int(codons["frame"].iloc[0])
    codons = codons[codons["frame"].astype(int) == int(frame)].sort_values("codon_index")
    codons = codons.head(max_points)
    if codons.empty:
        raise ValueError(f"No codon rows found for sequence {sequence!r}, frame {frame}.")
    products = _filter_sequence(tables.get("codon_transition_products"), sequence)
    if products is not None:
        products = products[products["frame"].astype(int) == int(frame)].head(max_points)
    fano = _filter_sequence(tables.get("fano_interactions"), sequence)
    if fano is not None:
        if "mode" in fano.columns:
            fano = fano[fano["mode"] == "codon"]
        if "frame" in fano.columns:
            fano_frame = pd.to_numeric(fano["frame"], errors="coerce")
            fano = fano[fano_frame == int(frame)]
    usage = _filter_sequence(tables.get("codon_usage_fano_features"), sequence)
    if usage is not None and "frame" in usage.columns:
        usage = usage[usage["frame"].astype(int) == int(frame)]

    image, draw, panels, fonts = _base_canvas(f"FanoSeq codon multipanel: {sequence}, frame {frame}")
    _draw_component_panel(draw, panels[0], fonts, codons, "codon_index", "Codon octonion components")
    _draw_series_panel(
        draw,
        panels[1],
        fonts,
        codons,
        "codon_index",
        ["gc1", "gc2", "gc3", "codon_associator_score"],
        "Codon descriptors",
    )
    _draw_codon_score_panel(draw, panels[2], fonts, products, codons)
    _draw_fano_heatmap(draw, panels[3], fonts, fano, "Codon Fano-line contributions")
    _draw_codon_usage_panel(draw, panels[4], fonts, usage)
    _draw_text_panel(
        draw,
        panels[5],
        fonts,
        "Run summary",
        _codon_summary_lines(codons, products, fano, usage),
    )
    return image


def _base_canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw, list[tuple[int, int, int, int]], dict[str, ImageFont.ImageFont]]:
    image = Image.new("RGB", (1600, 960), BACKGROUND)
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": ImageFont.load_default(size=26),
        "panel": ImageFont.load_default(size=18),
        "body": ImageFont.load_default(size=13),
        "small": ImageFont.load_default(size=11),
    }
    draw.text((50, 28), title, fill=INK, font=fonts["title"])
    panels: list[tuple[int, int, int, int]] = []
    left = 45
    top = 85
    width = 480
    height = 385
    gap_x = 35
    gap_y = 35
    for row in range(2):
        for column in range(3):
            x0 = left + column * (width + gap_x)
            y0 = top + row * (height + gap_y)
            panels.append((x0, y0, x0 + width, y0 + height))
    return image, draw, panels, fonts


def _select_sequence(table: pd.DataFrame, sequence_id: str | None) -> pd.DataFrame:
    if table.empty:
        raise ValueError("Selected table is empty.")
    if sequence_id is None:
        sequence_id = str(table["sequence_id"].iloc[0])
    selected = table[table["sequence_id"].astype(str) == str(sequence_id)]
    if selected.empty:
        available = ", ".join(table["sequence_id"].astype(str).drop_duplicates().head(8).tolist())
        raise ValueError(f"Sequence {sequence_id!r} not found. Available examples: {available}")
    return selected.copy()


def _filter_sequence(table: pd.DataFrame | None, sequence_id: str) -> pd.DataFrame | None:
    if table is None or table.empty or "sequence_id" not in table.columns:
        return None
    return table[table["sequence_id"].astype(str) == sequence_id].copy()


def _draw_panel_frame(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    title: str,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=8, fill=PANEL, outline=(210, 218, 226), width=1)
    draw.text((x0 + 14, y0 + 10), title, fill=INK, font=fonts["panel"])
    return x0 + 48, y0 + 48, x1 - 18, y1 - 78


def _draw_component_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    table: pd.DataFrame,
    x_column: str,
    title: str,
) -> None:
    columns = [f"e{i}" for i in range(8) if f"e{i}" in table.columns]
    _draw_line_panel(draw, rect, fonts, table, x_column, columns, title)


def _draw_series_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    table: pd.DataFrame,
    x_column: str,
    columns: list[str],
    title: str,
) -> None:
    available = [column for column in columns if column in table.columns]
    _draw_line_panel(draw, rect, fonts, table, x_column, available, title)


def _draw_score_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    products: pd.DataFrame | None,
    triplets: pd.DataFrame | None,
    title: str,
) -> None:
    rows: list[pd.DataFrame] = []
    if products is not None and not products.empty:
        rows.append(products[["position", "transition_score"]].rename(columns={"transition_score": "transition"}))
    if triplets is not None and not triplets.empty:
        rows.append(triplets[["position", "associator_score"]].rename(columns={"associator_score": "associator"}))
    if not rows:
        _draw_empty_panel(draw, rect, fonts, title, "No transition/triplet rows")
        return
    merged = rows[0]
    for row in rows[1:]:
        merged = merged.merge(row, on="position", how="outer")
    _draw_line_panel(draw, rect, fonts, merged.sort_values("position"), "position", [c for c in merged.columns if c != "position"], title)


def _draw_codon_score_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    products: pd.DataFrame | None,
    codons: pd.DataFrame,
) -> None:
    rows: list[pd.DataFrame] = []
    if products is not None and not products.empty:
        rows.append(products[["position", "transition_score"]].rename(columns={"position": "x", "transition_score": "transition"}))
    if "codon_associator_score" in codons.columns:
        rows.append(codons[["codon_index", "codon_associator_score"]].rename(columns={"codon_index": "x", "codon_associator_score": "associator"}))
    if not rows:
        _draw_empty_panel(draw, rect, fonts, "Codon transition scores", "No score rows")
        return
    merged = rows[0]
    for row in rows[1:]:
        merged = merged.merge(row, on="x", how="outer")
    _draw_line_panel(draw, rect, fonts, merged.sort_values("x"), "x", [c for c in merged.columns if c != "x"], "Codon transition scores")


def _draw_line_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    table: pd.DataFrame,
    x_column: str,
    columns: list[str],
    title: str,
) -> None:
    plot_rect = _draw_panel_frame(draw, rect, fonts, title)
    if table.empty or not columns:
        _draw_center_text(draw, plot_rect, fonts, "No plottable columns")
        return
    x_values = pd.to_numeric(table[x_column], errors="coerce").to_numpy(dtype=float)
    y_arrays = [pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float) for column in columns]
    valid_values = np.concatenate([values[np.isfinite(values)] for values in y_arrays if np.isfinite(values).any()])
    if valid_values.size == 0 or not np.isfinite(x_values).any():
        _draw_center_text(draw, plot_rect, fonts, "No finite values")
        return
    y_min = float(np.min(valid_values))
    y_max = float(np.max(valid_values))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    _draw_axes(draw, plot_rect, fonts, x_min, x_max, y_min, y_max)
    for index, column in enumerate(columns):
        y_values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        points = [
            _map_point(plot_rect, x, y, x_min, x_max, y_min, y_max)
            for x, y in zip(x_values, y_values, strict=False)
            if np.isfinite(x) and np.isfinite(y)
        ]
        if len(points) >= 2:
            draw.line(points, fill=COLORS[index % len(COLORS)], width=2)
        for point in points[:: max(1, len(points) // 30)]:
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=COLORS[index % len(COLORS)])
    _draw_legend(draw, rect, fonts, columns)


def _draw_fano_heatmap(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    fano: pd.DataFrame | None,
    title: str,
) -> None:
    plot_rect = _draw_panel_frame(draw, rect, fonts, title)
    if fano is None or fano.empty:
        _draw_center_text(draw, plot_rect, fonts, "No Fano attribution table")
        return
    pivot = fano.pivot_table(
        index="fano_line",
        columns="position",
        values="line_contribution_norm",
        aggfunc="mean",
        fill_value=0.0,
    )
    if pivot.empty:
        _draw_center_text(draw, plot_rect, fonts, "No Fano values")
        return
    x0, y0, x1, y1 = plot_rect
    label_width = 82
    heat_rect = (x0 + label_width, y0, x1, y1 - 18)
    values = pivot.to_numpy(dtype=float)
    max_value = float(np.max(values)) if values.size else 0.0
    rows, cols = values.shape
    cell_w = max(1, (heat_rect[2] - heat_rect[0]) / max(cols, 1))
    cell_h = max(1, (heat_rect[3] - heat_rect[1]) / max(rows, 1))
    for row_index, line in enumerate(pivot.index.astype(str)):
        y = heat_rect[1] + row_index * cell_h
        draw.text((x0, int(y + cell_h * 0.25)), line, fill=INK, font=fonts["small"])
        for col_index in range(cols):
            x = heat_rect[0] + col_index * cell_w
            color = _heat_color(values[row_index, col_index], max_value)
            draw.rectangle((int(x), int(y), int(x + cell_w + 1), int(y + cell_h + 1)), fill=color)
    draw.rectangle(heat_rect, outline=INK, width=1)
    draw.text((heat_rect[0], y1 - 16), "position ->", fill=MUTED, font=fonts["small"])


def _draw_top_fano_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    fano: pd.DataFrame | None,
) -> None:
    if fano is None or fano.empty:
        _draw_empty_panel(draw, rect, fonts, "Mean Fano-line contribution", "No Fano attribution table")
        return
    summary = (
        fano.groupby("fano_line", sort=False)["line_contribution_norm"]
        .mean()
        .sort_values(ascending=False)
    )
    table = pd.DataFrame({"rank": np.arange(len(summary)), "mean_norm": summary.to_numpy(dtype=float)})
    _draw_line_panel(draw, rect, fonts, table, "rank", ["mean_norm"], "Mean Fano-line contribution")
    x0, y0, _, _ = rect
    for index, label in enumerate(list(summary.index.astype(str))[:5]):
        draw.text((x0 + 270, y0 + 46 + index * 16), f"{index + 1}. {label}", fill=INK, font=fonts["small"])


def _draw_codon_usage_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    usage: pd.DataFrame | None,
) -> None:
    if usage is None or usage.empty:
        _draw_empty_panel(draw, rect, fonts, "Top codon frequencies", "No codon usage table")
        return
    top = usage.sort_values("frequency", ascending=False).head(12).reset_index(drop=True)
    table = pd.DataFrame({"rank": top.index.to_numpy(dtype=float), "frequency": top["frequency"].astype(float)})
    _draw_line_panel(draw, rect, fonts, table, "rank", ["frequency"], "Top codon frequencies")
    x0, y0, _, _ = rect
    for index, row in top.head(6).iterrows():
        label = f"{int(index) + 1}. {row['codon']}/{row['amino_acid']}"
        draw.text((x0 + 300, y0 + 46 + int(index) * 16), label, fill=INK, font=fonts["small"])


def _window_summary_lines(
    windows: pd.DataFrame,
    products: pd.DataFrame | None,
    triplets: pd.DataFrame | None,
    fano: pd.DataFrame | None,
) -> list[str]:
    lines = [
        f"sequence_id: {windows['sequence_id'].iloc[0]}",
        f"windows: {len(windows)}",
        f"position range: {int(windows['position'].min())}-{int(windows['position'].max())}",
    ]
    if "gc_content" in windows.columns and not windows["gc_content"].isna().all():
        lines.append(f"mean GC: {float(windows['gc_content'].mean()):.3f}")
    if products is not None and not products.empty:
        lines.append(f"max transition: {float(products['transition_score'].max()):.3f}")
    if triplets is not None and not triplets.empty:
        lines.append(f"max associator: {float(triplets['associator_score'].max()):.3f}")
    if fano is not None and not fano.empty:
        top_line = fano.groupby("fano_line")["line_contribution_norm"].mean().idxmax()
        lines.append(f"top Fano line: {top_line}")
    return lines


def _codon_summary_lines(
    codons: pd.DataFrame,
    products: pd.DataFrame | None,
    fano: pd.DataFrame | None,
    usage: pd.DataFrame | None,
) -> list[str]:
    lines = [
        f"sequence_id: {codons['sequence_id'].iloc[0]}",
        f"frame: {int(codons['frame'].iloc[0])}",
        f"codons: {len(codons)}",
    ]
    if "gc3" in codons.columns:
        lines.append(f"mean GC3: {float(codons['gc3'].mean()):.3f}")
    if "codon_associator_score" in codons.columns:
        lines.append(f"max codon associator: {float(codons['codon_associator_score'].max()):.3f}")
    if products is not None and not products.empty:
        lines.append(f"max transition: {float(products['transition_score'].max()):.3f}")
    if fano is not None and not fano.empty:
        top_line = fano.groupby("fano_line")["line_contribution_norm"].mean().idxmax()
        lines.append(f"top Fano line: {top_line}")
    if usage is not None and not usage.empty:
        top_codon = usage.sort_values("frequency", ascending=False).iloc[0]
        lines.append(f"top codon: {top_codon['codon']} ({float(top_codon['frequency']):.3f})")
    return lines


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline=INK, width=1)
    for fraction in (0.25, 0.5, 0.75):
        x = int(x0 + fraction * (x1 - x0))
        y = int(y0 + fraction * (y1 - y0))
        draw.line((x, y0, x, y1), fill=GRID, width=1)
        draw.line((x0, y, x1, y), fill=GRID, width=1)
    draw.text((x0, y1 + 4), f"{x_min:.0f}", fill=MUTED, font=fonts["small"])
    draw.text((x1 - 35, y1 + 4), f"{x_max:.0f}", fill=MUTED, font=fonts["small"])
    draw.text((x0 - 42, y1 - 8), f"{y_min:.2g}", fill=MUTED, font=fonts["small"])
    draw.text((x0 - 42, y0 - 4), f"{y_max:.2g}", fill=MUTED, font=fonts["small"])


def _map_point(
    rect: tuple[int, int, int, int],
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[int, int]:
    x0, y0, x1, y1 = rect
    px = x0 + (x - x_min) / (x_max - x_min) * (x1 - x0)
    py = y1 - (y - y_min) / (y_max - y_min) * (y1 - y0)
    return int(px), int(py)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    labels: list[str],
) -> None:
    x0, _, x1, y1 = rect
    start_x = x0 + 14
    y = y1 - 42
    x = start_x
    for index, label in enumerate(labels[:8]):
        color = COLORS[index % len(COLORS)]
        text_width = _text_width(fonts["small"], label)
        segment_width = 28 + text_width + 22
        if x + segment_width > x1 - 14 and x > start_x:
            x = start_x
            y += 16
        draw.line((x, y + 6, x + 18, y + 6), fill=color, width=3)
        draw.text((x + 22, y), label, fill=INK, font=fonts["small"])
        x += segment_width


def _text_width(font: ImageFont.ImageFont, text: str) -> int:
    try:
        return int(font.getlength(text))
    except AttributeError:
        return int(font.getbbox(text)[2])


def _draw_text_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    title: str,
    lines: list[str],
) -> None:
    x0, y0, _, _ = rect
    _draw_panel_frame(draw, rect, fonts, title)
    y = y0 + 54
    for line in lines:
        draw.text((x0 + 22, y), line, fill=INK, font=fonts["body"])
        y += 24


def _draw_empty_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    title: str,
    message: str,
) -> None:
    plot_rect = _draw_panel_frame(draw, rect, fonts, title)
    _draw_center_text(draw, plot_rect, fonts, message)


def _draw_center_text(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    message: str,
) -> None:
    x0, y0, x1, y1 = rect
    draw.text(((x0 + x1) // 2 - 70, (y0 + y1) // 2), message, fill=MUTED, font=fonts["body"])


def _heat_color(value: float, max_value: float) -> tuple[int, int, int]:
    if max_value <= 0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, value / max_value))
    if t < 0.5:
        local = t / 0.5
        return (
            int(35 + local * 20),
            int(55 + local * 150),
            int(130 - local * 50),
        )
    local = (t - 0.5) / 0.5
    return (
        int(55 + local * 200),
        int(205 + local * 20),
        int(80 - local * 55),
    )
