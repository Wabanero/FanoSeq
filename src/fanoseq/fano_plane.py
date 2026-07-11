"""Explicit Fano-plane objects, tables, plots, and feature summaries."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from fanoseq.axis_schemes import get_axis_scheme
from fanoseq.octonion import FANO_LINES


@dataclass(frozen=True)
class FanoPoint:
    """One imaginary-axis point in the Fano plane."""

    axis: int
    symbol: str
    label: str


@dataclass(frozen=True)
class FanoLine:
    """One oriented line in the Fano plane."""

    index: int
    axes: tuple[int, int, int]
    label: str
    interpretation: str

    @property
    def key(self) -> str:
        """Return the compact table key for this line."""
        return line_key(self.axes)


@dataclass(frozen=True)
class FanoPlane:
    """The project Fano plane as an explicit computational object."""

    scheme_id: str | None = None

    @property
    def points(self) -> tuple[FanoPoint, ...]:
        labels = _axis_labels(self.scheme_id)
        return tuple(FanoPoint(axis, f"e{axis}", labels[axis]) for axis in range(1, 8))

    @property
    def lines(self) -> tuple[FanoLine, ...]:
        labels, line_labels, interpretations = _line_metadata(self.scheme_id)
        rows: list[FanoLine] = []
        for index, axes in enumerate(FANO_LINES):
            fallback = " / ".join(labels[axis] for axis in axes)
            rows.append(
                FanoLine(
                    index=index,
                    axes=axes,
                    label=line_labels.get(axes, fallback),
                    interpretation=interpretations.get(
                        axes,
                        "Fano-plane line under the project octonion convention.",
                    ),
                )
            )
        return tuple(rows)

    def point_table(self) -> pd.DataFrame:
        """Return Fano-plane points with axis labels."""
        line_membership = _axis_line_membership()
        rows = []
        for point in self.points:
            rows.append(
                {
                    "axis_scheme_id": self.scheme_id or "basis",
                    "axis": point.axis,
                    "symbol": point.symbol,
                    "axis_label": point.label,
                    "incident_fano_lines": ",".join(line_membership[point.axis]),
                    "degree": len(line_membership[point.axis]),
                }
            )
        return pd.DataFrame(rows)

    def line_table(self) -> pd.DataFrame:
        """Return oriented Fano-plane lines."""
        labels = _axis_labels(self.scheme_id)
        rows = []
        for line in self.lines:
            a, b, c = line.axes
            rows.append(
                {
                    "axis_scheme_id": self.scheme_id or "basis",
                    "line_index": line.index,
                    "fano_line": line.key,
                    "axis_a": a,
                    "axis_b": b,
                    "axis_c": c,
                    "axis_a_label": labels[a],
                    "axis_b_label": labels[b],
                    "axis_c_label": labels[c],
                    "line_label": line.label,
                    "interpretation": line.interpretation,
                    "orientation_rule": f"e{a}*e{b}=+e{c}; e{b}*e{c}=+e{a}; e{c}*e{a}=+e{b}",
                }
            )
        return pd.DataFrame(rows)

    def incidence_table(self) -> pd.DataFrame:
        """Return point-line incidence rows."""
        labels = _axis_labels(self.scheme_id)
        rows = []
        for line in self.lines:
            axes = line.axes
            for position, axis in enumerate(axes):
                rows.append(
                    {
                        "axis_scheme_id": self.scheme_id or "basis",
                        "line_index": line.index,
                        "fano_line": line.key,
                        "line_label": line.label,
                        "axis": axis,
                        "symbol": f"e{axis}",
                        "axis_label": labels[axis],
                        "position_in_oriented_line": position,
                        "oriented_next_axis": axes[(position + 1) % 3],
                        "oriented_previous_axis": axes[(position - 1) % 3],
                    }
                )
        return pd.DataFrame(rows)

    def pair_table(self) -> pd.DataFrame:
        """Return ordered imaginary-axis products induced by Fano lines."""
        labels = _axis_labels(self.scheme_id)
        line_lookup = {frozenset(line.axes): line for line in self.lines}
        rows = []
        for line in self.lines:
            a, b, c = line.axes
            for left, right, output in ((a, b, c), (b, c, a), (c, a, b)):
                rows.append(
                    _pair_row(
                        self.scheme_id,
                        labels,
                        line,
                        left,
                        right,
                        output,
                        +1,
                    )
                )
                rows.append(
                    _pair_row(
                        self.scheme_id,
                        labels,
                        line,
                        right,
                        left,
                        output,
                        -1,
                    )
                )

        for left in range(1, 8):
            for right in range(left + 1, 8):
                line = line_lookup[frozenset((left, right, _third_axis(left, right)))]
                rows.append(
                    {
                        "axis_scheme_id": self.scheme_id or "basis",
                        "left_axis": left,
                        "right_axis": right,
                        "left_axis_label": labels[left],
                        "right_axis_label": labels[right],
                        "unordered_pair": f"({left},{right})",
                        "output_axis": _third_axis(left, right),
                        "output_axis_label": labels[_third_axis(left, right)],
                        "product_sign": np.nan,
                        "product_rule": "unique Fano line through unordered pair",
                        "fano_line": line.key,
                        "line_label": line.label,
                        "is_ordered_product": False,
                    }
                )
        return pd.DataFrame(rows)

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return all explicit Fano-plane tables."""
        return {
            "fano_plane_points": self.point_table(),
            "fano_plane_lines": self.line_table(),
            "fano_plane_incidence": self.incidence_table(),
            "fano_plane_pairs": self.pair_table(),
        }


def line_key(axes: tuple[int, int, int]) -> str:
    """Return the stable string key for a Fano line."""
    return f"({axes[0]},{axes[1]},{axes[2]})"


FANO_LINE_KEYS = tuple(line_key(line) for line in FANO_LINES)


def fano_plane_tables(axis_scheme_id: str | None = None) -> dict[str, pd.DataFrame]:
    """Return explicit Fano-plane point, line, incidence, and pair tables."""
    return FanoPlane(axis_scheme_id).tables()


def build_fano_line_features(interactions: pd.DataFrame) -> pd.DataFrame:
    """Summarize Fano-line attribution rows into a feature family."""
    if interactions.empty:
        return _empty_feature_table()
    required = {"sequence_id", "fano_line", "line_contribution_norm"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"fano_interactions is missing columns: {', '.join(sorted(missing))}.")

    table = interactions.copy()
    table["line_contribution_norm"] = pd.to_numeric(
        table["line_contribution_norm"],
        errors="coerce",
    ).fillna(0.0)
    group_columns = _feature_group_columns(table)
    rows: list[dict[str, object]] = []
    for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row: dict[str, object] = dict(zip(group_columns, key_values))
        profile = _line_profile(group)
        line_sums = profile["sum"]
        total = float(line_sums.sum())
        shares = line_sums / total if total > 0 else line_sums
        entropy = _normalized_entropy(shares.to_numpy(dtype=float))

        row["n_fano_rows"] = int(len(group))
        row["total_line_contribution_norm"] = total
        row["fano_line_entropy"] = entropy
        row["fano_line_evenness"] = entropy
        row["fano_line_profile_l2"] = float(np.linalg.norm(shares.to_numpy(dtype=float)))

        dominant_line = str(shares.idxmax()) if total > 0 else "NA"
        row["dominant_fano_line"] = dominant_line
        row["dominant_fano_line_share"] = float(shares.max()) if total > 0 else 0.0
        row["dominant_fano_line_label"] = _dominant_line_label(group, dominant_line)

        for key in FANO_LINE_KEYS:
            prefix = _feature_prefix(key)
            row[f"{prefix}_sum"] = float(line_sums[key])
            row[f"{prefix}_mean"] = float(profile.loc[key, "mean"])
            row[f"{prefix}_max"] = float(profile.loc[key, "max"])
            row[f"{prefix}_std"] = float(profile.loc[key, "std"])
            row[f"{prefix}_share"] = float(shares[key])

        axis_shares = _axis_incident_shares(shares)
        for axis in range(1, 8):
            row[f"axis_e{axis}_incident_share"] = axis_shares[axis]
        dominant_axis = max(axis_shares, key=axis_shares.get)
        row["dominant_axis"] = int(dominant_axis)
        row["dominant_axis_incident_share"] = float(axis_shares[dominant_axis])
        rows.append(row)
    return pd.DataFrame(rows)


def build_fano_line_stability(
    interactions: pd.DataFrame,
    n_bootstrap: int = 100,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap stability of dominant Fano-line profiles."""
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0.")
    if interactions.empty:
        return pd.DataFrame(
            columns=[
                "sequence_id",
                "mode",
                "seq_type",
                "axis_scheme_id",
                "frame",
                "dominant_fano_line",
                "dominant_line_stability",
                "mean_profile_cosine_to_full",
                "min_profile_cosine_to_full",
                "n_bootstrap",
                "n_fano_rows",
            ]
        )

    table = interactions.copy()
    table["line_contribution_norm"] = pd.to_numeric(
        table["line_contribution_norm"],
        errors="coerce",
    ).fillna(0.0)
    group_columns = _feature_group_columns(table)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_meta = dict(zip(group_columns, key_values))
        full_profile = _line_share_vector(group)
        dominant = FANO_LINE_KEYS[int(np.argmax(full_profile))] if full_profile.sum() > 0 else "NA"
        dominant_hits = 0
        cosines: list[float] = []
        indices = np.arange(len(group))
        for _ in range(n_bootstrap):
            sample = group.iloc[rng.choice(indices, size=len(group), replace=True)]
            sample_profile = _line_share_vector(sample)
            sample_dominant = FANO_LINE_KEYS[int(np.argmax(sample_profile))]
            dominant_hits += int(sample_dominant == dominant)
            cosines.append(_cosine_similarity(full_profile, sample_profile))
        row = dict(group_meta)
        row.update(
            {
                "dominant_fano_line": dominant,
                "dominant_fano_line_label": _dominant_line_label(group, dominant),
                "dominant_line_stability": dominant_hits / n_bootstrap,
                "mean_profile_cosine_to_full": float(np.mean(cosines)) if cosines else 0.0,
                "min_profile_cosine_to_full": float(np.min(cosines)) if cosines else 0.0,
                "n_bootstrap": int(n_bootstrap),
                "n_fano_rows": int(len(group)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fano_plane(
    output_path: str | Path,
    axis_scheme_id: str | None = None,
    size: int = 1200,
) -> Path:
    """Draw a PNG Fano-plane diagram for the project convention."""
    if size < 600:
        raise ValueError("size must be at least 600 pixels.")
    plane = FanoPlane(axis_scheme_id)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    fonts = _plot_fonts(size)
    margin = int(size * 0.08)
    title = "FanoSeq Fano plane"
    if axis_scheme_id is not None:
        title += f": {axis_scheme_id}"
    draw.text((margin, margin // 2), title, fill=(20, 35, 45), font=fonts["title"])

    coords = _diagram_coordinates(size)
    colors = _line_colors()
    line_paths = _diagram_line_paths()
    for line in plane.lines:
        color = colors[line.index % len(colors)]
        path = line_paths[line.axes]
        if path == "circle":
            _draw_inner_circle(draw, coords, color, width=max(4, size // 150))
        else:
            points = [coords[axis] for axis in path]
            draw.line(points, fill=color, width=max(4, size // 150), joint="curve")
        _draw_line_label(draw, coords, line, color, fonts["small"])

    for point in plane.points:
        x, y = coords[point.axis]
        radius = int(size * 0.024)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(20, 74, 124),
            outline=(8, 35, 68),
            width=max(2, size // 300),
        )
        label = f"{point.symbol}: {point.label}"
        _draw_point_label(draw, (x, y), label, fonts["point"], size)

    legend_x = margin
    legend_y = int(size * 0.90)
    draw.text(
        (legend_x, legend_y),
        "Each colored line contains three imaginary axes; "
        "orientation is recorded in the exported tables.",
        fill=(80, 90, 100),
        font=fonts["small"],
    )
    image.save(output)
    return output


def _axis_labels(scheme_id: str | None) -> dict[int, str]:
    if scheme_id is None:
        return {axis: f"e{axis}" for axis in range(1, 8)}
    return get_axis_scheme(scheme_id).axis_labels()


def _line_metadata(
    scheme_id: str | None,
) -> tuple[dict[int, str], dict[tuple[int, int, int], str], dict[tuple[int, int, int], str]]:
    labels = _axis_labels(scheme_id)
    if scheme_id is None:
        return labels, {}, {}
    scheme = get_axis_scheme(scheme_id)
    line_labels = {line.axes: line.label for line in scheme.fano_lines}
    interpretations = {line.axes: line.interpretation for line in scheme.fano_lines}
    return labels, line_labels, interpretations


def _axis_line_membership() -> dict[int, list[str]]:
    membership = {axis: [] for axis in range(1, 8)}
    for line in FANO_LINES:
        key = line_key(line)
        for axis in line:
            membership[axis].append(key)
    return membership


def _pair_row(
    scheme_id: str | None,
    labels: dict[int, str],
    line: FanoLine,
    left: int,
    right: int,
    output: int,
    sign: int,
) -> dict[str, object]:
    sign_text = "+" if sign > 0 else "-"
    return {
        "axis_scheme_id": scheme_id or "basis",
        "left_axis": left,
        "right_axis": right,
        "left_axis_label": labels[left],
        "right_axis_label": labels[right],
        "unordered_pair": f"({min(left, right)},{max(left, right)})",
        "output_axis": output,
        "output_axis_label": labels[output],
        "product_sign": sign,
        "product_rule": f"e{left}*e{right}={sign_text}e{output}",
        "fano_line": line.key,
        "line_label": line.label,
        "is_ordered_product": True,
    }


def _third_axis(left: int, right: int) -> int:
    for line in FANO_LINES:
        if left in line and right in line:
            return next(axis for axis in line if axis not in {left, right})
    raise ValueError(f"No Fano line contains axes {left} and {right}.")


def _feature_group_columns(table: pd.DataFrame) -> list[str]:
    columns = ["sequence_id"]
    for column in ("mode", "seq_type", "axis_scheme_id", "frame"):
        if column in table.columns:
            columns.append(column)
    return columns


def _line_profile(group: pd.DataFrame) -> pd.DataFrame:
    stats = (
        group.groupby("fano_line")["line_contribution_norm"]
        .agg(["sum", "mean", "max", "std"])
        .reindex(FANO_LINE_KEYS)
        .fillna(0.0)
    )
    stats["std"] = stats["std"].fillna(0.0)
    return stats


def _line_share_vector(group: pd.DataFrame) -> np.ndarray:
    profile = _line_profile(group)["sum"].to_numpy(dtype=float)
    total = float(profile.sum())
    return profile / total if total > 0 else profile


def _normalized_entropy(values: np.ndarray) -> float:
    positive = values[values > 0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)) / log(len(FANO_LINE_KEYS)))


def _feature_prefix(line: str) -> str:
    return "fano_line_" + line.strip("()").replace(",", "_")


def _dominant_line_label(group: pd.DataFrame, line: str) -> str:
    if line == "NA" or "line_label" not in group.columns:
        return "NA"
    labels = group.loc[group["fano_line"] == line, "line_label"].dropna().astype(str)
    return labels.iloc[0] if not labels.empty else "NA"


def _axis_incident_shares(shares: pd.Series) -> dict[int, float]:
    values = {axis: 0.0 for axis in range(1, 8)}
    for line in FANO_LINES:
        share = float(shares[line_key(line)])
        for axis in line:
            values[axis] += share
    return values


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = sqrt(float(np.dot(left, left))) * sqrt(float(np.dot(right, right)))
    if denom == 0:
        return 1.0
    return float(np.dot(left, right) / denom)


def _empty_feature_table() -> pd.DataFrame:
    columns = [
        "sequence_id",
        "mode",
        "seq_type",
        "axis_scheme_id",
        "frame",
        "n_fano_rows",
        "total_line_contribution_norm",
        "fano_line_entropy",
        "fano_line_evenness",
        "fano_line_profile_l2",
        "dominant_fano_line",
        "dominant_fano_line_share",
        "dominant_fano_line_label",
    ]
    for key in FANO_LINE_KEYS:
        prefix = _feature_prefix(key)
        columns.extend(
            [
                f"{prefix}_sum",
                f"{prefix}_mean",
                f"{prefix}_max",
                f"{prefix}_std",
                f"{prefix}_share",
            ]
        )
    for axis in range(1, 8):
        columns.append(f"axis_e{axis}_incident_share")
    columns.extend(["dominant_axis", "dominant_axis_incident_share"])
    return pd.DataFrame(columns=columns)


def _plot_fonts(size: int) -> dict[str, ImageFont.ImageFont]:
    try:
        title = ImageFont.truetype("arial.ttf", max(24, size // 34))
        point = ImageFont.truetype("arial.ttf", max(14, size // 70))
        small = ImageFont.truetype("arial.ttf", max(12, size // 86))
    except OSError:
        title = ImageFont.load_default()
        point = ImageFont.load_default()
        small = ImageFont.load_default()
    return {"title": title, "point": point, "small": small}


def _diagram_coordinates(size: int) -> dict[int, tuple[int, int]]:
    return {
        1: (int(size * 0.50), int(size * 0.13)),
        2: (int(size * 0.16), int(size * 0.78)),
        3: (int(size * 0.33), int(size * 0.46)),
        4: (int(size * 0.84), int(size * 0.78)),
        5: (int(size * 0.67), int(size * 0.46)),
        6: (int(size * 0.50), int(size * 0.78)),
        7: (int(size * 0.50), int(size * 0.57)),
    }


def _diagram_line_paths() -> dict[tuple[int, int, int], tuple[int, ...] | str]:
    return {
        (1, 2, 3): (1, 3, 2),
        (1, 4, 5): (1, 5, 4),
        (1, 7, 6): (1, 7, 6),
        (2, 4, 6): (2, 6, 4),
        (2, 5, 7): (2, 7, 5),
        (3, 4, 7): (3, 7, 4),
        (3, 6, 5): "circle",
    }


def _line_colors() -> list[tuple[int, int, int]]:
    return [
        (20, 74, 124),
        (217, 5, 85),
        (46, 125, 50),
        (239, 124, 0),
        (106, 61, 154),
        (0, 137, 123),
        (120, 86, 50),
    ]


def _draw_inner_circle(
    draw: ImageDraw.ImageDraw,
    coords: dict[int, tuple[int, int]],
    color: tuple[int, int, int],
    width: int,
) -> None:
    x3, y3 = coords[3]
    x5, _ = coords[5]
    _, y6 = coords[6]
    center_x = (x3 + x5) / 2
    center_y = ((y6 * y6) - (y3 * y3) - ((x5 - x3) / 2) ** 2) / (2 * (y6 - y3))
    radius = abs(y6 - center_y)
    draw.ellipse(
        (
            int(center_x - radius),
            int(center_y - radius),
            int(center_x + radius),
            int(center_y + radius),
        ),
        outline=color,
        width=width,
    )


def _draw_line_label(
    draw: ImageDraw.ImageDraw,
    coords: dict[int, tuple[int, int]],
    line: FanoLine,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    xs = [coords[axis][0] for axis in line.axes]
    ys = [coords[axis][1] for axis in line.axes]
    x = int(sum(xs) / 3)
    y = int(sum(ys) / 3)
    draw.text((x + 8, y + 8), line.key, fill=color, font=font)


def _draw_point_label(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    label: str,
    font: ImageFont.ImageFont,
    size: int,
) -> None:
    x, y = point
    offset_x = int(size * 0.03)
    offset_y = -int(size * 0.025)
    if x > size * 0.70:
        offset_x = -int(size * 0.22)
    if x < size * 0.25:
        offset_x = int(size * 0.03)
    if y > size * 0.70:
        offset_y = int(size * 0.03)
    draw.text((x + offset_x, y + offset_y), label, fill=(20, 35, 45), font=font)
