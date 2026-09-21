"""Dataset validation, inspection reporting, and lightweight SVG EDA plots."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
TARGET = "charges"
SOURCE_TARGET = "charges_usd"
VALID_CATEGORIES = {
    "sex": ["female", "male"],
    "smoker": ["no", "yes"],
    "region": ["northeast", "northwest", "southeast", "southwest"],
}


@dataclass
class DatasetAudit:
    raw: pd.DataFrame
    modeling: pd.DataFrame
    report: dict[str, Any]


def _python_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def load_and_validate_dataset(
    dataset_path: str | Path,
    *,
    remove_duplicate_observations: bool = True,
) -> DatasetAudit:
    """Load the published file, report issues, and create a safe modeling view.

    The source CSV is never edited. ``record_id`` is retained for traceability but
    excluded as a predictor. The supplied ``charges_usd`` column is explicitly
    renamed to the internal target name ``charges``.
    """

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = pd.read_csv(path)
    source_columns = raw.columns.tolist()
    target_source_column = SOURCE_TARGET if SOURCE_TARGET in raw.columns else TARGET
    required = set(FEATURES + [target_source_column])
    missing_columns = sorted(required.difference(raw.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    frame = raw.rename(columns={target_source_column: TARGET}).copy()
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("string").str.strip().str.lower()

    numeric_columns = NUMERIC_FEATURES + [TARGET]
    non_numeric: dict[str, int] = {}
    for column in numeric_columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        non_numeric[column] = int((converted.isna() & frame[column].notna()).sum())
        frame[column] = converted

    missing_values = {column: int(value) for column, value in raw.isna().sum().items()}
    duplicate_subset = FEATURES + [TARGET]
    duplicate_mask = frame.duplicated(subset=duplicate_subset, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    duplicate_rows = []
    if duplicate_count:
        duplicate_rows = [
            {key: _python_value(value) for key, value in row.items()}
            for row in frame.loc[
                frame.duplicated(subset=duplicate_subset, keep=False),
                (["record_id"] if "record_id" in frame.columns else []) + duplicate_subset,
            ].to_dict(orient="records")
        ]

    invalid_counts = {
        "non_numeric_age": non_numeric["age"],
        "non_numeric_bmi": non_numeric["bmi"],
        "non_numeric_children": non_numeric["children"],
        "non_numeric_charges": non_numeric[TARGET],
        "age_outside_0_120": int((frame["age"].notna() & ~frame["age"].between(0, 120)).sum()),
        "bmi_nonpositive_or_over_100": int(
            (frame["bmi"].notna() & ((frame["bmi"] <= 0) | (frame["bmi"] > 100))).sum()
        ),
        "children_negative_or_noninteger": int(
            (
                frame["children"].notna()
                & ((frame["children"] < 0) | (frame["children"] % 1 != 0))
            ).sum()
        ),
        "charges_negative": int((frame[TARGET].notna() & (frame[TARGET] < 0)).sum()),
    }
    for column, categories in VALID_CATEGORIES.items():
        invalid_counts[f"invalid_{column}"] = int(
            (frame[column].notna() & ~frame[column].isin(categories)).sum()
        )

    if any(non_numeric.values()) or any(value for key, value in invalid_counts.items() if not key.startswith("invalid_")):
        raise ValueError(f"Invalid numeric/range values found: {invalid_counts}")
    category_errors = {
        key: value for key, value in invalid_counts.items() if key.startswith("invalid_") and value
    }
    if category_errors:
        raise ValueError(f"Invalid categorical values found: {category_errors}")

    modeling = frame.copy()
    if remove_duplicate_observations:
        modeling = modeling.loc[~duplicate_mask].copy()

    descriptive = (
        frame[NUMERIC_FEATURES + [TARGET]]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        .round(4)
        .to_dict()
    )
    descriptive = {
        column: {stat: _python_value(value) for stat, value in values.items()}
        for column, values in descriptive.items()
    }
    category_counts = {
        column: {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}
        for column in CATEGORICAL_FEATURES
    }
    charges = frame[TARGET]
    charge_distribution = {
        "mean": float(charges.mean()),
        "median": float(charges.median()),
        "standard_deviation": float(charges.std()),
        "skewness": float(charges.skew()),
        "minimum": float(charges.min()),
        "maximum": float(charges.max()),
        "quantiles": {
            str(q): float(value)
            for q, value in charges.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).items()
        },
    }
    report = {
        "dataset_path": str(path.resolve()),
        "raw_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "source_columns": source_columns,
        "internal_target": TARGET,
        "source_target_column": target_source_column,
        "data_types": {column: str(dtype) for column, dtype in raw.dtypes.items()},
        "missing_values": missing_values,
        "exact_duplicates_including_record_id": int(raw.duplicated().sum()),
        "duplicate_observations_excluding_record_id": duplicate_count,
        "duplicate_rows": duplicate_rows,
        "duplicate_policy": (
            "Source file preserved; later duplicate excluded from modeling view to avoid split leakage."
            if remove_duplicate_observations and duplicate_count
            else "No duplicate observations removed from modeling view."
        ),
        "modeling_rows": int(len(modeling)),
        "category_counts": category_counts,
        "invalid_value_counts": invalid_counts,
        "descriptive_statistics": descriptive,
        "charges_distribution": charge_distribution,
        "record_id": {
            "present": "record_id" in raw.columns,
            "used_as_feature": False,
            "unique": bool(raw["record_id"].is_unique) if "record_id" in raw.columns else None,
        },
    }
    return DatasetAudit(raw=raw, modeling=modeling, report=report)


def write_dataset_inspection_report(report: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = report["descriptive_statistics"]
    charges = report["charges_distribution"]
    categories = report["category_counts"]
    duplicate_rows = report["duplicate_rows"]
    duplicate_description = "None"
    if duplicate_rows:
        duplicate_description = "; ".join(
            f"record_id={row.get('record_id')} ({row['age']}, {row['sex']}, BMI {row['bmi']}, "
            f"children {row['children']}, smoker {row['smoker']}, {row['region']}, charge {row['charges']})"
            for row in duplicate_rows
        )

    lines = [
        "# Dataset Inspection Report",
        "",
        "## Source and schema",
        "",
        f"- Raw shape: **{report['raw_shape'][0]} rows × {report['raw_shape'][1]} columns**.",
        f"- Source columns: `{', '.join(report['source_columns'])}`.",
        f"- The supplied `{report['source_target_column']}` column is explicitly mapped to internal target `{report['internal_target']}`; the source CSV is not changed.",
        "- `record_id` is an identifier and is not used as a predictor.",
        f"- Modeling view after the documented duplicate policy: **{report['modeling_rows']} rows**.",
        "",
        "## Data types",
        "",
        "| Column | Source dtype |",
        "|---|---|",
        *[f"| {column} | {dtype} |" for column, dtype in report["data_types"].items()],
        "",
        "## Data quality",
        "",
        f"- Missing values: `{json.dumps(report['missing_values'], sort_keys=True)}`.",
        f"- Exact duplicates including `record_id`: **{report['exact_duplicates_including_record_id']}**.",
        f"- Duplicate observations excluding `record_id`: **{report['duplicate_observations_excluding_record_id']}**.",
        f"- Duplicate records: {duplicate_description}.",
        f"- Policy: {report['duplicate_policy']}",
        f"- Invalid-value checks: `{json.dumps(report['invalid_value_counts'], sort_keys=True)}`.",
        "",
        "## Ranges and categories",
        "",
        f"- Age: {stats['age']['min']:.0f}–{stats['age']['max']:.0f} years.",
        f"- BMI: {stats['bmi']['min']:.2f}–{stats['bmi']['max']:.2f}.",
        f"- Children: {stats['children']['min']:.0f}–{stats['children']['max']:.0f}.",
        f"- Sex: `{json.dumps(categories['sex'], sort_keys=True)}`.",
        f"- Smoker: `{json.dumps(categories['smoker'], sort_keys=True)}`.",
        f"- Region: `{json.dumps(categories['region'], sort_keys=True)}`.",
        "",
        "## Charges distribution",
        "",
        f"- Minimum: {charges['minimum']:.2f}; maximum: {charges['maximum']:.2f}.",
        f"- Mean: {charges['mean']:.2f}; median: {charges['median']:.2f}; standard deviation: {charges['standard_deviation']:.2f}.",
        f"- Skewness: {charges['skewness']:.3f}, indicating a pronounced right tail.",
        f"- 25th/75th percentiles: {charges['quantiles']['0.25']:.2f} / {charges['quantiles']['0.75']:.2f}.",
        f"- 95th/99th percentiles: {charges['quantiles']['0.95']:.2f} / {charges['quantiles']['0.99']:.2f}.",
        "",
        "## Basic descriptive statistics",
        "",
        "| Variable | Count | Mean | Std. dev. | Minimum | Median | Maximum |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {column} | {values['count']:.0f} | {values['mean']:.4f} | {values['std']:.4f} | {values['min']:.4f} | {values['50%']:.4f} | {values['max']:.4f} |"
            for column, values in stats.items()
        ],
        "",
        "## Context limitation",
        "",
        "This is the supplied published US medical-charge dataset, not Malaysian or NSS patient-level data. "
        "The bundle documentation describes the classic dataset as simulated from US demographic statistics. "
        "It is retained only as a secondary benchmark. No rows were fabricated or merged with NSS, LIAM, or Malaysian hospital-pricing data.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_document(title: str, body: str, width: int = 900, height: int = 600) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        '<style>text{font-family:Arial,sans-serif;fill:#172554}.title{font-size:24px;font-weight:700}.axis{font-size:13px}.small{font-size:11px}.grid{stroke:#dbeafe;stroke-width:1}.frame{stroke:#475569;stroke-width:1.2}</style>'
        f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{escape(title)}</text>{body}</svg>'
    )


def _linear_scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    if source_max == source_min:
        return (target_min + target_max) / 2
    return target_min + (value - source_min) * (target_max - target_min) / (source_max - source_min)


def _axes(x_label: str, y_label: str, width: int = 900, height: int = 600) -> str:
    return (
        '<line x1="80" y1="520" x2="860" y2="520" class="frame"/>'
        '<line x1="80" y1="70" x2="80" y2="520" class="frame"/>'
        f'<text x="470" y="570" text-anchor="middle" class="axis">{escape(x_label)}</text>'
        f'<text x="22" y="295" text-anchor="middle" class="axis" transform="rotate(-90 22 295)">{escape(y_label)}</text>'
    )


def _write_histogram(values: pd.Series, path: Path, title: str, x_label: str) -> None:
    counts, edges = np.histogram(values.to_numpy(dtype=float), bins=24)
    max_count = max(int(counts.max()), 1)
    body = [_axes(x_label, "Count")]
    bar_width = 780 / len(counts)
    for index, count in enumerate(counts):
        height = 430 * count / max_count
        x = 80 + index * bar_width
        body.append(f'<rect x="{x:.2f}" y="{520-height:.2f}" width="{bar_width-1:.2f}" height="{height:.2f}" fill="#2563eb" opacity="0.82"/>')
    for fraction in np.linspace(0, 1, 6):
        x = 80 + fraction * 780
        label = edges[0] + fraction * (edges[-1] - edges[0])
        body.append(f'<text x="{x:.1f}" y="542" text-anchor="middle" class="small">{label:,.0f}</text>')
    path.write_text(_svg_document(title, "".join(body)), encoding="utf-8")


def _write_scatter(frame: pd.DataFrame, x_column: str, path: Path, title: str) -> None:
    x_min, x_max = float(frame[x_column].min()), float(frame[x_column].max())
    y_min, y_max = float(frame[TARGET].min()), float(frame[TARGET].max())
    body = [_axes(x_column.upper(), "Charges (source dataset scale)")]
    for _, row in frame.iterrows():
        x = _linear_scale(float(row[x_column]), x_min, x_max, 85, 855)
        y = _linear_scale(float(row[TARGET]), y_min, y_max, 515, 75)
        color = "#dc2626" if row["smoker"] == "yes" else "#2563eb"
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{color}" opacity="0.42"/>')
    body.append('<circle cx="690" cy="82" r="5" fill="#2563eb"/><text x="701" y="86" class="small">non-smoker</text>')
    body.append('<circle cx="785" cy="82" r="5" fill="#dc2626"/><text x="796" y="86" class="small">smoker</text>')
    path.write_text(_svg_document(title, "".join(body)), encoding="utf-8")


def _write_group_bars(groups: Iterable[str], means: Iterable[float], path: Path, title: str) -> None:
    labels = list(groups)
    values = [float(value) for value in means]
    max_value = max(values) * 1.1 if values else 1
    body = [_axes("Group", "Mean charges")]
    slot = 760 / max(len(labels), 1)
    for index, (label, value) in enumerate(zip(labels, values)):
        height = 420 * value / max_value
        x = 92 + index * slot
        body.append(f'<rect x="{x:.1f}" y="{520-height:.1f}" width="{slot*0.68:.1f}" height="{height:.1f}" rx="3" fill="#0f766e"/>')
        body.append(f'<text x="{x+slot*0.34:.1f}" y="{508-height:.1f}" text-anchor="middle" class="small">{value:,.0f}</text>')
        body.append(f'<text x="{x+slot*0.34:.1f}" y="542" text-anchor="middle" class="small">{escape(str(label))}</text>')
    path.write_text(_svg_document(title, "".join(body)), encoding="utf-8")


def _write_correlation_heatmap(correlation: pd.DataFrame, path: Path) -> None:
    columns = correlation.columns.tolist()
    cell = 92
    start_x, start_y = 230, 120
    body: list[str] = []
    for index, label in enumerate(columns):
        body.append(f'<text x="{start_x+index*cell+cell/2}" y="102" text-anchor="middle" class="axis">{escape(label)}</text>')
        body.append(f'<text x="{start_x-12}" y="{start_y+index*cell+cell/2+5}" text-anchor="end" class="axis">{escape(label)}</text>')
    for row_index, row_name in enumerate(columns):
        for col_index, col_name in enumerate(columns):
            value = float(correlation.loc[row_name, col_name])
            strength = int(235 - abs(value) * 150)
            color = f"rgb({strength},{strength},{255 if value >= 0 else strength})" if value >= 0 else f"rgb(255,{strength},{strength})"
            x, y = start_x + col_index * cell, start_y + row_index * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff"/>')
            body.append(f'<text x="{x+cell/2}" y="{y+cell/2+5}" text-anchor="middle" class="axis">{value:.2f}</text>')
    path.write_text(_svg_document("Numerical correlation matrix", "".join(body)), encoding="utf-8")


def generate_eda_plots(frame: pd.DataFrame, plots_dir: str | Path) -> dict[str, str]:
    plots = Path(plots_dir)
    plots.mkdir(parents=True, exist_ok=True)
    outputs = {
        "charges_distribution": plots / "charges_distribution.svg",
        "age_vs_charges": plots / "age_vs_charges.svg",
        "bmi_vs_charges": plots / "bmi_vs_charges.svg",
        "smoker_vs_charges": plots / "smoker_vs_charges.svg",
        "sex_vs_charges": plots / "sex_vs_charges.svg",
        "children_vs_charges": plots / "children_vs_charges.svg",
        "region_vs_charges": plots / "region_vs_charges.svg",
        "correlation": plots / "numerical_correlation.svg",
    }
    _write_histogram(frame[TARGET], outputs["charges_distribution"], "Distribution of medical charges", "Charges (source dataset scale)")
    _write_scatter(frame, "age", outputs["age_vs_charges"], "Age vs medical charges")
    _write_scatter(frame, "bmi", outputs["bmi_vs_charges"], "BMI vs medical charges")
    for column in ["smoker", "sex", "children", "region"]:
        grouped = frame.groupby(column, observed=True)[TARGET].mean().sort_values(ascending=False)
        _write_group_bars(grouped.index.astype(str), grouped.values, outputs[f"{column}_vs_charges"], f"Mean charges by {column}")
    correlation = frame[NUMERIC_FEATURES + [TARGET]].corr()
    _write_correlation_heatmap(correlation, outputs["correlation"])
    return {name: str(path) for name, path in outputs.items()}


def summarize_eda(frame: pd.DataFrame) -> dict[str, Any]:
    grouped = {
        column: frame.groupby(column, observed=True)[TARGET].agg(["count", "mean", "median", "std"]).round(2).to_dict(orient="index")
        for column in ["smoker", "sex", "children", "region"]
    }
    return {
        "numeric_correlations_with_charges": frame[NUMERIC_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).round(4).to_dict(),
        "group_summaries": grouped,
        "charges_skewness": float(frame[TARGET].skew()),
    }
