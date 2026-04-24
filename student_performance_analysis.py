import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")


def find_dataset_path(data_dir: Path) -> Optional[Path]:
    """Return the first CSV/XLSX file found in data_dir."""
    candidates = sorted(
        list(data_dir.glob("*.csv"))
        + list(data_dir.glob("*.xlsx"))
        + list(data_dir.glob("*.xls"))
    )
    return candidates[0] if candidates else None


def load_dataset(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file format: {suffix}")


def normalize_col(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def detect_target_column(columns: List[str]) -> Optional[str]:
    keywords = {
        "finalgrade",
        "finalmarks",
        "finalscore",
        "grade",
        "result",
        "gpa",
        "cgpa",
        "writingscore",
        "mathscore",
        "readingscore",
    }
    normalized_map = {col: normalize_col(col) for col in columns}

    for col, ncol in normalized_map.items():
        if ncol in keywords:
            return col

    for col, ncol in normalized_map.items():
        if "final" in ncol and ("grade" in ncol or "mark" in ncol or "score" in ncol):
            return col

    for col, ncol in normalized_map.items():
        if "score" in ncol:
            return col

    return None


def ensure_final_grade(df: pd.DataFrame) -> pd.DataFrame:
    if "Final_Grade" in df.columns:
        return df
    score_cols = [
        c for c in df.columns
        if normalize_col(c) in {"mathscore", "readingscore", "writingscore"}
    ]
    if len(score_cols) >= 1:
        df = df.copy()
        df["Final_Grade"] = df[score_cols].mean(axis=1).round(1)
    return df


def descriptive_stats(df_numeric: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(
        {
            "mean": df_numeric.mean(numeric_only=True),
            "median": df_numeric.median(numeric_only=True),
            "variance": df_numeric.var(numeric_only=True),
            "std_dev": df_numeric.std(numeric_only=True),
            "min": df_numeric.min(numeric_only=True),
            "max": df_numeric.max(numeric_only=True),
        }
    )
    return stats.sort_index()


def save_histograms(df_numeric: pd.DataFrame, output_path: Path) -> None:
    n_cols = len(df_numeric.columns)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(
        nrows=(n_cols + 2) // 3,
        ncols=3,
        figsize=(15, 4 * ((n_cols + 2) // 3)),
        squeeze=False,
    )

    flat_axes = axes.flatten()
    for idx, col in enumerate(df_numeric.columns):
        sns.histplot(df_numeric[col].dropna(), kde=True, ax=flat_axes[idx], color="#2a9d8f")
        flat_axes[idx].set_title(f"Histogram: {col}")

    for j in range(idx + 1, len(flat_axes)):
        flat_axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_boxplots(df_numeric: pd.DataFrame, output_path: Path) -> None:
    n_cols = len(df_numeric.columns)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(
        nrows=(n_cols + 2) // 3,
        ncols=3,
        figsize=(15, 4 * ((n_cols + 2) // 3)),
        squeeze=False,
    )

    flat_axes = axes.flatten()
    for idx, col in enumerate(df_numeric.columns):
        sns.boxplot(y=df_numeric[col], ax=flat_axes[idx], color="#f4a261")
        flat_axes[idx].set_title(f"Box Plot: {col}")

    for j in range(idx + 1, len(flat_axes)):
        flat_axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_scatterplots(df_numeric: pd.DataFrame, target_col: str, output_path: Path) -> None:
    features = [col for col in df_numeric.columns if col != target_col]
    if not features:
        return

    n_cols = len(features)
    fig, axes = plt.subplots(
        nrows=(n_cols + 2) // 3,
        ncols=3,
        figsize=(15, 4 * ((n_cols + 2) // 3)),
        squeeze=False,
    )

    flat_axes = axes.flatten()
    for idx, col in enumerate(features):
        sns.scatterplot(data=df_numeric, x=col, y=target_col, ax=flat_axes[idx], color="#264653")
        # add regression line
        sns.regplot(data=df_numeric, x=col, y=target_col, ax=flat_axes[idx], scatter=False, color="#e76f51")
        flat_axes[idx].set_title(f"{col} vs {target_col}")

    for j in range(idx + 1, len(flat_axes)):
        flat_axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_correlation_heatmap(df_numeric: pd.DataFrame, output_path: Path) -> None:
    corr = df_numeric.corr()
    plt.figure(figsize=(max(6, len(corr.columns)), max(5, len(corr.columns) * 0.8)))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze student performance data using descriptive stats and visualizations."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="Path to CSV/XLSX dataset. If omitted, first file in data/ is used.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    file_path = Path(args.file) if args.file else find_dataset_path(data_dir)

    if file_path is None:
        raise FileNotFoundError(
            "No dataset found. Add a CSV/XLSX file in 'data/' or pass --file <path>."
        )

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = load_dataset(file_path)
    if df.empty:
        raise ValueError("Dataset is empty.")

    df = ensure_final_grade(df)
    df_numeric = df.select_dtypes(include=["number"]).copy()
    if df_numeric.empty:
        raise ValueError("No numeric columns found for statistical analysis.")

    stats = descriptive_stats(df_numeric)
    corr = df_numeric.corr(numeric_only=True)

    stats.to_csv(output_dir / "descriptive_statistics.csv")
    corr.to_csv(output_dir / "correlation_matrix.csv")

    save_histograms(df_numeric, output_dir / "histograms.png")
    save_boxplots(df_numeric, output_dir / "boxplots.png")
    save_correlation_heatmap(df_numeric, output_dir / "correlation_heatmap.png")

    target_col = detect_target_column(df_numeric.columns.tolist())
    if target_col:
        save_scatterplots(df_numeric, target_col, output_dir / "scatterplots_vs_target.png")

    print("Analysis complete.")
    print(f"Dataset used: {file_path}")
    print(f"Numeric columns analyzed: {len(df_numeric.columns)}")
    print(f"Descriptive statistics: {output_dir / 'descriptive_statistics.csv'}")
    print(f"Correlation matrix: {output_dir / 'correlation_matrix.csv'}")
    print(f"Correlation heatmap: {output_dir / 'correlation_heatmap.png'}")
    print(f"Histograms: {output_dir / 'histograms.png'}")
    print(f"Box plots: {output_dir / 'boxplots.png'}")
    if target_col:
        print(f"Scatter plots vs target ({target_col}): {output_dir / 'scatterplots_vs_target.png'}")
    else:
        print("Target column not auto-detected, so scatterplot-vs-target was skipped.")


if __name__ == "__main__":
    main()
