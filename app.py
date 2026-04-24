from io import BytesIO
import argparse
import socket
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import plotly.express as px
from flask import Flask, render_template, request, send_file

app = Flask(__name__)


@app.get("/sample-dataset")
def sample_dataset():
    sample_path = Path(__file__).resolve().parent / "StudentsPerformance.csv"
    if not sample_path.exists():
        return "Sample dataset not found.", 404
    return send_file(
        sample_path,
        as_attachment=True,
        download_name="StudentsPerformance.csv",
        mimetype="text/csv",
    )


def find_default_dataset() -> Optional[Path]:
    project_root = Path(__file__).resolve().parent
    preferred_name = project_root / "StudentsPerformance.csv"
    if preferred_name.exists():
        return preferred_name

    search_paths = [project_root, project_root / "data"]
    candidates: List[Path] = []
    for base in search_paths:
        if base.exists():
            candidates.extend(sorted(base.glob("*.csv")))
            candidates.extend(sorted(base.glob("*.xlsx")))
            candidates.extend(sorted(base.glob("*.xls")))

    return candidates[0] if candidates else None


def load_dataframe_from_path(file_path: Path) -> pd.DataFrame:
    lower_name = file_path.name.lower()
    if lower_name.endswith(".csv"):
        return pd.read_csv(file_path)
    if lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
        return pd.read_excel(file_path)
    raise ValueError("Unsupported file format. Please use CSV or Excel.")


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
    """Compute Final_Grade from math/reading/writing scores if not present."""
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


def load_uploaded_dataframe(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))
    if lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
        return pd.read_excel(BytesIO(file_bytes))
    raise ValueError("Unsupported file format. Please upload CSV or Excel.")


def build_histogram_figure(df_numeric: pd.DataFrame) -> str:
    long_df = df_numeric.melt(var_name="Feature", value_name="Value").dropna()
    fig = px.histogram(
        long_df,
        x="Value",
        facet_col="Feature",
        facet_col_wrap=3,
        nbins=20,
        title="Histograms of Academic Variables",
        color_discrete_sequence=["#2dd4bf"],
    )
    fig.update_layout(
        height=max(500, 250 * ((len(df_numeric.columns) + 2) // 3)),
        margin=dict(t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    fig.for_each_annotation(lambda ann: ann.update(text=ann.text.split("=")[-1]))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_bar_chart_figure(df_numeric: pd.DataFrame) -> str:
    mean_scores = (
        df_numeric.mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Average"})
    )
    fig = px.bar(
        mean_scores,
        x="Feature",
        y="Average",
        title="Average Marks by Feature",
        color="Average",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(
        height=500,
        margin=dict(t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis_title="Feature",
        yaxis_title="Average Marks",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_scatter_figure(df_numeric: pd.DataFrame, target_col: str) -> Optional[str]:
    features = [col for col in df_numeric.columns if col != target_col]
    if not features:
        return None

    long_df = (
        df_numeric[features + [target_col]]
        .melt(
            id_vars=[target_col],
            value_vars=features,
            var_name="Feature",
            value_name="FeatureValue",
        )
        .dropna()
    )

    fig = px.scatter(
        long_df,
        x="FeatureValue",
        y=target_col,
        facet_col="Feature",
        facet_col_wrap=3,
        trendline="ols",
        title=f"Scatter Plots: Features vs {target_col}",
        color_discrete_sequence=["#60a5fa"],
        opacity=0.75,
    )
    fig.update_layout(
        height=max(500, 280 * ((len(features) + 2) // 3)),
        margin=dict(t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    fig.for_each_annotation(lambda ann: ann.update(text=ann.text.split("=")[-1]))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def analyze_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    df = ensure_final_grade(df)
    df_numeric = df.select_dtypes(include=["number"]).copy()
    if df_numeric.empty:
        raise ValueError("No numeric columns found in the uploaded dataset.")

    descriptive_stats = (
        pd.DataFrame(
            {
                "mean": df_numeric.mean(),
                "median": df_numeric.median(),
                "variance": df_numeric.var(),
                "std_dev": df_numeric.std(),
                "min": df_numeric.min(),
                "max": df_numeric.max(),
            }
        )
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    correlation_matrix = df_numeric.corr().round(3)
    target_col = detect_target_column(df_numeric.columns.tolist())

    total_students = int(df.shape[0])
    means = df_numeric.mean()
    avg_marks = round(means.mean(), 2) if not means.empty else 0
    best_metric = means.idxmax() if not means.empty else "N/A"
    best_metric_value = round(means.max(), 2) if not means.empty else 0

    return {
        "row_count": total_students,
        "column_count": int(df.shape[1]),
        "numeric_columns": list(df_numeric.columns),
        "target_col": target_col,
        "total_students": total_students,
        "avg_marks": avg_marks,
        "best_metric": best_metric,
        "best_metric_value": best_metric_value,
        "descriptive_stats_html": descriptive_stats.round(3).to_html(
            index=False, classes="table"
        ),
        "correlation_html": correlation_matrix.to_html(classes="table"),
        "bar_chart_plot": build_bar_chart_figure(df_numeric),
        "histogram_plot": build_histogram_figure(df_numeric),
        "scatter_plot": build_scatter_figure(df_numeric, target_col)
        if target_col
        else None,
        "preview_html": df.head(10).to_html(index=False, classes="table"),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    results = None
    filename = None

    if request.method == "POST":
        uploaded = request.files.get("dataset")

        if uploaded is not None and uploaded.filename != "":
            filename = uploaded.filename
            try:
                file_bytes = uploaded.read()
                df = load_uploaded_dataframe(uploaded.filename, file_bytes)
                results = analyze_dataframe(df)
                error = None
            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html", error=error, results=results, filename=filename
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Student Performance Analytics web app."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/IP to bind. Use 0.0.0.0 for LAN access.",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on."
    )
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    args = parser.parse_args()

    if args.host == "0.0.0.0":
        local_ip = socket.gethostbyname(socket.gethostname())
        print(f"LAN access: http://{local_ip}:{args.port}")
        print(f"Local access: http://127.0.0.1:{args.port}")
    else:
        print(f"Server running at: http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=args.debug)
