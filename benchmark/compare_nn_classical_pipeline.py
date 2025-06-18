# compare_nn_classical_pipeline.py

import os
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple


def load_metrics(csv_path: str) -> pd.Series:
    """
    Load evaluation metrics from a CSV file.

    Parameters:
        csv_path (str): Path to the evaluation_metrics.csv file.

    Returns:
        pd.Series: A single-row Series of evaluation metrics.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    return pd.read_csv(csv_path).iloc[0]


def compare_metrics(classical_metrics: pd.Series, nn_metrics: pd.Series) -> pd.DataFrame:
    """
    Create a DataFrame comparing classical and NN metrics side-by-side.

    Parameters:
        classical_metrics (pd.Series): Metrics from the classical pipeline.
        nn_metrics (pd.Series): Metrics from the neural network pipeline.

    Returns:
        pd.DataFrame: Comparison table.
    """
    comparison = pd.DataFrame({
        "Metric": classical_metrics.index,
        "Classical": classical_metrics.values,
        "Neural Network": nn_metrics.values
    })

    print("\n=== Comparison of Classical vs Neural Network Pipelines ===")
    print(comparison.to_string(index=False))
    return comparison


def plot_comparison_subplots(comparison_df: pd.DataFrame, output_html: str) -> None:
    """
    Create two subplot bar charts:
    - One for count metrics (TP, FP, FN, TN)
    - One for percentage metrics (precision, recall, f1)

    Parameters:
        comparison_df (pd.DataFrame): DataFrame with metrics for both pipelines.
        output_html (str): Path to save the HTML visualization.
    """
    from plotly.subplots import make_subplots

    # Split metrics
    count_metrics = ["TP", "FP", "FN", "TN"]
    percent_metrics = ["precision", "recall", "f1"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Detection Counts", "Performance Scores (%)")
    )

    # Plot count metrics
    for method, color in zip(["Classical", "Neural Network"], ["lightskyblue", "salmon"]):
        values = comparison_df[comparison_df["Metric"].isin(count_metrics)]
        fig.add_trace(
            go.Bar(x=values["Metric"], y=values[method], name=f"{method} Counts", marker_color=color),
            row=1, col=1
        )

    # Plot percentage metrics (×100)
    for method, color in zip(["Classical", "Neural Network"], ["lightskyblue", "salmon"]):
        values = comparison_df[comparison_df["Metric"].isin(percent_metrics)].copy()
        values[method] = values[method] * 100
        fig.add_trace(
            go.Bar(x=values["Metric"], y=values[method], name=f"{method} %", marker_color=color, showlegend=False),
            row=1, col=2
        )

    fig.update_layout(
        title_text="Comparison of Classical vs Neural Network Pipelines",
        barmode="group",
        bargap=0.25,
        template="plotly_white",
        width=1000,
        height=500
    )

    fig.write_html(output_html)
    print(f"\n✅ Saved subplot comparison to: {output_html}")


def main(classical_csv: str, nn_csv: str, output_html: str) -> None:
    """
    Load metrics from classical and NN pipelines, compare them and generate visualization.

    Parameters:
        classical_csv (str): Path to classical pipeline metrics CSV.
        nn_csv (str): Path to neural network pipeline metrics CSV.
        output_html (str): Path to output HTML plot.
    """
    classical_metrics = load_metrics(classical_csv)
    nn_metrics = load_metrics(nn_csv)

    # Ensure same metric order
    common_metrics = classical_metrics.index.intersection(nn_metrics.index)
    classical_metrics = classical_metrics[common_metrics]
    nn_metrics = nn_metrics[common_metrics]

    comparison_df = compare_metrics(classical_metrics, nn_metrics)
    plot_comparison_subplots(comparison_df, output_html)


if __name__ == "__main__":
    # Define default paths relative to project structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    classical_csv = os.path.join(project_root, "output", "classical_pipeline", "pred_csv", "evaluation_metrics.csv")
    nn_csv = os.path.join(project_root, "output", "nn_pipeline", "pred_csv", "evaluation_metrics.csv")
    output_html = os.path.join(project_root, "output", "compare_pipelines.html")

    main(classical_csv, nn_csv, output_html)
