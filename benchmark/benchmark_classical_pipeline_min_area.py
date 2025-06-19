# benchmark_classical_pipeline_min_area.py

import os
import subprocess
import time
import argparse
import pandas as pd
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import write_html
from tqdm import tqdm
from typing import Optional, List

from benchmark_classical_pipeline import get_image_files, run_pipeline_for_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_benchmark_for_min_area(
        min_area: int,
        run_script: str,
        eval_script: str,
        image_dir: str,
        base_output_dir: str,
        ground_truth_csv: str,
        overwrite: bool
) -> Optional[pd.DataFrame]:
    """
    Runs the classical pipeline benchmark for a given min_area value.

    Args:
        min_area: Minimum contour area to consider for detection.
        run_script: Path to the run_classical_pipeline.py script.
        eval_script: Path to the evaluation script.
        image_dir: Directory containing test images (.ppm).
        base_output_dir: Base directory to store prediction CSVs.
        ground_truth_csv: Path to ground truth annotations.
        overwrite: Whether to overwrite existing predictions.

    Returns:
        A pandas DataFrame containing evaluation results with `min_area` included,
        or None if the evaluation file is missing.
    """
    logging.info(f"\n=== Running for min_area = {min_area} ===")
    output_dir = os.path.join(base_output_dir, f"pred_csv_min_area_{min_area}")
    os.makedirs(output_dir, exist_ok=True)

    image_files = get_image_files(image_dir)
    logging.info(f"Processing {len(image_files)} test images...")

    for fname in tqdm(image_files, desc=f"min_area={min_area}"):
        image_path = os.path.join(image_dir, fname)
        run_pipeline_for_image(run_script, image_path, output_dir, overwrite, min_area)

    subprocess.run([
        "python", eval_script,
        "--ground_truth", ground_truth_csv,
        # "--image_dir", image_dir,
        "--pred_dir", output_dir
    ], check=True)

    eval_path = os.path.join(output_dir, "evaluation_metrics.csv")
    if not os.path.exists(eval_path):
        logging.warning(f"Evaluation file not found for min_area={min_area}")
        return None

    df = pd.read_csv(eval_path)
    df["min_area"] = min_area
    return df


def visualize_results_plotly(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves an HTML visualization showing F1, Precision, and Recall scores vs min_area.

    Args:
        df: DataFrame with columns ['f1', 'precision', 'recall', 'min_area'].
        output_path: Path to save the generated HTML file.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("F1 Score vs min_area", "Precision vs min_area", "Recall vs min_area"),
        horizontal_spacing=0.15
    )

    df_sorted = df.sort_values("min_area")

    fig.add_trace(go.Scatter(
        x=df_sorted["min_area"], y=df_sorted["f1"],
        mode="lines+markers", name="F1 Score"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_sorted["min_area"], y=df_sorted["precision"],
        mode="lines+markers", name="Precision"
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=df_sorted["min_area"], y=df_sorted["recall"],
        mode="lines+markers", name="Recall"
    ), row=1, col=3)

    fig.update_layout(
        title_text="Evaluation Metrics vs min_area",
        height=400, width=1000,
        showlegend=False
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_html(fig, output_path)


def main(overwrite_predictions: bool) -> None:
    """
    Main function to benchmark the classical pipeline across various min_area values
    and visualize the impact on detection performance.

    Args:
        overwrite_predictions: Whether to re-run predictions if files already exist.
    """
    start_time = time.time()

    min_areas: List[int] = [300, 150, 100, 75, 50, 25]
    results: List[pd.DataFrame] = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    run_script = os.path.join(project_root, "classical_pipeline", "run_classical_pipeline.py")
    eval_script = os.path.join(project_root, "benchmark", "evaluate_predictions.py")
    image_dir = os.path.join(project_root, "data", "GTSRB", "Final_Test", "Images")
    ground_truth_csv = os.path.join(image_dir, "GT-final_test.test.csv")
    base_output_dir = os.path.join(project_root, "output", "classical_pipeline")

    for min_area in min_areas:
        df = run_benchmark_for_min_area(min_area, run_script, eval_script, image_dir, base_output_dir, ground_truth_csv,
                                        overwrite_predictions)
        if df is not None:
            results.append(df)

    if results:
        combined = pd.concat(results, ignore_index=True)
        result_path = os.path.join(project_root, "output", "classical_pipeline_different_min_area.csv")
        combined.to_csv(result_path, index=False)
        logging.info(f"Saved combined results to: {result_path}")

        html_path = os.path.join(project_root, "output", "classical_pipeline_different_min_area.html")
        visualize_results_plotly(combined, html_path)

    elapsed = time.time() - start_time
    logging.info(f"Benchmark completed in {elapsed:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite_predictions",
        action="store_true",
        help="If set, re-run predictions even if already exist"
    )
    args = parser.parse_args()

    main(overwrite_predictions=args.overwrite_predictions)
