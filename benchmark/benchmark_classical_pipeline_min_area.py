# benchmark_classical_pipeline_min_area.py
import os
import subprocess
import time
import argparse
import pandas as pd
import logging
from tqdm import tqdm
from benchmark_classical_pipeline import (
    get_image_files,
    run_pipeline_for_image
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_benchmark_for_min_area(min_area, run_script, eval_script, image_dir, base_output_dir, ground_truth_csv, overwrite):
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
        "--image_dir", image_dir,
        "--pred_dir", output_dir
    ], check=True)

    eval_path = os.path.join(output_dir, "evaluation_metrics.csv")
    if not os.path.exists(eval_path):
        logging.warning(f"Evaluation file not found for min_area={min_area}")
        return None

    df = pd.read_csv(eval_path)
    df["min_area"] = min_area
    return df


def main(overwrite_predictions: bool):
    start_time = time.time()

    min_areas = [300, 150, 100, 75, 50, 25]
    results = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    run_script = os.path.join(project_root, "classical_pipeline", "run_classical_pipeline.py")
    eval_script = os.path.join(project_root, "benchmark", "evaluate_predictions.py")
    image_dir = os.path.join(project_root, "data", "GTSRB", "Final_Test", "Images")
    ground_truth_csv = os.path.join(image_dir, "GT-final_test.test.csv")
    base_output_dir = os.path.join(project_root, "output", "classical_pipeline")

    for min_area in min_areas:
        df = run_benchmark_for_min_area(min_area, run_script, eval_script, image_dir, base_output_dir, ground_truth_csv, overwrite_predictions)
        if df is not None:
            results.append(df)

    if results:
        combined = pd.concat(results, ignore_index=True)
        result_path = os.path.join(project_root, "output", "classical_pipeline_different_min_area.csv")
        combined.to_csv(result_path, index=False)
        logging.info(f"Saved combined results to: {result_path}")

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
