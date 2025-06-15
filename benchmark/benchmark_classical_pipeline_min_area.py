import argparse
import os
import subprocess
import time
import logging
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_image_files(image_dir):
    return [f for f in os.listdir(image_dir) if f.endswith(".ppm")]


def run_pipeline_for_image(run_script, image_path, output_csv_dir, overwrite):
    pred_csv_name = os.path.splitext(os.path.basename(image_path))[0] + ".csv"
    pred_csv_path = os.path.join(output_csv_dir, pred_csv_name)

    if not overwrite and os.path.exists(pred_csv_path):
        logging.info(f"Skipping {os.path.basename(image_path)} (prediction already exists).")
        return

    subprocess.run([
        "python", run_script,
        "--image", image_path,
        "--pred_csv_dir", output_csv_dir
    ], check=True)


def run_benchmark_for_min_area(min_area, run_script, image_dir, base_output_dir, overwrite):
    logging.info(f"\n=== Running for min_area = {min_area} ===")
    output_dir = os.path.join(base_output_dir, f"pred_csv_min_area_{min_area}")
    os.makedirs(output_dir, exist_ok=True)

    image_files = get_image_files(image_dir)
    if not image_files:
        logging.error(f"No .ppm images found in {image_dir}")
        return None

    logging.info(f"Processing {len(image_files)} test images...")
    for fname in tqdm(image_files, desc=f"min_area={min_area}"):
        image_path = os.path.join(image_dir, fname)
        run_pipeline_for_image(run_script, image_path, output_dir, overwrite)

    subprocess.run([
        "python", "benchmark/evaluate_predictions.py",
        "--ground_truth", os.path.join(image_dir, "GT-final_test.test.csv"),
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
    min_areas = [300, 150, 100, 75, 50, 25]
    results = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    run_script = os.path.join(project_root, "classical_pipeline", "run_classical_pipeline.py")
    image_dir = os.path.join(project_root, "data", "GTSRB", "Final_Test", "Images")
    base_output_dir = os.path.join(project_root, "output", "classical_pipeline")

    start_time = time.time()
    for min_area in min_areas:
        df = run_benchmark_for_min_area(min_area, run_script, image_dir, base_output_dir, overwrite_predictions)
        if df is not None:
            results.append(df)

    all_results = pd.concat(results, ignore_index=True)
    result_path = os.path.join(project_root, "output", "classical_pipeline_different_min_area.csv")
    all_results.to_csv(result_path, index=False)

    elapsed = time.time() - start_time
    logging.info(f"Benchmark completed in {elapsed:.2f} seconds.")
    logging.info(f"Saved combined results to: {result_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite_predictions",
        action="store_true",
        help="If set, re-run predictions even if already exist"
    )
    args = parser.parse_args()

    main(overwrite_predictions=args.overwrite_predictions)
