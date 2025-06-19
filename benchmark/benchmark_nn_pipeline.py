# benchmark_nn_pipeline.py

import os
import subprocess
import time
import argparse
from tqdm import tqdm
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_image_files(image_dir: str) -> List[str]:
    """
    Returns a list of all .ppm image files in a directory.

    Args:
        image_dir: Directory containing image files.

    Returns:
        List of filenames ending with '.ppm'.
    """
    return [f for f in os.listdir(image_dir) if f.endswith(".ppm")]


def run_pipeline_for_image(
    run_script: str,
    image_path: str,
    output_csv_dir: str,
    model_path: str,
    overwrite: bool
) -> None:
    """
    Runs the NN detection pipeline on a single image.

    Args:
        run_script: Path to run_nn_pipeline.py.
        image_path: Path to the input image.
        output_csv_dir: Directory where prediction CSVs will be saved.
        model_path: Path to the trained NN model file (.pth).
        overwrite: Whether to overwrite existing predictions.
    """
    pred_csv_name = os.path.splitext(os.path.basename(image_path))[0] + ".csv"
    pred_csv_path = os.path.join(output_csv_dir, pred_csv_name)

    if not overwrite and os.path.exists(pred_csv_path):
        logging.info(f"Skipping {os.path.basename(image_path)} (prediction already exists).")
        return

    cmd = [
        "python", run_script,
        "--image", image_path,
        "--pred_csv_dir", output_csv_dir,
        "--model", model_path
    ]
    subprocess.run(cmd, check=True)


def run_single_benchmark(
    run_script: str,
    eval_script: str,
    image_dir: str,
    output_dir: str,
    model_path: str,
    ground_truth_csv: str,
    overwrite: bool
) -> None:
    """
    Runs the full benchmark of the NN pipeline on all test images and evaluates results.

    Args:
        run_script: Path to the NN pipeline script.
        eval_script: Path to the evaluation script.
        image_dir: Directory of test images.
        output_dir: Directory where prediction CSVs will be saved.
        model_path: Path to the trained NN model file.
        ground_truth_csv: Path to ground truth CSV file.
        overwrite: Whether to overwrite existing predictions.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = get_image_files(image_dir)
    if not image_files:
        logging.error(f"No .ppm images found in {image_dir}")
        return

    logging.info(f"Processing {len(image_files)} test images...")

    for fname in tqdm(image_files, desc="Running NN pipeline"):
        image_path = os.path.join(image_dir, fname)
        run_pipeline_for_image(run_script, image_path, output_dir, model_path, overwrite)

    subprocess.run([
        "python", eval_script,
        "--ground_truth", ground_truth_csv,
        # "--image_dir", image_dir,
        "--pred_dir", output_dir
    ], check=True)


def main(overwrite_predictions: bool) -> None:
    """
    Sets up and executes the neural network pipeline benchmark process.

    Args:
        overwrite_predictions: Whether to re-run predictions even if they exist.
    """
    start_time = time.time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    run_script = os.path.join(project_root, "nn_pipeline", "run_nn_pipeline.py")
    eval_script = os.path.join(project_root, "benchmark", "evaluate_predictions.py")
    model_path = os.path.join(project_root, "nn_pipeline", "simple_cnn.pth")
    image_dir = os.path.join(project_root, "data", "GTSRB", "Final_Test", "Images")
    ground_truth_csv = os.path.join(image_dir, "GT-final_test.test.csv")
    output_dir = os.path.join(project_root, "output", "nn_pipeline", "pred_csv")

    run_single_benchmark(
        run_script=run_script,
        eval_script=eval_script,
        image_dir=image_dir,
        output_dir=output_dir,
        model_path=model_path,
        ground_truth_csv=ground_truth_csv,
        overwrite=overwrite_predictions
    )

    elapsed = time.time() - start_time
    logging.info(f"NN Benchmark completed in {elapsed:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite_predictions",
        action="store_true",
        help="If set, re-run predictions even if already exist"
    )
    args = parser.parse_args()

    main(overwrite_predictions=args.overwrite_predictions)
