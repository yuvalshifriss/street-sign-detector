import os
import subprocess
import time
import argparse
import logging
from typing import List, Optional
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_image_files(image_dir: str) -> List[str]:
    """
    Retrieves all .ppm image files from the specified directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        List[str]: List of .ppm image filenames.
    """
    return [f for f in os.listdir(image_dir) if f.endswith(".ppm")]


def run_pipeline_for_image(
    run_script: str,
    image_path: str,
    output_csv_dir: str,
    overwrite: bool,
    min_area: Optional[int] = None
) -> None:
    """
    Runs the classical pipeline on a single image.

    Args:
        run_script (str): Path to the run_classical_pipeline.py script.
        image_path (str): Full path to the input image.
        output_csv_dir (str): Directory to save the prediction CSV.
        overwrite (bool): Whether to overwrite existing predictions.
        min_area (Optional[int]): Optional area threshold for filtering predictions.
    """
    pred_csv_name = os.path.splitext(os.path.basename(image_path))[0] + ".csv"
    pred_csv_path = os.path.join(output_csv_dir, pred_csv_name)

    if not overwrite and os.path.exists(pred_csv_path):
        logging.info(f"Skipping {os.path.basename(image_path)} (prediction already exists).")
        return

    cmd = ["python", run_script, "--image", image_path, "--pred_csv_dir", output_csv_dir]
    if min_area is not None:
        cmd += ["--min_area", str(min_area)]

    subprocess.run(cmd, check=True)


def run_single_benchmark(
    run_script: str,
    eval_script: str,
    image_dir: str,
    output_dir: str,
    ground_truth_csv: str,
    overwrite: bool
) -> None:
    """
    Runs the classical detection pipeline on all images and evaluates the predictions.

    Args:
        run_script (str): Path to the run_classical_pipeline.py script.
        eval_script (str): Path to the evaluate_predictions.py script.
        image_dir (str): Directory containing test images.
        output_dir (str): Directory to store prediction CSV files.
        ground_truth_csv (str): Path to the ground truth CSV file.
        overwrite (bool): Whether to overwrite existing predictions.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = get_image_files(image_dir)
    if not image_files:
        logging.error(f"No .ppm images found in {image_dir}")
        return

    logging.info(f"Processing {len(image_files)} test images...")

    for fname in tqdm(image_files, desc="Running classical pipeline"):
        image_path = os.path.join(image_dir, fname)
        run_pipeline_for_image(run_script, image_path, output_dir, overwrite)

    subprocess.run([
        "python", eval_script,
        "--ground_truth", ground_truth_csv,
        "--image_dir", image_dir,
        "--pred_dir", output_dir
    ], check=True)


def main(overwrite_predictions: bool) -> None:
    """
    Main function to benchmark the classical detection pipeline on the full test set.

    Args:
        overwrite_predictions (bool): If True, re-run predictions even if CSVs already exist.
    """
    start_time = time.time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    run_script = os.path.join(project_root, "classical_pipeline", "run_classical_pipeline.py")
    eval_script = os.path.join(project_root, "benchmark", "evaluate_predictions.py")
    image_dir = os.path.join(project_root, "data", "GTSRB", "Final_Test", "Images")
    ground_truth_csv = os.path.join(image_dir, "GT-final_test.test.csv")
    output_dir = os.path.join(project_root, "output", "classical_pipeline", "pred_csv")

    run_single_benchmark(run_script, eval_script, image_dir, output_dir, ground_truth_csv, overwrite_predictions)

    elapsed = time.time() - start_time
    logging.info(f"Classic benchmark completed in {elapsed:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark the classical street sign detection pipeline.")
    parser.add_argument(
        "--overwrite_predictions",
        action="store_true",
        help="If set, re-run predictions even if already exist"
    )
    args = parser.parse_args()

    main(overwrite_predictions=args.overwrite_predictions)
