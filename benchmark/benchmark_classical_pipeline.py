import os
import subprocess
import time
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main(overwrite_predictions: bool):
    # === Start timer ===
    start_time = time.time()

    # === Root relative to this file ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # === Paths ===
    image_dir = os.path.join(project_root, "data", "GTSRB", "Final_Test", "Images")
    output_csv_dir = os.path.join(project_root, "output", "classical_pipeline", "pred_csv")

    # === Ensure CSV output dir exists ===
    os.makedirs(output_csv_dir, exist_ok=True)

    # === Collect test images ===
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".ppm")]

    if not image_files:
        logging.error(f"No .ppm images found in {image_dir}")
        exit(1)

    logging.info(f"Processing {len(image_files)} test images...")

    run_script = os.path.abspath(os.path.join(script_dir, "..", "classical_pipeline", "run_classical_pipeline.py"))

    for fname in tqdm(image_files, desc="Running classical pipeline"):
        image_path = os.path.join(image_dir, fname)
        pred_csv_name = fname.replace(".ppm", ".csv")
        pred_csv_path = os.path.join(output_csv_dir, pred_csv_name)

        if not overwrite_predictions and os.path.exists(pred_csv_path):
            logging.info(f"Skipping {fname} (prediction already exists).")
            continue

        subprocess.run([
            "python",
            run_script,
            "--image", image_path,
            "--pred_csv_dir", output_csv_dir
        ], check=True)

    # === Run evaluation after predictions ===
    evaluate_script = os.path.join(project_root, "benchmark", "evaluate_predictions.py")
    gt_csv = os.path.join(image_dir, "GT-final_test.test.csv")

    logging.info("Running evaluation on predicted results...")
    subprocess.run([
        "python",
        evaluate_script,
        "--ground_truth", gt_csv,
        "--pred_dir", output_csv_dir
    ])

    # === Report elapsed time ===
    elapsed = time.time() - start_time
    logging.info(f"Benchmark completed in {elapsed:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite_predictions", action="store_true", help="If set, re-run predictions even if already exist")
    args = parser.parse_args()

    main(overwrite_predictions=args.overwrite_predictions)
