import argparse
import os
import subprocess
from tqdm import tqdm
import time
import logging


def main(overwrite_predictions):
    min_areas = [300, 150, 100, 75, 50, 25]
    results = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    run_script = os.path.abspath(os.path.join(script_dir, "..", "classical_pipeline", "run_classical_pipeline.py"))
    image_dir = os.path.join(project_root, "data", "GTSRB", "Final_Test", "Images")
    output_csv_dir = os.path.join(project_root, "output", "classical_pipeline")

    for m in min_areas:
        cur_output_csv_dir = os.path.join(output_csv_dir, f"pred_csv_min_area_{m}")
        os.makedirs(cur_output_csv_dir, exist_ok=True)
        image_files = [f for f in os.listdir(image_dir) if f.endswith(".ppm")]
        if not image_files:
            logging.error(f"No .ppm images found in {image_dir}")
            exit(1)
        logging.info(f"Processing {len(image_files)} test images...")
        for fname in tqdm(image_files, desc="Running classical pipeline"):
            image_path = os.path.join(image_dir, fname)
            pred_csv_name = fname.replace(".ppm", ".csv")
            pred_csv_path = os.path.join(cur_output_csv_dir, pred_csv_name)
            if not overwrite_predictions and os.path.exists(pred_csv_path):
                logging.info(f"Skipping {fname} (prediction already exists).")
                continue
            subprocess.run([
                "python",
                run_script,
                "--image", image_path,
                "--pred_csv_dir", cur_output_csv_dir
            ], check=True)

        subprocess.run([
            "python", "benchmark/evaluate_predictions.py",
            "--ground_truth", "data/GTSRB/Final_Test/Images/GT-final_test.test.csv",
            "--pred_dir", cur_output_csv_dir
        ], check=True)

        df = pd.read_csv(os.path.join(cur_output_csv_dir, "evaluation_metrics.csv"))
        df['min_area'] = m
        results.append(df)

    pd.concat(results).to_csv("classical_pipeline_different_min_area.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite_predictions", action="store_true", help="If set, re-run predictions even if already exist")
    args = parser.parse_args()

    main(overwrite_predictions=args.overwrite_predictions)