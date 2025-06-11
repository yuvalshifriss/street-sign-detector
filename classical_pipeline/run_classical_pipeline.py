import argparse
import cv2
import os
import csv
import logging

from classical_pipeline.candidate_detector import get_candidate_regions
from utils.func_utils import show

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image


def save_predictions_to_csv(predictions, csv_path):
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'w', 'h'])  # CSV header (no filename needed per file)
        for (x, y, w, h) in predictions:
            writer.writerow([x, y, w, h])


def main(image_path, pred_png_dir=None, pred_csv_dir=None, show_result=False):
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return

    image_filename = os.path.basename(image_path)
    candidate_boxes = get_candidate_regions(image, show_steps=show_result)
    image_with_boxes = draw_boxes(image.copy(), candidate_boxes)

    if pred_png_dir:
        os.makedirs(pred_png_dir, exist_ok=True)
        output_filename = os.path.splitext(image_filename)[0] + ".png"
        output_img_path = os.path.join(pred_png_dir, output_filename)
        cv2.imwrite(output_img_path, image_with_boxes)
        logging.info(f"Saved annotated image to: {output_img_path}")

    if show_result:
        show(image_with_boxes, title="Final Result")

    if pred_csv_dir:
        os.makedirs(pred_csv_dir, exist_ok=True)
        output_filename = os.path.splitext(image_filename)[0] + ".csv"
        output_csv_path = os.path.join(pred_csv_dir, output_filename)
        save_predictions_to_csv(candidate_boxes, output_csv_path)
        logging.info(f"Saved predictions to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect street sign regions using a classical image processing pipeline.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--pred_png_dir", help="Directory to save annotated PNG images (optional)")
    parser.add_argument("--pred_csv_dir", help="Directory to save combined predictions.csv (optional)")
    parser.add_argument("--show", action="store_true", help="Display debugging and final result images")
    args = parser.parse_args()

    main(
        image_path=args.image,
        pred_png_dir=args.pred_png_dir,
        pred_csv_dir=args.pred_csv_dir,
        show_result=args.show
    )
