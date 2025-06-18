import argparse
import cv2
import os
import csv
import logging
from typing import List, Tuple, Optional

from classical_pipeline.candidate_detector import get_candidate_regions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def draw_boxes(image: 'np.ndarray', boxes: List[Tuple[int, int, int, int]],
               color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> 'np.ndarray':
    """
    Draws bounding boxes on an image.

    Args:
        image: The input image as a NumPy array.
        boxes: List of bounding boxes in (x, y, w, h) format.
        color: RGB color for the box outlines.
        thickness: Thickness of the rectangle lines.

    Returns:
        Annotated image with rectangles drawn.
    """
    for (x, y, w, h) in boxes:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image


def save_predictions_to_csv(predictions: List[Tuple[int, int, int, int]], csv_path: str) -> None:
    """
    Saves bounding box predictions to a CSV file.

    Args:
        predictions: List of bounding boxes in (x, y, w, h) format.
        csv_path: File path where predictions will be saved.
    """
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'w', 'h'])
        for (x, y, w, h) in predictions:
            writer.writerow([x, y, w, h])


def main(image_path: str,
         pred_png_dir: Optional[str] = None,
         pred_csv_dir: Optional[str] = None,
         min_area: int = 25) -> None:
    """
    Main entry point to process an image using the classical detection pipeline.

    Args:
        image_path: Path to the input image (.ppm format expected).
        pred_png_dir: Directory to save annotated images (optional).
        pred_csv_dir: Directory to save prediction CSV files (optional).
        min_area: Minimum area threshold to filter out small detections.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return

    image_filename = os.path.basename(image_path)
    cur_save_path = None
    if pred_png_dir:
        cur_save_path = os.path.join(pred_png_dir, image_filename.replace('ppm', 'png'))

    candidate_boxes = get_candidate_regions(image, save_path=cur_save_path, min_area=min_area)

    # Save annotated image if requested
    if pred_png_dir:
        os.makedirs(pred_png_dir, exist_ok=True)
        image_with_boxes = draw_boxes(image.copy(), candidate_boxes)
        output_filename = os.path.splitext(image_filename)[0] + ".png"
        output_img_path = os.path.join(pred_png_dir, output_filename)
        cv2.imwrite(output_img_path, image_with_boxes)
        logging.info(f"Saved annotated image to: {output_img_path}")

    # Save predictions to CSV if requested
    if pred_csv_dir:
        os.makedirs(pred_csv_dir, exist_ok=True)
        output_filename = os.path.splitext(image_filename)[0] + ".csv"
        output_csv_path = os.path.join(pred_csv_dir, output_filename)
        save_predictions_to_csv(candidate_boxes, output_csv_path)
        logging.info(f"Saved predictions to: {output_csv_path}")


if __name__ == "__main__":
    # run example:
    # python run_classical_pipeline.py \
    # --image "R:\projects\street-sign-detector\data\GTSRB\Final_Test\Images\00042.ppm" \
    # --pred_png_dir "R:\projects\street-sign-detector\output\classical_pipeline\pred_png" \
    # --pred_csv_dir "R:\projects\street-sign-detector\output\classical_pipeline\pred_csv" \
    # --min_area 300
    parser = argparse.ArgumentParser(description="Detect street sign regions using a classical image processing pipeline.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--pred_png_dir", help="Directory to save annotated PNG images (optional)")
    parser.add_argument("--pred_csv_dir", help="Directory to save prediction CSV files (optional)")
    parser.add_argument("--min_area", type=int, default=300, help="Minimum area threshold for detected bounding boxes")
    args = parser.parse_args()

    main(
        image_path=args.image,
        pred_png_dir=args.pred_png_dir,
        pred_csv_dir=args.pred_csv_dir,
        min_area=args.min_area
    )
