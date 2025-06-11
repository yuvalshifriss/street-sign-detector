import argparse
import cv2
import os
import csv
from classical_pipeline.candidate_detector import get_candidate_regions
from utils.func_utils import show

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image

def save_predictions(predictions, image_filename, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['filename', 'x', 'y', 'w', 'h'])  # header
        for (x, y, w, h) in predictions:
            writer.writerow([image_filename, x, y, w, h])

def main(image_path, output_path=None, pred_csv_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filename = os.path.basename(image_path)
    candidates = get_candidate_regions(image, output_path)
    result = draw_boxes(image.copy(), candidates)
    if output_path:
        show(result)
        output_img_path = os.path.join(output_path, os.path.basename(image_path).replace('.ppm', '.png'))
        cv2.imwrite(output_img_path, result)
        print(f"Saved output image to: {output_img_path}")

    if pred_csv_path:
        save_predictions(candidates, filename, pred_csv_path)
        print(f"Saved predictions to: {pred_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save output images (optional)")
    parser.add_argument("--pred_csv", help="Path to save predicted boxes as CSV (optional)")
    args = parser.parse_args()

    main(args.image, args.output, args.pred_csv)
