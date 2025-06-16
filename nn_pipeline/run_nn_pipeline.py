import argparse
import os
import csv
import logging
import cv2
import torch
from torchvision import transforms
from PIL import Image

from nn_pipeline.simple_cnn import SimpleCNN

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for (x, y, w, h) in boxes:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image


def save_predictions_to_csv(predictions, csv_path):
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'w', 'h'])
        for (x, y, w, h) in predictions:
            writer.writerow([x, y, w, h])


def predict_bounding_box(model, image_path, device, transform):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()[0]
    return [tuple(output)]  # Output as a list of one bounding box


def main(image_path, model_path, pred_png_dir=None, pred_csv_dir=None):
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    image_filename = os.path.basename(image_path)
    predicted_boxes = predict_bounding_box(model, image_path, device, transform)

    # Save annotated image if requested
    if pred_png_dir:
        os.makedirs(pred_png_dir, exist_ok=True)
        image_with_boxes = draw_boxes(image.copy(), predicted_boxes)
        output_filename = os.path.splitext(image_filename)[0] + ".png"
        output_img_path = os.path.join(pred_png_dir, output_filename)
        cv2.imwrite(output_img_path, image_with_boxes)
        logging.info(f"Saved annotated image to: {output_img_path}")

    # Save predictions to CSV if requested
    if pred_csv_dir:
        os.makedirs(pred_csv_dir, exist_ok=True)
        output_filename = os.path.splitext(image_filename)[0] + ".csv"
        output_csv_path = os.path.join(pred_csv_dir, output_filename)
        save_predictions_to_csv(predicted_boxes, output_csv_path)
        logging.info(f"Saved predictions to: {output_csv_path}")


if __name__ == "__main__":
    # run example:
    # python run_nn_pipeline.py \
    # --image "R:\projects\street-sign-detector\data\GTSRB\Final_Test\Images\00042.ppm" \
    # --model "R:\projects\street-sign-detector\nn_pipeline\simple_cnn.pth" \
    # --pred_png_dir "R:\projects\street-sign-detector\output\nn_pipeline\pred_png" \
    # --pred_csv_dir "R:\projects\street-sign-detector\output\nn_pipeline\pred_csv"
    parser = argparse.ArgumentParser(description="Detect street sign bounding box using a neural network pipeline.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--model", required=True, help="Path to the trained PyTorch model (.pth)")
    parser.add_argument("--pred_png_dir", help="Directory to save annotated PNG images (optional)")
    parser.add_argument("--pred_csv_dir", help="Directory to save prediction CSV files (optional)")
    args = parser.parse_args()

    main(
        image_path=args.image,
        model_path=args.model,
        pred_png_dir=args.pred_png_dir,
        pred_csv_dir=args.pred_csv_dir
    )
