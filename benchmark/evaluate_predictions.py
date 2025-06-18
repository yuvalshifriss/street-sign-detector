import os
import argparse
import pandas as pd
import torch
import cv2
import base64
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import plotly.graph_objects as go
from torchvision import transforms
from PIL import Image
from nn_pipeline.simple_cnn import SimpleCNN


def compute_iou(boxA: Tuple[float, float, float, float],
                boxB: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    ax1, ay1, aw, ah = boxA
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = boxB
    bx2, by2 = bx1 + bw, by1 + bh

    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = aw * ah + bw * bh - intersection
    return intersection / union


def load_ground_truth(csv_path: str) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """Load ground truth bounding boxes from CSV file."""
    df = pd.read_csv(csv_path, sep=';')
    gt_dict = {}
    for _, row in df.iterrows():
        box = (row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'] - row['Roi.X1'], row['Roi.Y2'] - row['Roi.Y1'])
        fname = row['Filename']
        gt_dict.setdefault(fname, []).append(box)
    return gt_dict


def load_predictions(pred_dir: str) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """Load predicted bounding boxes from directory of CSV files."""
    pred_dict = {}
    for fname in os.listdir(pred_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(pred_dir, fname)
        df = pd.read_csv(fpath)
        if not {'x', 'y', 'w', 'h'}.issubset(df.columns):
            print(f"[WARNING] Skipping {fname}: missing expected columns.")
            continue
        base_img_name = fname.replace(".csv", ".ppm")
        boxes = [tuple(row) for row in df[['x', 'y', 'w', 'h']].to_numpy()]
        pred_dict[base_img_name] = boxes
    return pred_dict


def predict_with_model(model_path: str, image_dir: str) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """Generate predictions from a trained model on a directory of images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    predictions = {}
    for fname in tqdm(os.listdir(image_dir), desc="Predicting with model"):
        if not fname.endswith(".ppm"):
            continue
        img_path = os.path.join(image_dir, fname)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor).squeeze().cpu().numpy()
            x, y, w, h = output
            x, y = max(0, x), max(0, y)
            w, h = max(1, w), max(1, h)
            predictions[fname] = [(x, y, w, h)]
    return predictions


def evaluate_predictions(predictions: Dict[str, List[Tuple[float, float, float, float]]],
                         ground_truth: Dict[str, List[Tuple[int, int, int, int]]],
                         iou_threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate predictions using precision, recall, F1 score, and confusion metrics."""
    TP, FP, FN = 0, 0, 0
    for fname, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(fname, [])
        matched_gt = set()
        for pred in pred_boxes:
            best_iou, best_idx = 0, -1
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold:
                TP += 1
                matched_gt.add(best_idx)
            else:
                FP += 1
        FN += len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return dict(TP=TP, FP=FP, FN=FN, precision=precision, recall=recall, f1=f1)


def visualize_prediction_vs_ground_truth(image_dir: str,
                                         fname: str,
                                         preds: List[Tuple[float, float, float, float]],
                                         gts: List[Tuple[int, int, int, int]],
                                         pred_png: str) -> None:
    """Visualize predicted vs ground truth bounding boxes and save as HTML."""
    os.makedirs(pred_png, exist_ok=True)
    image_path = os.path.join(image_dir, fname)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape

    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{base64.b64encode(cv2.imencode('.png', image_rgb)[1]).decode()}",
            xref="x", yref="y",
            x=0, y=0,
            sizex=width, sizey=height,
            sizing="stretch",
            layer="below"
        )
    )

    for i, (x, y, w, h) in enumerate(gts):
        fig.add_shape(type="rect", x0=x, y0=height - y, x1=x + w, y1=height - (y + h),
                      line=dict(color="red"), name=f"GT {i}")
        fig.add_trace(go.Scatter(x=[x], y=[height - y], text=[f"GT {i}"], mode="text", showlegend=False))

    for i, (x, y, w, h) in enumerate(preds):
        fig.add_shape(type="rect", x0=x, y0=height - y, x1=x + w, y1=height - (y + h),
                      line=dict(color="lime"), name=f"PR {i}")
        fig.add_trace(go.Scatter(x=[x], y=[height - (y + h)], text=[f"PR {i}"], mode="text", showlegend=False))

    fig.update_layout(
        title_text=f"{fname} - Predicted (green) vs. GT (red)",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x"),
        width=width + 300,
        height=height + 300,
        margin=dict(l=30, r=30, t=100, b=30),
    )
    fig.update_yaxes(autorange="reversed")
    fig.write_html(os.path.join(pred_png, fname.replace('.ppm', '.html')))


if __name__ == "__main__":
    # run example 1:
    # --pred_dir
    # "R:\projects\street-sign-detector\output\classical_pipeline\pred_csv" --ground_truth
    # "R:\projects\street-sign-detector\data\GTSRB\Final_Test\Images\GT-final_test.test.csv" --image_dir
    # "R:\\projects\\street-sign-detector\\data\\GTSRB\\Final_Test\\Images"
    # run example 2:
    # --model "R:\projects\street-sign-detector\nn_pipeline\simple_cnn.pth"
    # --image_dir "R:\projects\street-sign-detector\data\GTSRB\Final_Test\Images"
    # --ground_truth "R:\projects\street-sign-detector\data\GTSRB\Final_Test\Images\GT-final_test.test.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", required=True, help="Path to GT-final_test.test.csv")
    parser.add_argument("--pred_dir", help="Directory with classical predictions (CSV format)")
    parser.add_argument("--model", help="Path to trained PyTorch model (for NN evaluation)")
    parser.add_argument("--image_dir", help="Optional: Directory containing images (required for model or visualization)")
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)

    if args.model:
        if not args.image_dir:
            raise ValueError("`--image_dir` must be provided when using `--model` for inference.")
        predictions = predict_with_model(args.model, args.image_dir)
    elif args.pred_dir:
        predictions = load_predictions(args.pred_dir)
    else:
        raise ValueError("Either --pred_dir or --model must be specified.")

    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth entries")

    metrics = evaluate_predictions(predictions, ground_truth)
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    if args.pred_dir:
        metrics_path = os.path.join(args.pred_dir, "evaluation_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"\nSaved evaluation metrics to {metrics_path}")

    if args.image_dir:
        for fname in tqdm(list(ground_truth.keys()), desc="Visualizing"):
            preds = predictions.get(fname, [])
            gts = ground_truth[fname]
            visualize_prediction_vs_ground_truth(args.image_dir, fname, preds, gts, f'{args.pred_dir}_png')
