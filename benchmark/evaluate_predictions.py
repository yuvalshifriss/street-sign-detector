import os
import argparse
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
import base64
import numpy as np
from torchvision import transforms
from PIL import Image

from nn_pipeline.simple_cnn import SimpleCNN


def compute_iou(boxA, boxB):
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


def load_ground_truth(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    gt_dict = {}
    for _, row in df.iterrows():
        box = (row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'] - row['Roi.X1'], row['Roi.Y2'] - row['Roi.Y1'])
        fname = row['Filename']
        gt_dict.setdefault(fname, []).append(box)
    return gt_dict


def load_predictions(pred_dir):
    pred_dict = {}
    for fname in os.listdir(pred_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(pred_dir, fname)
        df = pd.read_csv(fpath)

        if not set(['x', 'y', 'w', 'h']).issubset(df.columns):
            print(f"[WARNING] Skipping {fname}: missing expected columns.")
            continue

        base_img_name = fname.replace(".csv", ".ppm")
        boxes = [tuple(row) for row in df[['x', 'y', 'w', 'h']].to_numpy()]
        pred_dict[base_img_name] = boxes
    return pred_dict


def predict_with_model(model_path, image_dir):
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


def evaluate_predictions(predictions, ground_truth, iou_threshold=0.5):
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


def visualize_prediction_vs_ground_truth(image_dir, fname, preds, gts):
    image_path = os.path.join(image_dir, fname)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape

    fig = go.Figure()

    # Background image
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

    # Ground truth boxes (RED)
    for i, (x, y, w, h) in enumerate(gts):
        fig.add_shape(type="rect",
                      x0=x, y0=height - y,
                      x1=x + w, y1=height - (y + h),
                      line=dict(color="red"),
                      name=f"GT {i}")
        fig.add_trace(go.Scatter(
            x=[x], y=[height - y],
            text=[f"GT {i}"], mode="text",
            showlegend=False
        ))

    # Predicted boxes (GREEN)
    for i, (x, y, w, h) in enumerate(preds):
        fig.add_shape(type="rect",
                      x0=x, y0=height - y,
                      x1=x + w, y1=height - (y + h),
                      line=dict(color="lime"),
                      name=f"PR {i}")
        fig.add_trace(go.Scatter(
            x=[x], y=[height - (y + h)],
            text=[f"PR {i}"], mode="text",
            showlegend=False
        ))

    fig.update_layout(
        title_text=fname,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x"),
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.update_yaxes(autorange="reversed")
    fig.show(renderer="browser")


if __name__ == "__main__":
    # run example 1:
    # --pred_dir
    # "R:\projects\street-sign-detector\output\classical_pipeline\pred_csv"
    # --ground_truth
    # "R:\projects\street-sign-detector\data\GTSRB\Final_Test\Images\GT-final_test.test.csv"
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

    # Load ground truth
    ground_truth = load_ground_truth(args.ground_truth)

    # Load predictions (either from CSV or NN model)
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
        print(f"{k}: {v}")

    # Save metrics if using CSV-based predictions
    if args.pred_dir:
        metrics_path = os.path.join(args.pred_dir, "evaluation_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"\nSaved evaluation metrics to {metrics_path}")

    # Visualization (optional)
    if args.image_dir:
        for fname in tqdm(list(ground_truth.keys())[:10], desc="Visualizing"):
            preds = predictions.get(fname, [])
            gts = ground_truth[fname]
            visualize_prediction_vs_ground_truth(args.image_dir, fname, preds, gts)
