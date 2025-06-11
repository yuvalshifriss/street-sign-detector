import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_iou(boxA, boxB):
    # Convert to (x1, y1, x2, y2)
    ax1, ay1, aw, ah = boxA
    ax2, ay2 = ax1 + aw, ay1 + ah

    bx1, by1, bw, bh = boxB
    bx2, by2 = bx1 + bw, by1 + bh

    # Compute intersection
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    areaA = aw * ah
    areaB = bw * bh
    union = areaA + areaB - intersection

    return intersection / union

def load_ground_truth(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    gt_dict = {}
    for _, row in df.iterrows():
        box = (row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'] - row['Roi.X1'], row['Roi.Y2'] - row['Roi.Y1'])
        fname = row['Filename']
        if fname not in gt_dict:
            gt_dict[fname] = []
        gt_dict[fname].append(box)
    return gt_dict

def evaluate_predictions(predictions, ground_truth, iou_threshold=0.5):
    TP = 0
    FP = 0
    FN = 0

    for fname, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(fname, [])
        matched_gt = set()

        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= iou_threshold:
                TP += 1
                matched_gt.add(best_gt_idx)
            else:
                FP += 1

        # Any unmatched ground truth boxes are false negatives
        FN += len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def load_predictions(pred_csv):
    df = pd.read_csv(pred_csv)
    pred_dict = {}
    for _, row in df.iterrows():
        box = (row['x'], row['y'], row['w'], row['h'])
        fname = row['filename']
        if fname not in pred_dict:
            pred_dict[fname] = []
        pred_dict[fname].append(box)
    return pred_dict


def visualize_prediction_vs_ground_truth(image_dir, fname, preds, gts, save_path=None):
    image_path = os.path.join(image_dir, fname)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return

    # Draw ground truth boxes in RED
    for i, (x, y, w, h) in enumerate(gts):
        color = (0, 0, 255)  # red
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"GT {i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw predicted boxes in GREEN
    for i, (x, y, w, h) in enumerate(preds):
        color = (0, 255, 0)  # green
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"PR {i}", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Convert to RGB and show with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.title(fname)
    plt.axis('off')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    img_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.pardir, "data/GTSRB/Final_Test/Images/"))
    gt = load_ground_truth(os.path.join(img_dir, "GT-final_test.test.csv"))
    predictions = load_predictions(r'R:\\projects\\street-sign-detector\\output\\classical_pipeline\\00000.csv')
    metrics = evaluate_predictions(predictions, gt)
    print(metrics)
    visualize_prediction_vs_ground_truth(
        image_dir=img_dir,
        fname="00000.ppm",
        preds=predictions.get("00000.ppm", []),
        gts=gt.get("00000.ppm", []),
        save_path=None  # or specify a path to save the visualization
    )


