# 🛑 Street Sign Detector

This project implements and compares two pipelines for detecting street signs in images:

- ✅ A **Classical Computer Vision Pipeline** using OpenCV and color/edge-based techniques
- 🧠 A **Neural Network (CNN) Pipeline** using PyTorch for direct bounding box regression

It benchmarks both approaches on the [GTSRB dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign), evaluating performance on the Final Test set using precision, recall, and F1 score, with IoU-based matching to determine true positives.

---

## 📂 Project Structure
```text
street-sign-detector/
│
├── benchmark/
│   ├── benchmark_classical_pipeline.py
│   ├── benchmark_classical_pipeline_min_area.py
│   ├── benchmark_nn_pipeline.py
│   ├── compare_nn_classical_pipeline.py
│   └── evaluate_predictions.py
│
├── classical_pipeline/
│   ├── candidate_detector.py
│   ├── color_filtering.py
│   ├── edge_and_contours.py
│   └── run_classical_pipeline.py
│
├── nn_pipeline/
│   ├── run_nn_pipeline.py
│   ├── simple_cnn.py
│   ├── simple_cnn.pth
│   ├── simple_cnn_loss_plot.html
│   ├── simple_cnn_losses.csv
│   └── train_and_validate.py
│
├── output/
│   ├── classical_pipeline_different_min_area.csv
│   ├── classical_pipeline_different_min_area.html
│   ├── compare_pipelines.html
│   │
│   ├── classical_pipeline/
│   │   ├── pred_csv/
│   │   ├── pred_csv_png/
│   │   ├── pred_csv_min_area_25/
│   │   ├── pred_csv_min_area_25_png/
│   │   ├── pred_csv_min_area_50/
│   │   ├── pred_csv_min_area_50_png/
│   │   ├── pred_csv_min_area_75/
│   │   ├── pred_csv_min_area_75_png/
│   │   ├── pred_csv_min_area_100/
│   │   ├── pred_csv_min_area_100_png/
│   │   ├── pred_csv_min_area_150/
│   │   ├── pred_csv_min_area_150_png/
│   │   ├── pred_csv_min_area_300/
│   │   └── pred_csv_min_area_300_png/
│   │
│   └── nn_pipeline/
│       ├── pred_csv/
│       └── pred_csv_png/
│
└── data/
    └── GTSRB/
        ├── Final_Test/
        │   └── Images/
        │       ├── 00000...ppm
        │       └── GT-final_test.test.csv
        └── Final_Training/
            └── Images/
                ├── 00000.../
                │   ├── 00000_00000.ppm
                │   └── GT-00000.csv
```



