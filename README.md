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

# 🧰 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

# 📊 Data Source

The data used in this project comes from the [GTSRB – German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) hosted on Kaggle.

This dataset contains thousands of labeled images of German traffic signs, divided into:

- `Final_Training/Images/` – used to train the neural network model. Each class has its own subdirectory with images and a `GT-XXXX.csv` annotation file containing bounding boxes.
- `Final_Test/Images/` – used for evaluating both the classical and neural network pipelines. Includes images and a ground truth CSV file `GT-final_test.test.csv` with bounding box annotations.

All evaluations in this project are performed on the **Final Test set**, using **precision**, **recall**, and **F1 score**, computed via **IoU-based matching** between predicted and ground truth bounding boxes.

---

## 🧪 Classical Pipeline

### How it Works

The classical pipeline performs detection in several steps:

1. **Color filtering**: isolates red, blue, and white regions in the image (common street sign colors).
2. **Edge detection**: applies Canny edge detection to extract boundaries.
3. **Contour extraction**: finds contours and filters out small areas.
4. **Bounding box estimation**: returns bounding boxes around candidate contours.

### Parameter: `min_area`

To reduce false positives from noise, a **minimum area threshold** is applied. 

### How to Run

To run the classical pipeline on a single image:

```bash
python classical_pipeline/run_classical_pipeline.py \
  --image data/GTSRB/Final_Test/Images/00042.ppm \
  --pred_png_dir output/classical_pipeline/pred_png \
  --pred_csv_dir output/classical_pipeline/pred_csv \
  --min_area 300

