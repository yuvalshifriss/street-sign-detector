# 🛑 Street Sign Detector

This project implements and compares two pipelines for refining bounding boxes around traffic signs in images from the GTSRB dataset:

✅ A Classical Computer Vision Pipeline using OpenCV with color filtering and edge-based heuristics

🧠 A Neural Network Pipeline using PyTorch to directly regress bounding box coordinates

The [GTSRB dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) (German Traffic Sign Recognition Benchmark) dataset consists of **images, each containing exactly one traffic sign, along with ground-truth bounding boxes around those signs**. The goal is not general object detection, but rather **precisely estimating the location and size of the sign within the image.**

The two pipelines are evaluated on the Final Test set using precision, recall, and F1 score, with IoU-based matching to determine true positives.

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

### Run on Single Image

To run the classical pipeline on a single image:

```bash
python classical_pipeline/run_classical_pipeline.py 
  --image data/GTSRB/Final_Test/Images/00042.ppm 
  --pred_png_dir output/classical_pipeline/pred_png 
  --pred_csv_dir output/classical_pipeline/pred_csv 
  --min_area 300
```
![classical](https://github.com/user-attachments/assets/e34365be-a190-431d-abce-0ba042a162c8)


### Run on Full Test Set
```bash
python benchmark/benchmark_classical_pipeline.py
```
This will evaluate the model predictions against ground truth annotations and compute metrics such as precision, recall, and F1 score.

* The predictions are saved to CSV files in:
output/classical_pipeline/pred_csv/

* If image_dir is specified in run_single_benchmark() (commented out by default), annotated PNG images with bounding boxes will also be saved to:
output/classical_pipeline/pred_csv_png/

### 🎯 Choosing the min_area Parameter
To optimize the min_area threshold, we ran:
```bash
python benchmark/benchmark_classical_pipeline_min_area.py
```

This evaluated multiple min_area values (25, 50, 75, ..., 300).
Below is the result summary:

| min\_area | Precision | Recall    | F1 Score  |
| --------- | --------- | --------- | --------- |
| 300       | 0.702     | 0.138     | 0.231     |
| 150       | 0.646     | 0.184     | 0.286     |
| 100       | 0.596     | 0.207     | 0.308     |
| 75        | 0.564     | 0.228     | 0.325     |
| 50        | 0.506     | 0.273     | 0.355     |
| **25**    | **0.398** | **0.388** | **0.393** |

➡️ We chose min_area = 25 because it achieved the highest F1 score, offering the best trade-off between precision and recall.
![image](https://github.com/user-attachments/assets/cd6e2391-bef3-4819-9619-948561ee72de)

Download the HTML plot from [here](output/classical_pipeline_different_min_area.html)

## 🧠 Neural Network Pipeline
### How it works
A small CNN model is trained to directly regress the bounding box (x, y, w, h) for each image. Trained using MSE loss over the annotated GTSRB training data and evaluated against the Final Test set.

### Architecture  
| Layer (type) | Output Shape     | Param # |
|--------------|------------------|---------|
| Conv2d-1     | [-1, 16, 48, 48] | 1,216   |
| ReLU-2       | [-1, 16, 48, 48] | 0       |
| MaxPool2d-3  | [-1, 16, 24, 24] | 0       |
| Conv2d-4     | [-1, 32, 24, 24] | 4,640   |
| ReLU-5       | [-1, 32, 24, 24] | 0       |
| MaxPool2d-6  | [-1, 32, 12, 12] | 0       |
| Conv2d-7     | [-1, 64, 12, 12] | 18,496  |
| ReLU-8       | [-1, 64, 12, 12] | 0       |
| MaxPool2d-9  | [-1, 64, 6, 6]   | 0       |
| Flatten-10   | [-1, 2304]       | 0       |
| Linear-11    | [-1, 128]        | 295,040 |
| ReLU-12      | [-1, 128]        | 0       |
| Dropout-13   | [-1, 128]        | 0       |
| Linear-14    | [-1, 4]          | 516     |
| **Total**    |                  | **319,908** |

Trainable params: 319,908
Non-trainable params: 0

### Train the Model
```bash
python nn_pipeline/train_and_validate.py 
  --image_root data/GTSRB/Final_Training/Images 
  --epochs 20 
  --batch_size 64 
  --lr 0.001 
  --output_model nn_pipeline/simple_cnn.pth
```
Outputs:
* simple_cnn.pth: the trained model
* simple_cnn_losses.csv: loss log
* simple_cnn_loss_plot.html: interactive loss visualization

![image](https://github.com/user-attachments/assets/8fca5f3a-ecfa-4843-96ae-a06afc24bcfd)
Download the HTML plot from [here](nn_pipeline/simple_cnn_loss_plot.html)

### Run on Single Image
```bash
python nn_pipeline/run_nn_pipeline.py 
  --image data/GTSRB/Final_Test/Images/00042.ppm 
  --model nn_pipeline/simple_cnn.pth 
  --pred_png_dir output/nn_pipeline/pred_csv_png 
  --pred_csv_dir output/nn_pipeline/pred_csv
```

![nn](https://github.com/user-attachments/assets/db7fa194-2f62-45fe-944a-268cc514b030)

Download the HTML plot from [here](nn_pipeline/simple_cnn_loss_plot.html)


### Benchmark Entire Test Set
```bash
python benchmark/benchmark_nn_pipeline.py
```
This will evaluate the model predictions against ground truth annotations and compute metrics such as precision, recall, and F1 score.

* The predictions are saved to CSV files in:
output/nn_pipeline/pred_csv/

* If image_dir is specified in run_single_benchmark() (commented out by default), annotated PNG images with bounding boxes will also be saved to:
output/nn_pipeline/pred_csv_png/


### 📈 Comparison of Pipelines: Classical vs Neural Network Pipeline
```bash
python benchmark/compare_nn_classical_pipeline.py
```
Outputs:
compare_pipelines.html: Interactive Plotly bar chart of metrics

![image](https://github.com/user-attachments/assets/8210f795-1064-49ef-b2fa-520bca62bfdf)

Download the HTML plot from [here](output/compare_pipelines.html)

| Metric     | Classical | Neural Network |
|------------|-----------|----------------|
| TP         | 1744      | 10079          |
| FP         | 739       | 2551           |
| FN         | 10886     | 2551           |
| Precision  | 0.702     | 0.798          |
| Recall     | 0.138     | 0.798          |
| F1 Score   | 0.231     | 0.798          |

The results show a stark contrast between the classical and neural network pipelines:

* Precision: Both methods achieve high precision, but the neural network slightly outperforms with 0.798 vs 0.702. This means that when either method predicts a sign, it is often correct—but the neural network is even more reliable in this regard.

* Recall: This is where the classical pipeline falls short. With a recall of only 0.138, it misses most of the actual traffic signs, whereas the neural network captures nearly all of them (recall = 0.798).

* F1 Score: As a harmonic mean of precision and recall, the F1 score summarizes the overall effectiveness. The neural network achieves a very strong F1 score of 0.798, compared to only 0.231 for the classical method.

Conclusion:
While the classical approach is conservative and relatively precise, it misses a vast majority of signs. The neural network is both precise and highly comprehensive in detection, making it the clearly superior option for traffic sign detection in this project.

