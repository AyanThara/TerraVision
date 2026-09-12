# TerraVision 

**AI-Powered Satellite Land-Use, Change & Risk Monitoring**

TerraVision is an AI-powered geospatial intelligence project designed to analyze satellite imagery, understand land use, detect changes over time, quantify those changes, and eventually provide trend, prediction, and risk insights through an interactive web platform.

The project combines deep learning, satellite image analysis, and geospatial analytics into an end-to-end monitoring pipeline.

---

##  Project Pipeline

```text
Satellite Imagery
       ↓
Land-Use Classification
       ↓
Pixel-Level Segmentation
       ↓
Bi-Temporal Change Detection
       ↓
Change Quantification
       ↓
Historical Trends
       ↓
Future Prediction
       ↓
Risk Intelligence
       ↓
Interactive Web Platform
```

### Current Progress

| Phase    | Component                     | Status     |
| -------- | ----------------------------- | ---------- |
| Phase 1  | Data Pipeline & Preprocessing | ✅ Complete |
| Phase 2  | Land-Use Classification       | ✅ Complete |
| Phase 3  | Land-Cover Segmentation       | ✅ Complete |
| Phase 4  | Bi-Temporal Change Detection  | ✅ Complete |
| Phase 5  | Change Quantification         | ✅ Complete |
| Phase 6  | Historical Trends             | ✅ Complete |
| Phase 7  | Future Prediction             | ⏳ Planned  |
| Phase 8  | Risk Intelligence             | ⏳ Planned  |
| Phase 9  | Global Scaling                | ⏳ Planned  |
| Phase 10 | Web Platform                  | ⏳ Planned  |

---

## 🧠 Phase 2 — Land-Use Classification

Dataset: **EuroSAT RGB**

* 27,000 RGB satellite images
* 10 land-use classes
* Image resolution: 64×64
* Model: Custom SimpleCNN
* Trainable parameters: ~102K
* Optimizer: Adam
* Loss: Cross-Entropy
* Hardware: Apple Silicon MPS

### Test Results

**Test Accuracy: 89.22%**

Macro metrics:

* Precision: **88.80%**
* Recall: **88.86%**
* F1 Score: **88.75%**

Best-performing classes included Residential and Forest, while classes such as Highway, River, and PermanentCrop were more challenging.

---

## 🗺️ Phase 3 — Land-Cover Segmentation

Dataset: **DeepGlobe Land Cover**

The segmentation pipeline performs pixel-level land-cover prediction using a lightweight U-Net architecture.

### Model

* Architecture: Lightweight U-Net
* Parameters: **~7.76M**
* Input: 256×256 RGB images
* Output: 7 land-cover classes
* Hardware: Apple Silicon MPS

### Prototype Results

* Test Pixel Accuracy: **57.17%**
* Mean IoU: **17.19%**
* Mean Dice/F1: **23.29%**

The current model demonstrates a functional end-to-end segmentation pipeline but remains a prototype and requires further training/data optimization for high-quality production segmentation.

---

## 🛰️ Phase 4 — Bi-Temporal Change Detection

Dataset: **OSCD (OGC Sentinel-2 Change Detection Benchmark)**

TerraVision uses genuine bi-temporal satellite observations of the same geographic regions captured at different timestamps.

### Architecture

```text
Image A (Before) ──┐
                   ├──→ 6-Channel Siam-UNet
Image B (After) ───┘
                           ↓
                    Binary Change Mask
```

Model:

* Early-Fusion Siam-UNet
* ~7.76M parameters
* 256×256 input
* 6-channel input (RGB + RGB)
* Binary change output

Dataset subset:

* 122 training pairs
* 50 test pairs

### Prototype Results

* Pixel Accuracy: **83.74%**
* Change IoU: **9.70%**
* Change F1/Dice: **17.69%**

The lightweight experiment successfully demonstrates the complete bi-temporal change-detection pipeline. The current model is a prototype and can be improved with additional training and class-imbalance optimization.

---

## 📏 Phase 5 — Change Quantification

The predicted change mask is converted into quantitative information.

For the evaluated sample:

* Total pixels: **65,536**
* Changed pixels: **3,645**
* Unchanged pixels: **61,891**
* Change percentage: **5.56%**
* Unchanged percentage: **94.44%**
* Change intensity: **Moderate**

Using the Sentinel-2 10 m ground sampling assumption documented in the pipeline:

* Estimated changed area: **364,500 m²**
* Estimated changed area: **36.45 hectares**

Pixel-level measurements are treated as the authoritative result, while physical area is explicitly treated as a spatial estimate.

---

## 🧩 Project Structure

```text
TerraVision/
│
├── data/
│   └── datasets and cached data
│
├── notebooks/
│   ├── inspect_dataset.py
│   ├── verify_pipeline.py
│   ├── verify_segmentation_pipeline.py
│   └── verify_change_pipeline.py
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   │
│   ├── dataset_segmentation.py
│   ├── segmentation_model.py
│   ├── train_segmentation.py
│   ├── evaluate_segmentation.py
│   │
│   ├── dataset_change_detection.py
│   ├── change_detection_model.py
│   ├── train_change_detection.py
│   ├── evaluate_change_detection.py
│   │
│   └── quantify_change.py
│
├── models/
│   └── local model checkpoints
│
├── results/
│   ├── classification results
│   ├── segmentation results
│   ├── change-detection results
│   └── quantification results
│
├── docs/
│   └── project documentation
│
└── README.md
```

---

## 🛠️ Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Hugging Face Datasets
* Git & GitHub
* Apple Silicon MPS

---

## 📊 Datasets

### EuroSAT RGB

Used for land-use classification.

### DeepGlobe Land Cover

Used for pixel-level land-cover segmentation.

### OSCD

Used for genuine bi-temporal satellite change detection.

---

## 🎯 Planned Features

Future TerraVision phases will focus on:

* 📈 Historical land-cover change trends
* 🔮 Future change prediction
* ⚠️ Geospatial risk intelligence
* 🏙️ Urban expansion monitoring
* 🌾 Agricultural land monitoring
* 🌲 Environmental change monitoring
* 🏗️ Infrastructure/development monitoring
* 🗺️ Interactive global satellite map
* 📊 Analytical dashboards
* 📝 Automated change explanations and reports

---

## 🌐 Planned Web Platform

The final system is planned as an interactive web application where users can select a geographic region and analyze satellite observations.

Planned interface:

```text
Location
   ↓
Satellite imagery
   ↓
Land-use layer
   ↓
Change layer
   ↓
Quantification
   ↓
Historical trends
   ↓
Prediction
   ↓
Risk intelligence
```

The frontend has **not yet been implemented**.

---

## 🔬 Current Project Status

TerraVision currently has a functioning machine-learning pipeline through **Phase 5: Change Quantification**.

The current results represent **prototype experiments**, not production-grade satellite intelligence. Future work will focus on improving model performance, expanding temporal analysis, adding prediction and risk intelligence, and integrating the pipeline into an interactive geospatial web application.

---

## 📌 Vision

> **Turn satellite imagery into actionable geospatial intelligence.**

TerraVision aims to move beyond simply viewing satellite imagery by automatically analyzing how land is being used, how it changes over time, how much it changes, and what those changes could mean for future development, agriculture, infrastructure, and environmental monitoring.
