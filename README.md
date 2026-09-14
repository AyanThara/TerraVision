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
| Phase 1  | Data Pipeline & Preprocessing | ✅ Completed |
| Phase 2  | Land-Use Classification       | ✅ Completed |
| Phase 3  | Land-Cover Segmentation       | ✅ Completed |
| Phase 4  | Bi-Temporal Change Detection  | ✅ Complete |
| Phase 5  | Change Quantification         | ✅ Complete |
| Phase 6  | Historical Trends             | ✅ Complete |
| Phase 7  | Future Prediction             | ✅ Complete |
| Phase 8  | Risk Intelligence             | ✅ Complete |
| Phase 9  | Global Scaling & Geospatial Integration | ✅ Complete |
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

## 📈 Phase 6 — Historical Change Trends

TerraVision connects dynamically to the **Microsoft Planetary Computer STAC API** to retrieve genuine multi-temporal **Sentinel-2 MSI Level-2A** (surface reflectance) observations over a fixed geographic region (Las Vegas / Henderson Urban Expansion Zone).

Pairwise change detection was performed across five consecutive observations spanning **2020–2024** (four consecutive multi-temporal intervals) using the Phase 4 Siam-UNet model checkpoint.

### Summary Metrics

* Sensor: **Sentinel-2 MSI (Level-2A BOA Surface Reflectance)**
* Historical Observations: **5 genuine STAC scenes (2020–2024)**
* Consecutive Periods: **4 intervals (P1: 2020–2021, P2: 2021–2022, P3: 2022–2023, P4: 2023–2024)**
* Mean Annual Detected Change Rate: **1.29 %/year**
* Mean Annual Detected Change Area: **8.49 ha/year**
* Overall Trend: **Decreasing** (OLS slope = **-0.7336 %/year²**)
* Cumulative Detected Change: **33.62 hectares** (5.13% of ROI)

*Important Scientific Limitation:* Phase 6 outputs are model-detected historical change estimates from the Siam-UNet prototype applied to satellite imagery and are not independently ground-truth validated for every historical observation. Cumulative change is strictly cumulative detected change across periods, not unique physical land transformed.

---

## 🔮 Phase 7 — Future Prediction

Phase 7 evaluates the historical time series to project future model-detected change using a lightweight, mathematically defensible statistical approach.

Given the small sample size ($N = 4$ observations, leaving $df = N - 2 = 2$ degrees of freedom), complex machine-learning models or high-order polynomials would overfit. A first-order Ordinary Least Squares (OLS) linear trend extrapolation with analytical Student's $t$ prediction intervals and a physical non-negativity constraint was implemented.

### Forecast Model & Projections

* Method: **First-order OLS trend extrapolation with physical non-negativity constraint**
* Degrees of Freedom: **$df = 2$ ($N = 4$ historical observations)**
* Historical OLS Slope: **-0.7336 %/year²**
* Forecast Horizons: **2025, 2026, 2027**

| Horizon | Point Forecast (Bounded) | 90% Prediction Interval | 95% Prediction Interval | Projected Cumulative Change |
| :--- | :--- | :--- | :--- | :--- |
| **2025 ($T+1$)** | **0.00 %/year** (0.00 ha/yr) | 0.00–4.34 %/year | **0.00–6.62 %/year** | 33.62 ha |
| **2026 ($T+2$)** | **0.00 %/year** (0.00 ha/yr) | 0.00–4.67 %/year | **0.00–7.45 %/year** | 33.62 ha |
| **2027 ($T+3$)** | **0.00 %/year** (0.00 ha/yr) | 0.00–5.11 %/year | **0.00–8.46 %/year** | 33.62 ha |

### Key Methodological & Statistical Notes

* **Point Forecast at Zero:** The point forecast is 0.00 %/year because extrapolating the historical decreasing trend ($-0.7336\ \%/\text{year}^2$) yields negative values, which are physically bounded at zero since land change rates cannot be negative.
* **Wide Prediction Intervals:** Prediction intervals are wide ($0\text{--}6.62\%$ for 2025, $0\text{--}8.46\%$ for 2027) because they are calculated using Student's $t$ critical values with only 2 degrees of freedom ($t_{2, 0.95} \approx 4.303$), properly reflecting small-sample epistemic uncertainty rather than false precision.
* **Forecast of Model-Detected Change:** Phase 7 forecasts model-detected change estimates, NOT guaranteed real-world land conversion or development.
* **Cumulative Metric Distinction:** Projected cumulative detected change measures cumulative detected change events across temporal intervals, NOT unique physical land transformed.

---

## ⚠️ Phase 8 — Risk Intelligence

Phase 8 synthesizes the completed outputs from Phase 5 (Change Quantification), Phase 6 (Historical Trends), and Phase 7 (Future Prediction) into a structured, transparent **Analytical Risk Indicator**.

The engine computes three orthogonal sub-scores and pairs them with an independent **Evidence Confidence Score** reflecting input data quality and degrees of freedom ($df=2$).

### Risk & Evidence Confidence Assessment

* **Overall Analytical Risk Indicator:** **34.32 / 100**
* **Risk Level:** **`MODERATE`** ($30.0 \le R < 60.0$)
* **Evidence Confidence Score:** **37.74 / 100**
* **Evidence Confidence Level:** **`LOW CONFIDENCE`** ($< 40.0$, small sample penalty)

### Sub-Score Multi-Factor Breakdown

| Dimension | Sub-Score | Weight | Key Indicator / Inputs | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Change Pressure ($S_{\text{pressure}}$)** | **36.82 / 100** | 40% | Cumulative: 5.13% ROI; Recent: 0.67 %/yr | Moderate historical transformation volume, mitigated by subdued recent velocity. |
| **Trend Momentum ($S_{\text{trend}}$)** | **25.55 / 100** | 30% | OLS Slope: $-0.7336\ \%/\text{year}^2$ | Decelerating momentum; transition pace has sharply declined from 2021 peak. |
| **Future Outlook ($S_{\text{forecast}}$)** | **39.72 / 100** | 30% | Point: 0.0 %/yr; 95% Bound: 43.39 ha/yr | Bounded point expectation is zero, but precautionary upper bound indicates tail exposure. |

### Key Methodological & Scientific Notes

* **Analytical Risk Indicator:** The risk score measures model-detected surface transformation pressure and statistical forecast dispersion. It does NOT prove ground-truth legal zoning, entitlement, or guaranteed real-world construction.
* **Evidence Confidence Score vs. Statistical Intervals:** The Evidence Confidence Score ($37.74 / 100$, `LOW CONFIDENCE`) is an engineering/analytical data-adequacy index. It must NOT be confused with formal statistical confidence intervals, coverage probabilities, or p-values.
* **Calibration Benchmarks:** The normalization thresholds ($10\%$ cumulative change, $3.0\ \%/\text{yr}$ velocity, $10.0\ \%/\text{yr}$ stress test) are explicitly defined as **TerraVision heuristic normalization benchmarks**—project-defined calibration constants for prototype scaling, not universally validated geospatial thresholds.

---

## 🌐 Phase 9 — Global Scaling & Geospatial Integration

Phase 9 transitions TerraVision from the single fixed demonstration site used in Phase 6 (Las Vegas / Henderson) into a production-ready, modular **Geospatial Pipeline** capable of accepting arbitrary geographic coordinates and bounding boxes worldwide.

The pipeline connects dynamically to the **Microsoft Planetary Computer STAC API** to discover and download genuine **Sentinel-2 MSI Level-2A** (surface reflectance) observations, enforces strict preprocessing standards matching the frozen Phase 4 Siam-UNet model, and unifies the full downstream analytical stack (Phase 5 Quantification, Phase 6 Trends, Phase 7 Predictions, and Phase 8 Risk Intelligence) without duplicate logic.

### Core Architectural Specifications

* **Geographic Input:** Supports arbitrary Latitude/Longitude points (with geodesic cosine latitude scaling) and explicit WGS84 bounding boxes `[min_lon, min_lat, max_lon, max_lat]`.
* **Territorial Envelope:** Enforces Sentinel-2 systematic acquisition limits ($-56^\circ \le \text{lat} \le +84^\circ$), rejecting polar ice sheets and open high seas.
* **Spatial Footprint Contract:**
  * **Standardized Patch:** Exactly $256 \times 256$ pixels.
  * **Nominal Spatial Resolution:** Sentinel-2 native 10 m Ground Sample Distance (GSD).
  * **Nominal Footprint:** Approximately $2.56\text{ km} \times 2.56\text{ km}$ ($655.36\text{ ha}$).
  * **Actual Footprint Calculation:** Dynamic calculation of actual ground width, height, and area using a **Haversine-based geographic footprint estimate** (spherical WGS84 mean radius), recorded in metadata alongside effective per-pixel GSD.
* **Temporal Capability Gate:**
  * **Mode A (Bi-Temporal, $N=2$ observations, $1$ interval):** Supports bi-temporal change detection, change quantification, and pairwise annualized rate. **Strictly gates out** historical trend regression, Student's t prediction intervals, and risk intelligence because $df = N - 2 = 0$. Intermediate observations are never invented or fabricated.
  * **Mode B (Multi-Temporal, $N \ge 4$ observations, $\ge 3$ intervals):** Enables defensible OLS trend regression ($df \ge 1$), analytical Student's t intervals, future prediction, and multi-factor risk intelligence.
* **Live STAC Discovery & Cloud Fallback:**
  * Searches `sentinel-2-l2a` Level-2A BOA surface reflectance.
  * Multi-tier progressive cloud filtering: queries strict threshold ($< 5\%$), adaptively relaxing up to $15\%$ with diagnostic metadata warnings if required.
* **Project Engineering Constraints:**
  * Default search tolerance: **$\pm 30$ days**.
  * Minimum temporal baseline: **$\ge 15$ days** between observations.
  * Explicitly documented as TerraVision project-defined engineering/design constraints, NOT universal scientific thresholds.
* **Preprocessing Audit & Model Compatibility:**
  * Strictly preserves Phase 4 audited training transformations: PIL RGB $\to$ `ToTensor()` ($[0, 255] \to [0.0, 1.0]$) $\to$ ImageNet `Normalize` ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  * Ingests a 6-channel early-fusion tensor $(1, 6, 256, 256)$ into the frozen Phase 4 Siam-UNet checkpoint (`7,763,905` parameters, `requires_grad=False`).

### Global Geographic Benchmark / Integration Test Locations

The pipeline defines five diverse geographic integration test locations representing distinct global biomes:

| Benchmark Key | Location Name | Biome / Setting | Bounding Box (WGS84) |
| :--- | :--- | :--- | :--- |
| `las_vegas` | Las Vegas / Henderson, USA | Arid / Desert Urban Expansion (Phase 6 Baseline) | `[-115.120, 36.000, -115.090, 36.030]` |
| `dubai` | Dubai South Logistics Corridor, UAE | Hyper-Arid / Megacity Infrastructure & Grading | `[55.150, 24.960, 55.175, 24.985]` |
| `rondonia` | Rondônia Agricultural Frontier, Brazil | Tropical Rainforest / Canopy Clearing | `[-62.900, -10.200, -62.875, -10.175]` |
| `munich` | Munich North Expansion, Germany | Temperate Europe / Commercial & Farmland Transition | `[11.560, 48.180, 11.585, 48.205]` |
| `lake_mead` | Lake Mead Shoreline & Marina, USA | Hydrological / Reservoir Shoreline Recession | `[-114.420, 36.010, -114.395, 36.035]` |

*Note: These benchmarks serve as geographic integration test environments; they are NOT claimed as ground-truth validated ML benchmark test sets.*

### Benchmark Demonstration Results (Dubai South Corridor, Mode A: N=2)

* **Temporal Baseline:** 2021-08-06 (Sentinel-2B, cloud: 0.66%) $\to$ 2024-07-26 (Sentinel-2A, cloud: 0.71%)
* **Temporal Separation:** 1,085 days (2.97 years)
* **Nominal Footprint:** $256 \times 256$ pixels @ 10 m GSD ($655.36\text{ ha}$)
* **Actual Geographic Footprint:** **$2,520.0\text{ m} \times 2,779.9\text{ m}$** ($700.52\text{ ha}$, Haversine-based estimate, Effective GSD: $9.84\text{ m} \times 10.86\text{ m}$)
* **Detected Change:** **$8.42\%$** ($5,518$ changed pixels out of $65,536$)
* **Nominal Changed Area:** **$55.18\text{ ha}$** (Annualized rate: $18.58\text{ ha/year}$, $2.83\ \%/\text{year}$)
* **Actual Changed Area:** **$58.98\text{ ha}$** (Annualized rate: $19.86\text{ ha/year}$)
* **Change Intensity:** **`Moderate`** ($2.0\% \le x < 10.0\%$)
* **Capability Gate Execution:** Mode A active ($N=2$). Change detection and quantification computed; historical trend regression, prediction intervals, and risk intelligence gated out.
* **Pipeline Runtime:** ~3.2 seconds end-to-end (including STAC search, parallel crop download, and MPS inference)

### Important Scientific Notes & Disclaimers

* **Decision Threshold & Noise:** The 0.5 probability threshold produced the reported change mask; interpretation remains subject to model limitations and requires independent validation.
* **Model-Detected Change vs. Real-World Conversion:** Model-detected change indicates optical surface reflectance discrepancies. It does NOT prove ground-truth legal land conversion, zoning entitlement, or verified real-world construction.
* **No Claim of Global Model Accuracy:** TerraVision does not claim validated global model accuracy. The model was trained on OSCD and evaluated across benchmark integration locations.
* **Authoritative Pixels vs. Spatial Estimates:** Pixel counts and percentages are the authoritative analytical measurements; physical areas are spatial approximations based on nominal 10 m GSD and Haversine-based geographic footprint estimates.
* **Radiometric Nuances:** 8-bit true-color visual renderings from live STAC endpoints have visual contrast adjustments that may influence model sensitivity compared to static benchmark datasets.

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
│   ├── verify_change_pipeline.py
│   ├── verify_historical_trends.py
│   ├── verify_future_prediction.py
│   ├── verify_risk_intelligence.py
│   └── verify_geospatial_pipeline.py
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
│   ├── quantify_change.py
│   ├── historical_trends.py
│   ├── future_prediction.py
│   ├── risk_intelligence.py
│   │
│   ├── geo_utils.py
│   ├── stac_client.py
│   └── geospatial_pipeline.py
│
├── models/
│   └── local model checkpoints
│
├── results/
│   ├── classification results
│   ├── segmentation results
│   ├── change-detection results
│   ├── quantification results
│   ├── historical_change_trends.json
│   ├── historical_change_trends.png
│   ├── future_prediction.json
│   ├── future_prediction.png
│   ├── risk_intelligence.json
│   ├── risk_intelligence.png
│   ├── global_geospatial_pipeline.json
│   └── global_geospatial_pipeline.png
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

* 🌍 Global scaling across diverse biomes (Phase 9)
* 🌐 Interactive geospatial web platform (Phase 10)
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

TerraVision currently has a functioning analytical/ML pipeline through **Phase 8: Risk Intelligence**.

The current results represent **prototype experiments**, not production-grade satellite intelligence. Future work will focus on improving model performance, global scaling (Phase 9), and integrating the pipeline into an interactive geospatial web application (Phase 10).

---

## 📌 Vision

> **Turn satellite imagery into actionable geospatial intelligence.**

TerraVision aims to move beyond simply viewing satellite imagery by automatically analyzing how land is being used, how it changes over time, how much it changes, and what those changes could mean for future development, agriculture, infrastructure, and environmental monitoring.
