# ⚡ Real-Time Thunderstorm Prediction System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost%20Classifier-00C49F.svg)](https://catboost.ai/)
[![MetPy](https://img.shields.io/badge/Meteorology-MetPy%20%7C%20Siphon-0088FE.svg)](https://unidata.github.io/MetPy/latest/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Operational Machine Learning System for 24-Hour Convective Hazard Forecasting Using Upper-Air Radiosonde Soundings, Derived Thermodynamic Indices, and Automated Wyoming Sounding Data Ingestion.**

---

## 📌 Repository About / Overview

The **Real-Time Thunderstorm Prediction System** is an end-to-end operational machine learning framework designed to classify and predict 24-hour thunderstorm occurrence for **Station 43150 (Visakhapatnam, India; 17.70° N, 83.30° E)**. 

By combining 25 annual cycles of radiosonde upper-air profiles (2000–2025) with ground-truth SYNOP present-weather reports, the system computes **28 atmospheric stability, moisture, and kinematic indices** via MetPy and predicts next-day severe weather risk using an Optuna-optimized **CatBoost Classifier**. The deployed Streamlit dashboard fetches live upper-air soundings via Siphon from the University of Wyoming database and maps raw probabilities into **7 calibrated operational risk tiers (35% to 95%)**.

---

## 🚀 Key Features

* **⚡ Real-Time Sounding Ingestion**: Automated live upper-air radiosonde retrieval for Station 43150 (00:00 & 12:00 UTC) via Siphon from the University of Wyoming database, with fallback to historical observations.
* **🔬 MetPy Diagnostic Engine**: Dynamic calculation of 28 atmospheric stability, thermodynamic, and kinematic indices (CAPE, PW, K-Index, Lifted Index, SWEAT, BRN, etc.).
* **🤖 Optimized Gradient-Boosted Classifier**: Deployed CatBoost model with ordered boosting and oblivious trees, natively handling missing sounding levels without imputation bias.
* **🎯 Recall-Constrained Threshold Calibration**: Custom decision threshold (.32$) optimized against the Critical Success Index (CSI) to prioritize early hazard warning over false alarm reduction.
* **📊 Operational Risk Scale**: Calibrated probability mapping converting raw model outputs into 7 actionable risk levels (*Very Low* to *Very High*).
* **🎨 Modern Executive Dashboard**: Streamlit interface displaying station information, thermodynamic index cards, model predictions, risk progress bars, interactive Plotly feature importances, and downloadable raw sounding data.

---

## 📊 Dataset & Ground Truth

| Parameter | Description |
| :--- | :--- |
| **Observation Station** | Station 43150 (Visakhapatnam, India; .70^\circ\text{ N}, 83.30^\circ\text{ E}$) |
| **Time Period** | 25 Years ( - 2025$), total 11,591 atmospheric soundings |
| **Data Sources** | 1. University of Wyoming Upper-Air Radiosonde Archive<br>2. SYNOP Present/Past Weather Reports (7wwW1W2 codes) |
| **Target Variable (Thunderstorm_24h)** | Binary indicator ($ = Thunderstorm observed within 24h, $ = Non-thunderstorm). Triggered by present weather ( \in [13, 17, 29, 91-99]$) or past weather (, W2 = 9$). |
| **Class Distribution** | Imbalanced: .78\%$ non-thunderstorm vs. .22\%$ thunderstorm soundings |

### 📅 Chronological Dataset Splitting (No Temporal Leakage)
To mirror operational forecasting reality and prevent temporal autocorrelation leakage:
* **Training Set (2000–2020)**: 9,619 soundings used for model parameter fitting.
* **Validation Set (2021–2022)**: 580 soundings reserved for hyperparameter tuning & threshold selection.
* **Testing Set (2023–2025)**: 1,392 soundings held out for final unbiased evaluation.

---

## 🌡️ Atmospheric Stability & Derived Features

The model utilizes **28 upper-air predictors** calculated using the MetPy atmospheric science library:

| Symbol | Parameter | Meteorological Significance |
| :--- | :--- | :--- |
| **PW** | Precipitable Water | Total column water vapour; primary moisture reservoir for deep convection. |
| **KI** | K-Index | Combines 850-500 hPa lapse rate and moisture at 850/700 hPa; signals air-mass storms. |
| **LI / LI_V** | Lifted Index (Standard & Virtual) | Environment minus parcel temperature at 500 hPa; negative values indicate instability. |
| **TT / CT / VT**| Totals Totals, Cross Totals, Vertical Totals | Static lapse rate & moisture contribution to convective instability. |
| **CAPE / CAPE_V**| Convective Available Potential Energy | Integrated positive buoyant energy between LFC and EL (updraft fuel). |
| **CIN / CIN_V** | Convective Inhibition | Energy barrier suppressing parcel ascent before reaching LFC. |
| **SWEAT** | Severe Weather Threat Index | Combines low/mid-level shear, wind speed, and moisture for severe convection. |
| **BRN / BRN_V** | Bulk Richardson Number | Ratio of CAPE to vertical wind shear; identifies storm organization mode. |
| **LCL_P / LCL_T**| Lifted Condensation Level | Pressure and temperature at which a rising parcel saturates (cloud base). |
| **THETAE_LCL** | Equivalent Potential Temp at LCL | Conserved moist thermodynamic potential temperature. |
| **THK_1000_500**| 1000–500 hPa Thickness | Geopotential thickness proportional to mean-layer virtual temperature. |

---

## ⚙️ Machine Learning Pipeline & Methodology

`
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Raw Radiosonde  │ ──►│ MetPy Diagnostic│ ──►│ Chronological   │
│ & SYNOP Data    │    │ Index Generator │    │ Train/Val/Test  │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐             ▼
│ Operational     │ ──►│ Threshold Search│ ◄─── ┌─────────────────┐
│ Streamlit App   │    │ (Max CSI @ 0.70)│      │ Optuna Hyper-   │
└─────────────────┘    └─────────────────┘      │ parameter Tuning│
                                                └─────────────────┘
`

1. **Missing Value Handling**: Incomplete soundings (e.g., terminating before EL) were preserved natively by CatBoost rather than imputed, as missingness itself carries physical signal (storm-top height / ascent ceiling).
2. **Multicollinearity**: All 28 features retained; tree ensembles natively manage correlated feature clusters (e.g., CAPE, KI, SWEAT).
3. **Hyperparameter Tuning**: Optuna (Tree-structured Parzen Estimator) search optimizing validation F1/AUC with uto_class_weights='Balanced'.
4. **Threshold Selection**: Swept decision thresholds (.20 - 0.80$) on validation data to maximize Critical Success Index (CSI) subject to a minimum recall constraint of .70$, selecting **.32$** as the operating threshold.

---

## 📈 Model Performance & Comparative Evaluation

Evaluated on the held-out **2023–2025 test set** (1,392 unseen soundings):

### 🏆 Final Model Evaluation (CatBoost @ 0.32 Threshold)

`
Test Classification Report (Held-out 2023-2025):
=================================================
               precision    recall  f1-score   support
No Thunderstorm     0.94      0.80      0.87      1219
   Thunderstorm     0.32      0.65      0.43       173

       Accuracy : 0.790
       ROC AUC  : 0.799
       Recall   : 0.653  (113 / 173 thunderstorms detected)
       CSI      : 0.275
       FAR      : 0.678
`

### 🎯 Confusion Matrix (Test Set: 2023–2025)
* **True Negatives (TN)**: 981
* **False Positives (FP)**: 238
* **False Negatives (FN)**: 60
* **True Positives (TP)**: 113

### ⚔️ Comparative Models Summary

| Model / Ensemble Strategy | ROC AUC | Recall (POD) | Precision | CSI (Threat Score) | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (Selected Operational)** | **0.799** | **0.653** | **0.322** | **0.275** | **0.431** |
| LightGBM | 0.805 | 0.613 | 0.307 | 0.260 | 0.409 |
| XGBoost | 0.807 | 0.630 | 0.313 | 0.265 | 0.418 |
| Equal Voting Ensemble | 0.805 | 0.589 | 0.326 | 0.265 | 0.420 |
| Weighted Ensemble (0.1 CB, 0.35 LGB, 0.55 XGB) | 0.807 | 0.503 | 0.330 | 0.249 | 0.398 |
| Stacking Ensemble (Logistic Regression Meta) | 0.805 | 0.555 | 0.328 | 0.258 | 0.412 |

> **Why CatBoost was selected**: CatBoost achieved the highest operational hazard detection recall (.3\%$) and Critical Success Index (.275$) among all individual models and ensembles. In severe weather forecasting, missed events (false negatives) carry significantly higher risk than false alarms.

---

## 🚦 Operational Probability Thresholds

Raw model probabilities are mapped into 7 operational risk tiers based on atmospheric instability thresholds:

| Operational Probability | Risk Category | Key Index Threshold Highlights |
| :---: | :--- | :--- |
| **95%** | **Very High** | $\text{PW} \ge 60.5\text{ mm}$, $\text{KI} \ge 35.6$, $\text{CAPE\_V} \ge 2325\text{ J/kg}$, $\text{SWEAT} \ge 223.4$ |
| **85%** | **High** | $\text{PW} \ge 55.4\text{ mm}$, $\text{KI} \ge 34.1$, $\text{CAPE\_V} \ge 2196\text{ J/kg}$, $\text{SWEAT} \ge 216.2$ |
| **75%** | **Moderately High**| $\text{PW} \ge 50.6\text{ mm}$, $\text{KI} \ge 32.9$, $\text{CAPE\_V} \ge 1799\text{ J/kg}$, $\text{SWEAT} \ge 215.8$ |
| **65%** | **Moderate** | $\text{PW} \ge 41.6\text{ mm}$, $\text{KI} \ge 26.8$, $\text{CAPE\_V} \ge 1541\text{ J/kg}$, $\text{SWEAT} \ge 157.5$ |
| **55%** | **Moderately Low** | $\text{PW} \ge 29.4\text{ mm}$, $\text{KI} \ge 16.4$, $\text{CAPE\_V} \ge 977\text{ J/kg}$, $\text{SWEAT} \ge 83.4$ |
| **45%** | **Low** | $\text{PW} \ge 22.5\text{ mm}$, $\text{KI} \ge -10.6$, $\text{CAPE\_V} \ge 56\text{ J/kg}$, $\text{SWEAT} \ge 59.4$ |
| **<35%**| **Very Low** | $\text{PW} < 22.5\text{ mm}$, $\text{KI} < -10.6$, $\text{CAPE\_V} < 56\text{ J/kg}$ |

---

## 📂 Project Directory Structure

`
Real-Time-Thunderstorm-Prediction/
├── .streamlit/
│   └── config.toml                           # Streamlit visual theme configuration
├── final_outputs/                            # Production Model & Mapping Artifacts
│   ├── CatBoost_Final.pkl                    # Trained CatBoost model binary
│   ├── feature_order.pkl                     # Expected feature sequence
│   ├── feature_importance.pkl                # Feature importance dictionary
│   ├── CatBoost_Operational_Probability_Mapping.csv
│   └── CatBoost_Operational_Thunderstorm_Thresholds.csv
├── catboost/                                 # Primary CatBoost Pipeline & Research
│   ├── data/                                 # Chronological splits (train, validation, test)
│   ├── models/                               # CatBoost research model backups
│   ├── notebooks/                            # Sequential Jupyter notebooks (01 to 05)
│   └── results/                              # Evaluation metrics, SHAP ranking & distribution plots
├── other_models/                             # Benchmark Models & Experiments
│   ├── lightgbm/                             # LightGBM data, model & training notebook
│   ├── xgboost/                              # XGBoost data, model & training notebook
│   └── ensemble/                             # Stacking & Voting ensemble research
├── app.py                                    # Streamlit Real-Time Interactive Dashboard
├── predict.py                                # Inference pipeline module
├── fetch_latest.py                           # Wyoming Sounding fetcher via Siphon
├── metpy_indices.py                          # Sounding index calculation engine
├── requirements.txt                          # Python dependencies
└── Thunderstorm_Report_Final.pdf             # Project Technical Report
`

---

## 🛠️ Installation & Setup Guide

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Clone Repository
`ash
git clone https://github.com/YOUR_USERNAME/Real-Time-Thunderstorm-Prediction.git
cd Real-Time-Thunderstorm-Prediction
`

### 3. Create Virtual Environment
`ash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
`

### 4. Install Dependencies
`ash
pip install -r requirements.txt
`

### 5. Run Streamlit Dashboard
`ash
streamlit run app.py
`
Open [http://localhost:8501](http://localhost:8501) in your web browser.

---

## 👨‍💻 Author & Acknowledgements

* **Project Developer**: Internship Project at **India Meteorological Department (IMD)**.
* **Data Sources**: 
  * Atmospheric Radiosonde Soundings: **University of Wyoming Department of Atmospheric Science**.
  * Surface SYNOP Present/Past Weather Observations: **India Meteorological Department (IMD)**.
* **Libraries Used**: MetPy, Siphon, CatBoost, LightGBM, XGBoost, Optuna, Streamlit, Plotly, Pandas, Scikit-Learn.
