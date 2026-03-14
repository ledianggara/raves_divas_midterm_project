# Clinical Risk Prediction – Mini Field Guide

**Project:** Heart Disease / Malignancy Risk Prediction from Structured Clinical Data  
**Authors:** RAVES DIVAS  
**Date:** March 2026  
**Course:** Midterm Project

---

## 1. Problem Statement

Cardiovascular disease is the leading cause of death worldwide, responsible for an estimated 17.9 million deaths annually (WHO, 2023). Early identification of high-risk patients enables timely interventions—lifestyle counselling, pharmacological management, or urgent referral—that measurably reduce morbidity and mortality.

This project addresses the following clinical question:

> **Given a patient's demographic characteristics and routinely collected clinical measurements, can we predict whether they carry a significant risk of heart disease (or malignancy) with enough sensitivity and specificity to be useful at the point of care?**

A machine-learning risk score could supplement—never replace—a clinician's judgement by flagging high-risk cases for prioritised review, reducing the cognitive burden of manual chart triage in busy outpatient or emergency settings.

---

## 2. Data Description

### Source

The primary target is the **Cleveland Heart Disease dataset** from the UCI Machine Learning Repository (Detrano et al., 1989), accessed via OpenML. When network access is unavailable the pipeline automatically falls back to the **Breast Cancer Wisconsin (Diagnostic) dataset** (Wolberg et al., 1992), which similarly frames a binary clinical risk classification task.

### Features (Heart Disease variant, 13 predictors)

| Feature     | Type        | Description                                    |
|-------------|-------------|------------------------------------------------|
| `age`       | Continuous  | Age in years                                   |
| `sex`       | Binary      | 1 = male, 0 = female                           |
| `cp`        | Ordinal     | Chest-pain type (0 = typical angina → 3)       |
| `trestbps`  | Continuous  | Resting blood pressure (mm Hg)                 |
| `chol`      | Continuous  | Serum cholesterol (mg/dl)                      |
| `fbs`       | Binary      | Fasting blood sugar > 120 mg/dl                |
| `restecg`   | Ordinal     | Resting ECG results (0–2)                      |
| `thalach`   | Continuous  | Maximum heart rate achieved (bpm)              |
| `exang`     | Binary      | Exercise-induced angina                        |
| `oldpeak`   | Continuous  | ST depression induced by exercise              |
| `slope`     | Ordinal     | Slope of peak-exercise ST segment              |
| `ca`        | Ordinal     | Number of major vessels coloured (0–3)         |
| `thal`      | Ordinal     | Thalassaemia type                              |
| **target**  | Binary      | **0 = no disease, 1 = disease present**        |

### Size and Balance

The Cleveland dataset contains **303 patients** (165 without disease, 138 with); a modest but well-characterised cohort that has been the benchmark for clinical ML for three decades. The Breast Cancer fallback contains **569 samples** (357 benign, 212 malignant), with similar class imbalance (~37 % positive rate).

### Limitations of the Data

| Limitation | Impact |
|---|---|
| **Small sample size** (n ≈ 300) | High variance in model estimates; large confidence intervals on metrics. |
| **Single-centre data** (Cleveland Clinic, 1980s) | Likely poor generalisability across populations, ethnicities, and modern care pathways. |
| **Ordinal proxies** (e.g. `thal`, `cp`) | Encoded categorical labels may not reflect the full clinical spectrum of findings. |
| **Missing values** (`ca`, `thal`) | Up to 6 % of rows contain `?` placeholders; imputed with column medians, introducing slight bias. |
| **No temporal dimension** | All features are snapshots; the model cannot capture disease trajectory or medication history. |
| **Target binarisation** | The original labels encode severity 0–4; collapsing to 0 vs. ≥1 discards prognostic nuance. |

---

## 3. Methods

### Approach and Rationale

A **supervised binary classification** framework was chosen because the task is well-defined (predict presence/absence) and the ground truth labels are available. Four algorithm families were evaluated to capture a progression from simple-and-interpretable to complex-and-accurate:

| Model | Rationale |
|---|---|
| **Logistic Regression** | Widely used in clinical risk scoring (e.g. Framingham, CHADS₂); coefficients are directly interpretable as log-odds. Serves as the linear baseline. |
| **Random Forest** | Handles non-linear feature interactions and collinearity naturally; resistant to outliers; provides Gini-based feature importances. |
| **Gradient Boosting** | Typically achieves the highest accuracy on tabular data; stochastic subsampling (`subsample=0.8`) reduces overfitting; shallower trees (`max_depth=4`) improve generalisation. |
| **MLP Neural Network** | Demonstrates a deep-learning approach on structured data; two hidden layers (128 → 64 units, ReLU activation); early stopping prevents overfitting. |

### Preprocessing

1. **Missing-value imputation** – column medians (robust to skew; avoids mean shift by outliers).
2. **Outlier clipping** – continuous features clipped to the 1st–99th percentile to dampen data-entry errors.
3. **Standard-score normalisation** (`StandardScaler`, fit on training data only) – zero mean, unit variance; required for Logistic Regression and MLP; harmless for tree methods.
4. **Stratified split** – 65 % train / 15 % validation / 20 % test, maintaining class proportions in each partition.

### Model Selection

**Stratified 5-fold cross-validation** on the training partition ranks models by mean ROC-AUC. The winning model is re-fitted on the combined train+validation set, and the held-out test set is evaluated exactly once to report final performance. This protocol prevents optimistic bias from data leakage.

**Why ROC-AUC?**  
In screening contexts the optimal decision threshold depends on the relative costs of missed diagnoses (false negatives) versus unnecessary investigations (false positives), which vary by setting. ROC-AUC summarises discrimination across all thresholds without committing to one—making it the standard metric for clinical risk score development.

---

## 4. Results

### Cross-Validation Summary (Breast Cancer Wisconsin Fallback, n = 569)

| Model               | CV AUC (mean ± std) | CV Accuracy | CV F1  |
|---------------------|---------------------|-------------|--------|
| Logistic Regression | 0.991 ± 0.007       | 0.956       | 0.940  |
| Random Forest       | 0.988 ± 0.013       | 0.956       | 0.940  |
| Gradient Boosting   | 0.991 ± 0.008       | 0.966       | 0.954  |
| **MLP Neural Network** | **0.992 ± 0.006** | **0.974** | **0.965** |

*Selected model: MLP Neural Network (highest mean CV AUC)*

### Test-Set Performance (20 % held-out, n = 114)

| Metric      | Value  | Clinical Interpretation |
|-------------|--------|-------------------------|
| **ROC-AUC** | **0.997** | Near-perfect discrimination; the model consistently ranks high-risk patients above low-risk ones. |
| Accuracy    | 0.982  | 98.2 % of patients correctly classified. |
| Recall (Sensitivity) | 0.976 | 97.6 % of disease-positive patients correctly flagged—very low missed-diagnosis rate. |
| Specificity | 0.986  | 98.6 % of disease-negative patients correctly cleared—very low over-referral rate. |
| Precision (PPV) | 0.976 | When the model raises an alert, it is correct 97.6 % of the time. |
| F1 Score    | 0.976  | Excellent balance between sensitivity and precision. |
| Brier Score | 0.018  | Excellent probability calibration (0 = perfect, 0.25 = uninformative). |

**Confusion matrix summary**: TP = 41, FP = 1, TN = 71, FN = 1 (out of 114 test patients).

### Interpretation

The MLP achieves near-perfect discrimination on this dataset, consistent with published benchmarks on the Breast Cancer Wisconsin data. The single false negative and single false positive suggest the model has learned robust decision boundaries. However, these headline figures must be interpreted cautiously given the small test set (n = 114); a single misclassification shifts recall by ≈ 2.4 percentage points.

---

## 5. Limitations and Next Steps

### Current Limitations

| Area | Limitation | Consequence |
|---|---|---|
| **Generalisability** | Single-centre, historical cohort | Cannot be deployed in a new clinical setting without prospective validation. |
| **Interpretability** | MLP is a black-box | Clinicians cannot audit individual predictions; may reduce trust and adoption. |
| **Calibration** | Brier score is good but not verified with Platt scaling or isotonic regression | Predicted probabilities may not be reliable enough for shared decision-making. |
| **Threshold** | Fixed at 0.50 | The optimal threshold should be derived from a cost-benefit analysis (e.g. via a decision-curve analysis). |
| **Class imbalance** | Mild imbalance (~37 % positive) handled by `class_weight` only | More severe imbalance would require SMOTE or other re-sampling. |
| **External validation** | Not performed | AUC on an unseen hospital's data could be substantially lower (optimism bias). |

### Recommended Next Steps

1. **Prospective validation** on a held-out institution's data using the same feature set.
2. **Calibration curve** (reliability diagram) and Platt scaling to ensure predicted probabilities are trustworthy for risk communication to patients.
3. **Explainability layer** – SHAP (SHapley Additive exPlanations) values to provide per-patient feature contributions, enabling clinicians to understand *why* a patient was flagged.
4. **Decision-curve analysis** to choose the operating threshold that maximises net clinical benefit relative to treat-all and treat-none strategies.
5. **Fairness audit** – test for performance disparities across age, sex, and ethnicity subgroups before any clinical deployment.
6. **Integration with richer data** – incorporate medication history, longitudinal lab trends, and imaging findings to improve predictive power and clinical utility.
7. **Regulatory and ethics review** – any clinical deployment in the EU or US requires conformity with the EU AI Act / FDA SaMD guidelines and IRB/ethics approval.

---

*This field guide is intended as an academic deliverable. The models described herein have not been clinically validated and must not be used for patient care decisions.*
