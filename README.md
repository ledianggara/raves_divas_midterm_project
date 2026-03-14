# Clinical Risk Prediction – Heart Disease

A machine-learning pipeline for predicting heart disease risk from structured
tabular clinical data.  Accompanies the **RAVES DIVAS Midterm Project**.

---

## Project Structure

```
raves_divas_midterm_project/
├── main.py                  # End-to-end pipeline entry point
├── requirements.txt         # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Data loading, cleaning, EDA (Step 1)
│   ├── model.py             # Model architectures (Step 2)
│   ├── train.py             # Training pipeline & cross-validation (Step 3)
│   └── evaluate.py          # Evaluation metrics & plots (Step 4)
├── docs/
│   └── field_guide.md       # Clinical mini field guide (memo)
└── outputs/                 # Auto-created; plots saved here
```

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/ledianggara/raves_divas_midterm_project.git
cd raves_divas_midterm_project
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Full Pipeline

```bash
python main.py
```

This will:
1. Download the **Heart Disease dataset** from OpenML (or fall back to the
   Breast Cancer Wisconsin dataset when offline).
2. Clean the data, run EDA, and save exploratory plots to `outputs/`.
3. Cross-validate four models (Logistic Regression, Random Forest,
   Gradient Boosting, MLP Neural Network) and select the best by ROC-AUC.
4. Re-fit the winning model on train + validation data.
5. Evaluate on the held-out test set and print a full metrics report.
6. Save all plots to the `outputs/` directory.

### 5. Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir DIR` | `outputs` | Directory for saved plots |
| `--no-plots` | off | Skip EDA plot generation (faster in CI) |
| `--threshold T` | `0.5` | Classification probability threshold |

**Examples:**

```bash
# Save plots to a custom directory
python main.py --output-dir results

# Skip EDA plots for a faster run
python main.py --no-plots

# Use a more sensitive threshold (flag more potential cases)
python main.py --threshold 0.35
```

---

## Output Files

After running, `outputs/` will contain:

| File | Description |
|------|-------------|
| `class_balance.png` | Bar chart of positive vs. negative class counts |
| `correlation_heatmap.png` | Pearson correlation matrix of all features |
| `feature_distributions.png` | Histograms of key features by class |
| `cv_comparison.png` | Cross-validation AUC comparison across models |
| `confusion_matrix.png` | Confusion matrix on the held-out test set |
| `roc_curve.png` | ROC curve with AUC annotation |
| `pr_curve.png` | Precision–Recall curve |
| `feature_importance.png` | Top feature importances (tree / linear models) |

---

## Clinical Mini Field Guide

See [`docs/field_guide.md`](docs/field_guide.md) for the 2–3 page clinical
memo covering:

- **Problem Statement** – the clinical question addressed
- **Data Description** – dataset overview and limitations
- **Methods** – modelling approach and design decisions
- **Results** – key performance metrics with interpretation
- **Limitations & Next Steps** – what would be done differently

---

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for package versions

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
```

---

## License

This project is for academic purposes only. The models have **not** been
clinically validated and must not be used for patient care decisions.
