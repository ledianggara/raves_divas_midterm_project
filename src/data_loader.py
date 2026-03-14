"""
src/data_loader.py
------------------
Data loading, cleaning, and exploratory data analysis (EDA)
for the Heart Disease Risk Prediction pipeline.

Dataset
~~~~~~~
Cleveland Heart Disease dataset (UCI Machine Learning Repository), accessed
via sklearn's OpenML integration.  Feature description:

    age        – Age in years
    sex        – 1 = male, 0 = female
    cp         – Chest-pain type (0–3)
    trestbps   – Resting blood pressure (mm Hg)
    chol       – Serum cholesterol (mg/dl)
    fbs        – Fasting blood sugar > 120 mg/dl (1 = true)
    restecg    – Resting ECG results (0–2)
    thalach    – Maximum heart rate achieved (bpm)
    exang      – Exercise-induced angina (1 = yes)
    oldpeak    – ST depression induced by exercise vs. rest
    slope      – Slope of peak-exercise ST segment (0–2)
    ca         – Number of major vessels coloured by fluoroscopy (0–3)
    thal       – Thalassaemia type (1 = normal, 2 = fixed defect, 3 = reversible)
    target     – 0 = no disease, 1 = disease present  (binary label)
"""

import os
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Constants                                                                    #
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20        # 20 % held-out test set
VAL_SIZE = 0.15         # 15 % of *training* data used for validation

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COL = "target"


# --------------------------------------------------------------------------- #
#  Data loading                                                                 #
# --------------------------------------------------------------------------- #

def _load_heart_disease_openml() -> pd.DataFrame:
    """Fetch the Cleveland Heart Disease dataset from OpenML (requires network)."""
    logger.info("Fetching heart-disease dataset from OpenML …")
    dataset = fetch_openml(name="heart-disease", version=1, as_frame=True, parser="auto")
    df = dataset.frame.copy()
    # OpenML may name the target column differently
    if "class" in df.columns:
        df = df.rename(columns={"class": TARGET_COL})
    # Binarise: 0 → no disease, 1–4 → disease present
    df[TARGET_COL] = (df[TARGET_COL].astype(float) > 0).astype(int)
    return df


def _load_fallback_breast_cancer() -> pd.DataFrame:
    """
    Fallback: sklearn's Breast Cancer Wisconsin dataset reframed as
    malignancy risk prediction (0 = benign, 1 = malignant).
    """
    logger.warning(
        "OpenML unavailable – falling back to Breast Cancer Wisconsin dataset."
    )
    from sklearn.datasets import load_breast_cancer

    bc = load_breast_cancer(as_frame=True)
    df = bc.frame.copy()
    # sklearn labels: 0=malignant, 1=benign.  Flip so that 1=malignant
    # (high-risk = positive class), matching clinical risk-prediction convention.
    df[TARGET_COL] = (df["target"] == 0).astype(int)
    return df


def load_raw_data() -> pd.DataFrame:
    """
    Load the clinical dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with a binary ``target`` column (1 = disease / high risk).
    """
    try:
        df = _load_heart_disease_openml()
        logger.info("Heart Disease dataset loaded  shape=%s", df.shape)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenML fetch failed (%s).", exc)
        df = _load_fallback_breast_cancer()
        logger.info("Fallback dataset loaded  shape=%s", df.shape)
    return df


# --------------------------------------------------------------------------- #
#  Preprocessing                                                                #
# --------------------------------------------------------------------------- #

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      1. Convert object/categorical columns to numeric.
      2. Replace '?' placeholders (common in UCI exports) with NaN.
      3. Impute remaining NaN values with column medians.
      4. Clip extreme outliers beyond the 1st/99th percentile for continuous
         features (reduces influence of data-entry errors).

    Parameters
    ----------
    df : pd.DataFrame
        Raw clinical dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    df = df.copy()

    # Replace '?' with NaN (UCI CSV convention)
    df.replace("?", np.nan, inplace=True)

    # Cast all non-target columns to float
    for col in df.columns:
        if col != TARGET_COL:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_pct = df.isnull().mean() * 100
    if missing_pct.max() > 0:
        logger.info(
            "Missing values detected (median imputation applied):\n%s",
            missing_pct[missing_pct > 0].round(2),
        )

    # Median imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != TARGET_COL:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Clip outliers (continuous features only – skip binary/ordinal flags)
    skip_clip = {TARGET_COL, "sex", "fbs", "exang", "cp", "restecg", "slope", "ca", "thal"}
    for col in numeric_cols:
        if col not in skip_clip:
            p01 = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=p01, upper=p99)

    logger.info("Cleaning complete – shape=%s, missing=%d", df.shape, df.isnull().sum().sum())
    return df


def run_eda(df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """
    Exploratory Data Analysis: prints summary statistics and saves plots.

    Plots produced
    ~~~~~~~~~~~~~~
    * Class balance bar chart
    * Correlation heat-map
    * Distribution strip-plots for key continuous features

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (must contain ``TARGET_COL``).
    output_dir : str
        Directory where PNG figures are saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=== EDA Summary ===")
    logger.info("Shape          : %s", df.shape)
    logger.info("Class balance  :\n%s", df[TARGET_COL].value_counts(normalize=True).round(3))
    logger.info("Descriptive stats:\n%s", df.describe().round(2).to_string())

    # --- 1. Class balance --------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET_COL].value_counts()
    ax.bar(
        ["No Disease / Low Risk", "Disease / High Risk"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=["#4C72B0", "#DD8452"],
    )
    ax.set_title("Class Balance")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "class_balance.png"), dpi=120)
    plt.close(fig)
    logger.info("Saved: %s/class_balance.png", output_dir)

    # --- 2. Correlation heat-map -------------------------------------------- #
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=120)
    plt.close(fig)
    logger.info("Saved: %s/correlation_heatmap.png", output_dir)

    # --- 3. Key feature distributions --------------------------------------- #
    continuous = [c for c in df.columns if c not in {TARGET_COL, "sex", "fbs", "exang"}][:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flatten(), continuous):
        sns.histplot(
            data=df,
            x=col,
            hue=TARGET_COL,
            kde=True,
            ax=ax,
            palette=["#4C72B0", "#DD8452"],
        )
        ax.set_title(f"Distribution: {col}")
    plt.suptitle("Feature Distributions by Risk Class", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_distributions.png"), dpi=120)
    plt.close(fig)
    logger.info("Saved: %s/feature_distributions.png", output_dir)


# --------------------------------------------------------------------------- #
#  Feature engineering & splitting                                              #
# --------------------------------------------------------------------------- #

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) from labels (y).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with ``TARGET_COL`` present.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target vector.
    """
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].astype(float)
    y = df[TARGET_COL].astype(int)
    logger.info("Features=%d  Samples=%d  Positive-rate=%.1f%%", X.shape[1], len(y), y.mean() * 100)
    return X, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Stratified train / validation / test split with standard-score normalisation.

    The scaler is fit *only* on the training set to prevent data leakage.

    Parameters
    ----------
    X, y : feature matrix and label vector.
    test_size : fraction reserved for the held-out test set.
    val_size  : fraction of the remaining training data used for validation.
    random_state : reproducibility seed.

    Returns
    -------
    dict with keys:
        X_train, X_val, X_test : scaled feature arrays (np.ndarray)
        y_train, y_val, y_test : label arrays (np.ndarray)
        scaler                 : fitted StandardScaler
        feature_names          : list of feature column names
    """
    X_arr = X.values
    y_arr = y.values

    # --- Hold-out test set ---
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_arr, y_arr,
        test_size=test_size,
        stratify=y_arr,
        random_state=random_state,
    )

    # --- Validation set carved from training data ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        stratify=y_trainval,
        random_state=random_state,
    )

    # --- Normalise (fit on train only) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    logger.info(
        "Split sizes – train=%d  val=%d  test=%d",
        len(y_train), len(y_val), len(y_test),
    )

    return {
        "X_train": X_train,
        "X_val":   X_val,
        "X_test":  X_test,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "scaler":  scaler,
        "feature_names": list(X.columns),
    }


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #

def load_and_preprocess(output_dir: str = "outputs", run_eda_plots: bool = True) -> dict:
    """
    End-to-end data pipeline: load → clean → EDA → split → scale.

    Parameters
    ----------
    output_dir    : directory for saving EDA plots.
    run_eda_plots : set to False to skip plot generation (e.g. in unit tests).

    Returns
    -------
    dict  (same structure as :func:`split_and_scale`)
    """
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    if run_eda_plots:
        run_eda(clean_df, output_dir=output_dir)
    X, y = prepare_features(clean_df)
    data = split_and_scale(X, y)
    return data
