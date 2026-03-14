"""
src/evaluate.py
---------------
Evaluation metrics and visualisation for clinical risk prediction.

Metrics reported
~~~~~~~~~~~~~~~~
* **Accuracy**     – overall correct predictions.
* **ROC-AUC**      – discriminative ability across all classification
                     thresholds; the primary metric for clinical screening.
* **Precision**    – positive predictive value (PPV).
* **Recall**       – sensitivity (true positive rate); especially important
                     when missing a disease is costly.
* **F1 Score**     – harmonic mean of precision and recall.
* **Specificity**  – true negative rate.
* **Brier Score**  – mean squared error of probability estimates; lower
                     is better; measures calibration quality.

Clinical interpretation guidance is printed alongside each metric so that
a reader without an ML background can understand the results.

Plots produced
~~~~~~~~~~~~~~
* Confusion matrix heat-map
* ROC curve with AUC shading
* Precision–Recall curve
* Feature importance bar chart (tree-based and linear models)
"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Core metric computation                                                      #
# --------------------------------------------------------------------------- #

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Compute the full evaluation metric suite.

    Parameters
    ----------
    y_true : ground-truth binary labels.
    y_pred : hard binary predictions (threshold = 0.5).
    y_prob : predicted probability of the positive class.

    Returns
    -------
    dict[str, float]
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "roc_auc":     roc_auc_score(y_true, y_prob),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "brier_score": brier_score_loss(y_true, y_prob),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    return metrics


# --------------------------------------------------------------------------- #
#  Human-readable report                                                        #
# --------------------------------------------------------------------------- #

def print_metrics_report(metrics: dict, model_name: str) -> None:
    """
    Print a formatted metrics table with clinical interpretation notes.

    Parameters
    ----------
    metrics    : dict from :func:`compute_metrics`.
    model_name : display name for the header.
    """
    bar = "=" * 62
    logger.info(bar)
    logger.info("  Evaluation Report – %s", model_name)
    logger.info(bar)
    logger.info("  %-22s  %8s   Clinical note", "Metric", "Value")
    logger.info("  " + "-" * 58)

    rows = [
        ("ROC-AUC",      "roc_auc",     "Primary: ≥0.80 = good discriminator"),
        ("Accuracy",     "accuracy",    "Overall % correct predictions"),
        ("Recall (Sens)","recall",      "% of sick patients correctly flagged"),
        ("Specificity",  "specificity", "% of healthy patients correctly cleared"),
        ("Precision",    "precision",   "% of flagged patients truly sick"),
        ("F1 Score",     "f1",          "Balance between Recall and Precision"),
        ("Brier Score",  "brier_score", "Probability calibration quality (↓ better)"),
    ]
    for label, key, note in rows:
        logger.info("  %-22s  %8.4f   %s", label, metrics[key], note)

    logger.info("  " + "-" * 58)
    logger.info(
        "  Confusion matrix:  TP=%d  FP=%d  TN=%d  FN=%d",
        metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"],
    )
    logger.info(bar)


# --------------------------------------------------------------------------- #
#  Visualisation helpers                                                        #
# --------------------------------------------------------------------------- #

def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str,
) -> None:
    """Save a labelled confusion matrix heat-map."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    classes = ["No Disease\n(Negative)", "Disease\n(Positive)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix – {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", path)


def _plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: str,
) -> None:
    """Save a ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#4C72B0", label=f"AUC = {auc_score:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#4C72B0")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC = 0.50)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate (1 – Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title(f"ROC Curve – {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", path)


def _plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: str,
) -> None:
    """Save a Precision–Recall curve with average-precision annotation."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, color="#DD8452", label=f"AP = {ap:.3f}")
    ax.axhline(baseline, color="grey", lw=1.2, linestyle="--",
               label=f"Baseline (prevalence = {baseline:.2f})")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"Precision–Recall Curve – {model_name}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(output_dir, "pr_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", path)


def _plot_feature_importance(
    model: object,
    feature_names: list,
    model_name: str,
    output_dir: str,
    top_n: int = 15,
) -> None:
    """
    Save a feature-importance bar chart.

    Works with:
    * Tree-based models (feature_importances_ attribute)
    * Linear models (coef_ attribute)
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)

    if importances is None:
        logger.info("Feature importance not available for %s – skipping.", model_name)
        return

    # Align with feature names (lengths may differ after preprocessing)
    n = min(len(feature_names), len(importances))
    importances = importances[:n]
    names = feature_names[:n]

    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_imp   = importances[sorted_idx]
    top_names = [names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(range(len(top_imp)), top_imp[::-1], color="#55A868", alpha=0.85)
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances – {model_name}")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", path)


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #

def evaluate_model(
    model: object,
    model_name: str,
    data: dict,
    output_dir: str = "outputs",
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate the fitted model on the held-out test set and produce all plots.

    Parameters
    ----------
    model       : fitted sklearn estimator.
    model_name  : display name used in plot titles and logs.
    data        : dict from :func:`src.data_loader.load_and_preprocess`
                  (must contain X_test, y_test, feature_names).
    output_dir  : directory for saving plots.
    threshold   : classification threshold (default 0.5).

    Returns
    -------
    dict[str, float]  – computed metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_test = data["X_test"]
    y_test = data["y_test"]

    # Probability estimates
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback for models without predict_proba (e.g. SVM with no probability)
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.ptp() + 1e-9)  # min-max scale

    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Detailed sklearn classification report
    logger.info(
        "\nClassification Report:\n%s",
        classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]),
    )

    print_metrics_report(metrics, model_name)

    # Plots
    _plot_confusion_matrix(y_test, y_pred, model_name, output_dir)
    _plot_roc_curve(y_test, y_prob, model_name, output_dir)
    _plot_pr_curve(y_test, y_prob, model_name, output_dir)
    _plot_feature_importance(model, data["feature_names"], model_name, output_dir)

    return metrics
