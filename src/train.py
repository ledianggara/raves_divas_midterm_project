"""
src/train.py
------------
Training pipeline: cross-validation, fitting, logging, and model selection.

Design decisions
~~~~~~~~~~~~~~~~
* **Stratified 5-fold cross-validation** is used to reliably estimate
  out-of-sample performance before touching the held-out test set.  This
  avoids the optimistic bias that comes from tuning on test data.
* **ROC-AUC** is the primary model-selection metric because the clinical
  cost of false negatives (missed disease) differs from false positives,
  and AUC captures performance across all decision thresholds.
* **Training logs** record fold-level and mean ± std metrics so the
  practitioner can assess variance, not just point estimates.
* The best model is re-fitted on the full train+validation split before
  final evaluation on the held-out test set.
"""

import logging
import time
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import clone

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Constants                                                                    #
# --------------------------------------------------------------------------- #
CV_FOLDS = 5
SCORING = {
    "roc_auc":   "roc_auc",
    "accuracy":  "accuracy",
    "f1":        "f1",
    "recall":    "recall",
    "precision": "precision",
}
PRIMARY_METRIC = "test_roc_auc"   # used to rank models


# --------------------------------------------------------------------------- #
#  Cross-validation                                                             #
# --------------------------------------------------------------------------- #

def cross_validate_models(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = CV_FOLDS,
    random_state: int = 42,
) -> dict:
    """
    Run stratified k-fold cross-validation for every model in the registry.

    Parameters
    ----------
    models       : dict of {name: estimator} from :func:`src.model.build_models`.
    X_train      : scaled training feature matrix.
    y_train      : training labels.
    cv_folds     : number of CV folds.
    random_state : seed for fold splitting.

    Returns
    -------
    cv_results : dict of {model_name: {metric: array_of_fold_scores}}
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results: dict = {}

    logger.info("=== Cross-Validation (%d-fold) ===", cv_folds)
    logger.info("%-26s  %6s ± %-6s  %6s ± %-6s  %6s ± %-6s",
                "Model", "AUC", "std", "Acc", "std", "F1", "std")
    logger.info("-" * 72)

    for name, estimator in models.items():
        t0 = time.perf_counter()
        scores = cross_validate(
            clone(estimator),
            X_train, y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            return_train_score=False,
        )
        elapsed = time.perf_counter() - t0
        cv_results[name] = scores

        auc_m  = scores["test_roc_auc"].mean()
        auc_s  = scores["test_roc_auc"].std()
        acc_m  = scores["test_accuracy"].mean()
        acc_s  = scores["test_accuracy"].std()
        f1_m   = scores["test_f1"].mean()
        f1_s   = scores["test_f1"].std()

        logger.info(
            "%-26s  %.4f ± %.4f  %.4f ± %.4f  %.4f ± %.4f  [%.1fs]",
            name, auc_m, auc_s, acc_m, acc_s, f1_m, f1_s, elapsed,
        )

    return cv_results


def select_best_model(cv_results: dict, models: dict) -> tuple[str, object]:
    """
    Choose the model with the highest mean cross-validation AUC.

    Parameters
    ----------
    cv_results : output of :func:`cross_validate_models`.
    models     : original model registry (used to retrieve the winner).

    Returns
    -------
    best_name      : str
    best_estimator : unfitted sklearn estimator (fresh clone)
    """
    best_name = max(
        cv_results,
        key=lambda n: cv_results[n][PRIMARY_METRIC].mean(),
    )
    best_estimator = clone(models[best_name])
    best_auc = cv_results[best_name][PRIMARY_METRIC].mean()
    logger.info(
        "Best model: '%s'  (CV AUC = %.4f)", best_name, best_auc
    )
    return best_name, best_estimator


# --------------------------------------------------------------------------- #
#  Final fit                                                                    #
# --------------------------------------------------------------------------- #

def fit_final_model(
    estimator: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> object:
    """
    Re-fit the chosen estimator on the combined train + validation data.

    Using all available non-test data for the final fit maximises the
    information available to the model while keeping the test set pristine.

    Parameters
    ----------
    estimator          : unfitted estimator (e.g. returned by clone()).
    X_train, y_train   : training split.
    X_val, y_val       : validation split (merged with train for final fit).

    Returns
    -------
    fitted estimator
    """
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    logger.info("Fitting final model on train+val  (n=%d) …", len(y_full))
    t0 = time.perf_counter()
    estimator.fit(X_full, y_full)
    elapsed = time.perf_counter() - t0
    logger.info("Fit complete  (%.2f s)", elapsed)
    return estimator


# --------------------------------------------------------------------------- #
#  Learning curve                                                               #
# --------------------------------------------------------------------------- #

def plot_cv_comparison(
    cv_results: dict,
    output_dir: str = "outputs",
) -> None:
    """
    Bar chart comparing mean CV AUC across all models with ± 1 std error bars.

    Parameters
    ----------
    cv_results : output of :func:`cross_validate_models`.
    output_dir : directory where the PNG is saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    names = list(cv_results.keys())
    means = [cv_results[n][PRIMARY_METRIC].mean() for n in names]
    stds  = [cv_results[n][PRIMARY_METRIC].std()  for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(names, means, yerr=stds, capsize=6, color=colours, alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"{CV_FOLDS}-Fold Cross-Validation AUC (mean ± std)")
    ax.set_xticklabels(names, rotation=15, ha="right")

    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean_val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "cv_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", path)


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #

def train_and_select(
    models: dict,
    data: dict,
    output_dir: str = "outputs",
) -> tuple[str, object]:
    """
    Full training pipeline:
      1. Cross-validate all models on the training set.
      2. Select the best model by CV AUC.
      3. Re-fit on train + validation.
      4. Save a comparison chart.

    Parameters
    ----------
    models     : dict from :func:`src.model.build_models`.
    data       : dict from :func:`src.data_loader.load_and_preprocess`.
    output_dir : where to save training plots.

    Returns
    -------
    best_name      : str
    fitted_model   : fitted sklearn estimator
    """
    cv_results = cross_validate_models(
        models,
        data["X_train"],
        data["y_train"],
    )
    plot_cv_comparison(cv_results, output_dir=output_dir)

    best_name, best_estimator = select_best_model(cv_results, models)
    fitted_model = fit_final_model(
        best_estimator,
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
    )
    return best_name, fitted_model
