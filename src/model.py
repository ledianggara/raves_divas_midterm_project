"""
src/model.py
------------
Model architectures for clinical risk prediction.

Four classifiers are provided so results can be compared side-by-side:

  1. LogisticRegression – transparent linear baseline; widely used in
     clinical decision-support systems.
  2. RandomForestClassifier – robust ensemble that handles non-linear
     interactions and provides feature importances.
  3. GradientBoostingClassifier – generally highest tabular accuracy via
     sequential boosting; interpretable via feature-importance plots.
  4. MLPClassifier (Multi-Layer Perceptron) – feed-forward neural network
     with two hidden layers; demonstrates a deep-learning approach on
     structured data.

All models share a consistent sklearn estimator interface so they can be
plugged into the same training and evaluation pipeline.
"""

import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Architecture definitions                                                     #
# --------------------------------------------------------------------------- #

# Logistic Regression
# -------------------
# l2-regularised logistic regression with liblinear solver.
# C=1.0 (inverse regularisation strength) is a standard starting point.
LOGISTIC_REGRESSION = LogisticRegression(
    C=1.0,
    solver="liblinear",
    max_iter=1000,
    class_weight="balanced",   # handles mild class imbalance
    random_state=42,
)

# Random Forest
# -------------
# 300 trees with max-depth capped at 10 to reduce overfitting.
# class_weight="balanced_subsample" re-weights each bootstrap sample.
RANDOM_FOREST = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
)

# Gradient Boosting
# -----------------
# Shrinkage rate (learning_rate) of 0.05 combined with 200 estimators and
# shallow trees (max_depth=4) is a common well-generalising configuration.
GRADIENT_BOOSTING = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=10,
    subsample=0.8,           # stochastic GB – improves generalisation
    max_features="sqrt",
    random_state=42,
)

# Multi-Layer Perceptron (Neural Network)
# ----------------------------------------
# Architecture: input → 128 → 64 → output
# relu activations, adam optimiser, l2 weight-decay via alpha parameter.
# early_stopping=True monitors validation loss to halt training when it
# stops improving, which acts as implicit regularisation.
MLP = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-3,              # l2 regularisation (weight decay)
    batch_size=32,
    learning_rate="adaptive",
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=False,
)


# --------------------------------------------------------------------------- #
#  Build model registry                                                         #
# --------------------------------------------------------------------------- #

def build_models() -> dict[str, object]:
    """
    Return an ordered dict mapping model names to fresh (unfitted) estimators.

    Using fresh instances ensures there is no state leakage between runs when
    this function is called multiple times (e.g. in cross-validation loops).

    Returns
    -------
    dict[str, sklearn-estimator]
        Keys:  "Logistic Regression", "Random Forest",
               "Gradient Boosting", "MLP (Neural Network)"
    """
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=10,
            subsample=0.8,
            max_features="sqrt",
            random_state=42,
        ),
        "MLP (Neural Network)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=32,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        ),
    }

    logger.info("Model registry built – %d architectures registered.", len(models))
    for name in models:
        logger.debug("  • %s", name)

    return models


def describe_model(name: str, estimator: object) -> str:
    """
    Return a human-readable one-line description of a model.

    Parameters
    ----------
    name      : model name (key in the registry).
    estimator : sklearn estimator.

    Returns
    -------
    str
    """
    descriptions = {
        "Logistic Regression": (
            "L2-regularised logistic regression – interpretable linear baseline."
        ),
        "Random Forest": (
            "300-tree bagging ensemble with balanced class weights."
        ),
        "Gradient Boosting": (
            "Sequential boosting with shrinkage lr=0.05, stochastic subsampling."
        ),
        "MLP (Neural Network)": (
            "Feed-forward network: input→128(ReLU)→64(ReLU)→sigmoid output."
        ),
    }
    return descriptions.get(name, repr(estimator))
