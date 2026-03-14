"""
main.py
=======
Heart Disease Risk Prediction – End-to-End Pipeline
====================================================

This script orchestrates the complete clinical risk prediction workflow:

  1. **Data Loading & Preprocessing (EDA)**
     Load the Cleveland Heart Disease dataset, clean missing values,
     clip outliers, and produce exploratory visualisations.

  2. **Model Architecture**
     Four classifiers are evaluated:
       - Logistic Regression   (linear baseline)
       - Random Forest         (bagging ensemble)
       - Gradient Boosting     (boosting ensemble)
       - MLP Neural Network    (feed-forward deep learning)

  3. **Training Pipeline**
     Stratified 5-fold cross-validation ranks models by ROC-AUC.
     The winner is re-fitted on the full train+validation split.

  4. **Evaluation Metrics**
     The held-out test set is used *once* to report:
     Accuracy, ROC-AUC, Precision, Recall, F1, Specificity, Brier Score,
     and a confusion matrix.

Outputs (saved to ./outputs/)
------------------------------
  class_balance.png         – EDA: target class distribution
  correlation_heatmap.png   – EDA: feature correlation matrix
  feature_distributions.png – EDA: histograms by class
  cv_comparison.png         – Training: cross-validation AUC bar chart
  confusion_matrix.png      – Evaluation: confusion matrix heat-map
  roc_curve.png             – Evaluation: ROC curve with AUC
  pr_curve.png              – Evaluation: Precision–Recall curve
  feature_importance.png    – Evaluation: top feature importances

Usage
-----
  python main.py                      # full pipeline, default settings
  python main.py --no-plots           # skip EDA plot generation
  python main.py --output-dir results # save outputs to ./results/
  python main.py --threshold 0.40     # adjust classification threshold

Requirements
------------
  pip install -r requirements.txt
"""

import argparse
import logging
import sys
import os

# --------------------------------------------------------------------------- #
#  Logging configuration                                                        #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")

# --------------------------------------------------------------------------- #
#  Imports (project modules)                                                    #
# --------------------------------------------------------------------------- #
from src.data_loader import load_and_preprocess  # noqa: E402
from src.model import build_models, describe_model  # noqa: E402
from src.train import train_and_select  # noqa: E402
from src.evaluate import evaluate_model  # noqa: E402


# --------------------------------------------------------------------------- #
#  Command-line interface                                                       #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clinical Risk Prediction – Heart Disease Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        metavar="DIR",
        help="Directory for saving plots and artefacts (default: %(default)s)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip EDA plot generation (speeds up run in CI environments)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="T",
        help="Classification probability threshold (default: %(default)s)",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Main pipeline                                                                #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 62)
    logger.info("  Heart Disease Risk Prediction Pipeline")
    logger.info("=" * 62)

    # ----------------------------------------------------------------------- #
    # STEP 1 – Data Loading and Preprocessing (EDA)                            #
    # ----------------------------------------------------------------------- #
    logger.info("\n[STEP 1]  Data Loading and Preprocessing")
    data = load_and_preprocess(
        output_dir=args.output_dir,
        run_eda_plots=(not args.no_plots),
    )

    # ----------------------------------------------------------------------- #
    # STEP 2 – Model Architecture                                              #
    # ----------------------------------------------------------------------- #
    logger.info("\n[STEP 2]  Building Model Registry")
    models = build_models()
    for name, est in models.items():
        logger.info("  • %-28s  %s", name, describe_model(name, est))

    # ----------------------------------------------------------------------- #
    # STEP 3 – Training Pipeline                                               #
    # ----------------------------------------------------------------------- #
    logger.info("\n[STEP 3]  Training Pipeline (5-fold CV + Final Fit)")
    best_name, fitted_model = train_and_select(
        models,
        data,
        output_dir=args.output_dir,
    )

    # ----------------------------------------------------------------------- #
    # STEP 4 – Evaluation Metrics                                              #
    # ----------------------------------------------------------------------- #
    logger.info("\n[STEP 4]  Evaluation on Held-Out Test Set")
    metrics = evaluate_model(
        fitted_model,
        model_name=best_name,
        data=data,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )

    # ----------------------------------------------------------------------- #
    # Summary                                                                   #
    # ----------------------------------------------------------------------- #
    logger.info("\n" + "=" * 62)
    logger.info("  PIPELINE COMPLETE")
    logger.info("  Best model  : %s", best_name)
    logger.info("  Test AUC    : %.4f", metrics["roc_auc"])
    logger.info("  Test F1     : %.4f", metrics["f1"])
    logger.info("  Test Recall : %.4f", metrics["recall"])
    logger.info("  Outputs     : %s/", args.output_dir)
    logger.info("=" * 62)


if __name__ == "__main__":
    main()
