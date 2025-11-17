import json
import os

# Notebook 02 - enhance with more details
nb02_cells = [
    {"cell_type": "markdown", "source": ["# Notebook 02: Regularization (Ridge & Lasso)\n\n## The Art of Restraint\n\nOverfitting is the enemy of generalization. Regularization tames it by penalizing complexity. Ridge shrinks coefficients smoothly. Lasso can zero them out entirely, performing automatic feature selection.\n\n### Why Regularize?\n\n- **Bias-variance tradeoff**: Reduce variance (overfitting) at the cost of slight bias\n- **Multicollinearity**: When features are correlated, coefficients become unstable\n- **Feature selection**: Lasso can automatically select important features\n\n### Ridge vs Lasso\n\n- **Ridge (L2)**: Penalizes sum of squared coefficients. Shrinks all coefficients toward zero, but rarely sets them to exactly zero.\n- **Lasso (L1)**: Penalizes sum of absolute coefficients. Can set coefficients to exactly zero, performing feature selection.\n\nWe use cross-validation to pick the regularization strength (alpha)."]},
    {"cell_type": "code", "source": ["import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import load_diabetes\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import RidgeCV, LassoCV\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\nfrom src.utils import set_seed\nset_seed(42)"]},
    {"cell_type": "markdown", "source": ["## TODO: Pipeline with RidgeCV"]},
    {"cell_type": "code", "source": ["# === TODO (you code this) ===\n# Pipeline: StandardScaler + RidgeCV over alphas logspace(1e-3..1e3)\n# Acceptance: Print best alpha, test RMSE, R2"]},
    {"cell_type": "markdown", "source": ["## TODO: Pipeline with LassoCV"]},
    {"cell_type": "code", "source": ["# === TODO (you code this) ===\n# Pipeline: StandardScaler + LassoCV. Compare metrics with Ridge.\n# Acceptance: Table with RMSE, MAE, R2 for both; 2-sentence comparison"]},
    {"cell_type": "markdown", "source": ["## TODO: Plot coefficient magnitudes"]},
    {"cell_type": "code", "source": ["# === TODO (you code this) ===\n# Plot coefficient magnitudes side by side for Ridge vs Lasso.\n# Acceptance: Figure with clear legend; note which coefficients go to zero with Lasso"]}
]

# Create minimal versions of remaining notebooks for now
notebooks = {
    "03_multicollinearity_vif_pca.ipynb": [
        {"cell_type": "markdown", "source": ["# Notebook 03: Multicollinearity & PCA\n\n## Detecting Redundancy\n\nWhen features are highly correlated, coefficients become unstable. VIF diagnoses the problem. PCA reveals underlying structure."]},
        {"cell_type": "code", "source": ["import numpy as np\nimport pandas as pd\nfrom sklearn.datasets import load_diabetes\nfrom sklearn.decomposition import PCA\nfrom statsmodels.stats.outliers_influence import variance_inflation_factor\nfrom src.utils import set_seed\nset_seed(42)"]},
        {"cell_type": "code", "source": ["# === TODO: Compute VIF for numeric features after standardization\n# Acceptance: VIF table printed; flag VIF > 10"]},
        {"cell_type": "code", "source": ["# === TODO: PCA on standardized X; plot cumulative explained variance\n# Acceptance: Scree plot and number of components for 90% variance"]},
        {"cell_type": "code", "source": ["# === TODO: Optional: refit Ridge on top k principal components\n# Acceptance: Report RMSE on PCs vs original features"]}
    ],
    "04_shap_linear_and_trees.ipynb": [
        {"cell_type": "markdown", "source": ["# Notebook 04: SHAP Values\n\n## Unpacking Predictions\n\nSHAP answers: How much did each feature contribute to this prediction? Based on game theory, ensuring local attributions sum to prediction minus baseline."]},
        {"cell_type": "code", "source": ["import shap\nimport xgboost as xgb\nfrom sklearn.linear_model import Ridge\nfrom src.utils import set_seed\nset_seed(42)"]},
        {"cell_type": "code", "source": ["# === TODO: Fit Ridge, compute SHAP with LinearExplainer on 500-row sample\n# Acceptance: SHAP summary plot; 2-sentence interpretation"]},
        {"cell_type": "code", "source": ["# === TODO: Fit XGBoost, compute SHAP with TreeExplainer\n# Acceptance: SHAP beeswarm plot; note top 3 features"]},
        {"cell_type": "code", "source": ["# === TODO: Compare SHAP ranking vs permutation importance\n# Acceptance: Table with top 5 features by each method"]}
    ],
    "05_cv_schemes_and_leakage.ipynb": [
        {"cell_type": "markdown", "source": ["# Notebook 05: Cross-Validation & Leakage\n\n## The Validation Trap\n\nData leakage is the silent killer. GroupKFold and TimeSeriesSplit prevent it."]},
        {"cell_type": "code", "source": ["from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit\nfrom src.utils import set_seed\nset_seed(42)"]},
        {"cell_type": "code", "source": ["# === TODO: Demonstrate KFold vs GroupKFold\n# Acceptance: Show GroupKFold gives more conservative estimate"]},
        {"cell_type": "code", "source": ["# === TODO: TimeSeriesSplit demo\n# Acceptance: Plot fold boundaries and compute fold metrics"]}
    ],
    "06_summary_quiz.ipynb": [
        {"cell_type": "markdown", "source": ["# Notebook 06: Summary & Reflection\n\n## Synthesizing Knowledge\n\nAnswer these questions in your own words to reinforce learning."]},
        {"cell_type": "markdown", "source": ["## Question 1: Permutation Importance\n\nIn your own words, define permutation importance and one case it can mislead.\n\n*(Write 3-6 lines here)*"]},
        {"cell_type": "markdown", "source": ["## Question 2: Ridge vs Lasso\n\nWhen do you prefer Ridge vs Lasso and why?\n\n*(Write 3-6 lines here)*"]},
        {"cell_type": "markdown", "source": ["## Question 3: PCA\n\nWhat PCA told you about redundancy in your features.\n\n*(Write 3-6 lines here)*"]},
        {"cell_type": "markdown", "source": ["## Question 4: SHAP Pitfalls\n\nOne pitfall of SHAP and how you would mitigate it.\n\n*(Write 3-6 lines here)*"]}
    ]
}

for filename, cells in notebooks.items():
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(f"notebooks/{filename}", "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Created {filename}")

print("All remaining notebooks created!")
