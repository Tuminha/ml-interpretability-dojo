# Smoke Test Plan - ML Interpretability Dojo

This document outlines acceptance criteria for each notebook. Use this as a checklist to verify that your implementations are correct and complete.

---

## Notebook 00: Invariance and Baselines

### Acceptance Criteria

- [ ] **A1.1**: Dataset loaded successfully
  - X and y shapes printed
  - Feature names displayed
  - Data types are correct (DataFrame/Series)

- [ ] **A1.2**: Baseline model fitted
  - LinearRegression trained on train set
  - Test RMSE computed and printed
  - R² score computed and printed

- [ ] **A1.3**: Invariance demonstrated
  - Test rows reordered randomly
  - Predictions compared (should be identical up to permutation)
  - Markdown cell explains: "Row order does not change predictions"

---

## Notebook 01: Permutation Importance

### Acceptance Criteria

- [ ] **A2.1**: Model trained with pipeline
  - Pipeline includes StandardScaler + Ridge/LinearRegression
  - Test RMSE and R² printed
  - Model performance is reasonable

- [ ] **A2.2**: Permutation importance computed
  - `sklearn.inspection.permutation_importance` used
  - n_repeats=10 (or similar)
  - Importances computed on test set
  - Bar plot created showing feature importances
  - Plot saved to `images/` folder

- [ ] **A2.3**: Manual permutation check
  - One strong feature manually permuted
  - RMSE recomputed after permutation
  - Metric delta shown (should increase if feature is important)
  - Short interpretation paragraph written

---

## Notebook 02: Regularization (Ridge & Lasso)

### Acceptance Criteria

- [ ] **A3.1**: RidgeCV implemented
  - Pipeline: StandardScaler + RidgeCV
  - Alpha range: logspace(1e-3, 1e3) or similar
  - Best alpha printed
  - Test RMSE and R² printed

- [ ] **A3.2**: LassoCV implemented
  - Pipeline: StandardScaler + LassoCV
  - Alpha range similar to Ridge
  - Best alpha printed
  - Test RMSE, MAE, R² printed

- [ ] **A3.3**: Comparison table created
  - Table with RMSE, MAE, R² for both models
  - 2-sentence comparison written
  - Explains when to use Ridge vs Lasso

- [ ] **A3.4**: Coefficient comparison plot
  - Side-by-side bar plot of Ridge vs Lasso coefficients
  - Clear legend
  - Note which coefficients go to zero with Lasso
  - Plot saved to `images/` folder

---

## Notebook 03: Multicollinearity & PCA

### Acceptance Criteria

- [ ] **A4.1**: VIF computed
  - VIF computed for all numeric features
  - Features standardized before VIF
  - VIF table printed
  - Features with VIF > 10 flagged
  - Interpretation written

- [ ] **A4.2**: PCA analysis
  - PCA fitted on standardized X
  - Cumulative explained variance plotted (scree plot)
  - Number of components for 90% variance reported
  - Plot saved to `images/` folder

- [ ] **A4.3**: Optional PCA refitting
  - Ridge refitted on top k principal components
  - Test RMSE compared to original features
  - Sentence on tradeoffs written

---

## Notebook 04: SHAP Values

### Acceptance Criteria

- [ ] **A5.1**: SHAP for linear model
  - Ridge model fitted on standardized data
  - Background sample of 500 rows created
  - LinearExplainer used
  - SHAP values computed on test sample
  - SHAP summary plot displayed
  - 2-sentence interpretation written
  - Plot saved to `images/` folder

- [ ] **A5.2**: SHAP for tree model
  - XGBoost regressor fitted (n_estimators=200, max_depth=3)
  - TreeExplainer used
  - SHAP values computed on small test sample
  - SHAP beeswarm plot displayed
  - Top 3 features identified with directionality
  - Plot saved to `images/` folder

- [ ] **A5.3**: SHAP vs Permutation Importance comparison
  - Top 5 features by SHAP ranked
  - Top 5 features by permutation importance ranked
  - Comparison table created
  - Short comment on agreement/disagreement written

---

## Notebook 05: Cross-Validation & Leakage

### Acceptance Criteria

- [ ] **A6.1**: KFold vs GroupKFold demonstration
  - Toy dataset with repeated "player_id" or group column
  - Groups synthesized by repeating indices
  - KFold cross-validation scores computed
  - GroupKFold cross-validation scores computed
  - Comparison shows GroupKFold gives more conservative estimate
  - Explanation written

- [ ] **A6.2**: TimeSeriesSplit demonstration
  - TimeSeriesSplit applied to dataset
  - Fold boundaries plotted (matplotlib figure)
  - Fold metrics computed
  - Short explanation of temporal splitting written
  - Plot saved to `images/` folder

---

## Notebook 06: Summary & Quiz

### Acceptance Criteria

- [ ] **A7.1**: Permutation importance definition
  - Written in own words
  - One case where it can mislead explained
  - 3-6 lines of clear prose

- [ ] **A7.2**: Ridge vs Lasso preference
  - When to prefer Ridge explained
  - When to prefer Lasso explained
  - Why each is chosen
  - 3-6 lines of clear prose

- [ ] **A7.3**: PCA insights
  - What PCA revealed about feature redundancy
  - Interpretation of explained variance
  - 3-6 lines of clear prose

- [ ] **A7.4**: SHAP pitfalls
  - One pitfall of SHAP identified
  - How to mitigate it explained
  - 3-6 lines of clear prose

- [ ] **A7.5**: Overall reflection
  - 12-18 lines total across all answers
  - Clear, concrete statements
  - Demonstrates understanding

---

## General Requirements

- [ ] All plots saved to `images/` folder
- [ ] All code cells have acceptance criteria checked
- [ ] Markdown cells are well-written and explanatory
- [ ] Notebooks run from top to bottom without errors
- [ ] Random seeds set for reproducibility
- [ ] Code is commented where helpful

---

## Success Criteria

A notebook is considered complete when:
1. All acceptance criteria are met
2. All TODOs are implemented
3. All plots are generated and saved
4. Markdown explanations are clear and educational
5. Code runs without errors
6. Results are interpretable and match expectations

---

**Note**: This is a learning dojo. If you're stuck, re-read the Markdown explanations and hints in the TODO cells. The goal is understanding, not just completion.

