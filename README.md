# 🥋 ML Interpretability Dojo

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Scikit--learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Mastering the Art of Understanding Your Models**

[🎯 Overview](#-the-journey) • [📚 Notebooks](#-the-path) • [🚀 Quick Start](#-getting-started) • [📊 Concepts](#-core-concepts)

</div>

---

## 👨‍💻 Author

<div align="center">

**Francisco Teixeira Barbosa**

[![GitHub](https://img.shields.io/badge/GitHub-Tuminha-black?style=flat&logo=github)](https://github.com/Tuminha)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/franciscotbarbosa)
[![Email](https://img.shields.io/badge/Email-cisco%40periospot.com-blue?style=flat&logo=gmail)](mailto:cisco@periospot.com)
[![Twitter](https://img.shields.io/badge/Twitter-cisco__research-1DA1F2?style=flat&logo=twitter)](https://twitter.com/cisco_research)

*Building durable intuition for model interpretability, one concept at a time*

</div>

---

## 🎯 The Journey

Imagine you've built a machine learning model that predicts patient outcomes with 95% accuracy. Impressive, right? But here's the question that keeps data scientists awake at night: **Why does it work?** What features drive the predictions? Can we trust it when it matters most?

This dojo is born from a simple realization: **understanding your model is not optional—it's essential**. Whether you're deploying in healthcare, finance, or any domain where decisions have consequences, interpretability bridges the gap between black-box predictions and actionable insights.

Through seven carefully crafted notebooks, we'll explore four pillars of model understanding:

1. **Invariance and Baselines** — Establishing what "normal" looks like
2. **Permutation Importance** — Measuring feature impact through controlled chaos
3. **Regularization (Ridge & Lasso)** — Taming complexity and selecting what matters
4. **Multicollinearity & PCA** — Diagnosing redundancy and finding structure
5. **SHAP Values** — Unpacking predictions into feature contributions
6. **Cross-Validation & Leakage** — Ensuring our insights are real, not artifacts

Each notebook is designed as a learning journey: 70% explanation, 30% guided implementation. You'll code the solutions yourself, because **understanding comes from doing**.

---

## 📚 The Path

### Notebook 00: Invariance and Baselines
**The Foundation** — Before we interpret complex models, we must understand what makes a model "well-behaved." Invariance teaches us that row order shouldn't matter. Baselines teach us what "random" looks like. Together, they form the bedrock of model evaluation.

**Key Concepts:**
- Row-order invariance in tabular models
- Mean predictor and majority class baselines
- Reproducibility through random seeds

### Notebook 01: Permutation Importance
**Breaking the Connection** — What happens when we randomly shuffle a feature? If the model's performance barely changes, that feature carries little signal. Permutation importance quantifies this intuition, giving us a model-agnostic way to rank features by their predictive power.

**Key Concepts:**
- Feature-target association breaking
- Metric drop as importance signal
- Difference from label permutation tests

### Notebook 02: Regularization (Ridge & Lasso)
**The Art of Restraint** — Overfitting is the enemy of generalization. Regularization tames it by penalizing complexity. Ridge shrinks coefficients smoothly. Lasso can zero them out entirely, performing automatic feature selection. Understanding when to use each is a superpower.

**Key Concepts:**
- Bias-variance tradeoff
- L1 (Lasso) vs L2 (Ridge) penalties
- Coefficient shrinkage vs selection
- Cross-validation for hyperparameter tuning

### Notebook 03: Multicollinearity & PCA
**The Redundancy Problem** — When features are highly correlated, coefficients become unstable and hard to interpret. Variance Inflation Factor (VIF) diagnoses the problem. Principal Component Analysis (PCA) reveals the underlying structure, rotating data to uncorrelated axes.

**Key Concepts:**
- Correlation matrices and VIF interpretation
- PCA as a diagnostic tool
- Explained variance and component loadings
- When to use PCA vs feature engineering

### Notebook 04: SHAP Values
**The Attribution Game** — SHAP (SHapley Additive exPlanations) answers: "How much did each feature contribute to this specific prediction?" It's based on game theory, ensuring local attributions sum to the prediction minus baseline. Different explainers for different models, each with tradeoffs.

**Key Concepts:**
- Shapley values from cooperative game theory
- LinearExplainer for linear models
- TreeExplainer for tree ensembles
- Local vs global interpretability
- Performance considerations and sampling

### Notebook 05: Cross-Validation & Leakage
**The Validation Trap** — Data leakage is the silent killer of model insights. When training and test sets share information they shouldn't, metrics become misleading. GroupKFold and TimeSeriesSplit prevent this, ensuring our interpretations reflect reality.

**Key Concepts:**
- KFold vs GroupKFold vs TimeSeriesSplit
- Temporal and group-based leakage
- Conservative vs optimistic estimates

### Notebook 06: Summary & Reflection
**Synthesizing Knowledge** — A quiz notebook to reinforce concepts through written reflection. In your own words, explain what you've learned and when each technique applies.

---

## 🚀 Getting Started

### Prerequisites

```bash
# Clone the repository
git clone <your-repo-url>
cd ML_dojo_project

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Workflow

1. **Start with Notebook 00** and work sequentially through 06
2. **Read the Markdown cells first** — they contain the conceptual foundation
3. **Implement the TODO cells** — code the solutions yourself for deeper retention
4. **Check acceptance criteria** — each TODO includes a checklist
5. **Save your plots** — all visualizations should be saved to `images/` folder
6. **Reflect in Notebook 06** — solidify your understanding through writing

### Quick Test

```bash
# Verify installation
python -c "import sklearn, shap, xgboost; print('All dependencies installed!')"

# Launch Jupyter
jupyter notebook
```

---

## 🧠 Core Concepts

### Why Interpretability Matters

In healthcare AI (like Periospot), interpretability isn't nice-to-have—it's a requirement. Clinicians need to understand why a model suggests a diagnosis. Regulators need to audit decisions. Patients deserve transparency.

But interpretability also makes you a better data scientist:
- **Debugging**: When a model fails, interpretability shows you where
- **Feature Engineering**: Understanding what matters guides new features
- **Trust**: Stakeholders trust models they can understand
- **Compliance**: Many regulations (GDPR, FDA) require explainability

### The Four Pillars

1. **Permutation Importance** — Model-agnostic, intuitive, fast. Best for initial feature ranking.
2. **Regularization** — Built into training, prevents overfitting, Lasso selects features.
3. **PCA** — Diagnostic tool for redundancy, dimensionality reduction, visualization.
4. **SHAP** — Local and global explanations, theoretically grounded, model-specific.

Each has strengths and weaknesses. The dojo teaches you when to use which.

---

## 📊 Project Structure

```
ml-interpretability-dojo/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── data/
│   ├── raw/                 # Raw datasets (optional)
│   └── processed/           # Processed data (gitignored)
├── notebooks/
│   ├── 00_invariance_and_baselines.ipynb
│   ├── 01_permutation_importance.ipynb
│   ├── 02_regularization_ridge_lasso.ipynb
│   ├── 03_multicollinearity_vif_pca.ipynb
│   ├── 04_shap_linear_and_trees.ipynb
│   ├── 05_cv_schemes_and_leakage.ipynb
│   └── 06_summary_quiz.ipynb
├── src/
│   ├── __init__.py
│   ├── loaders.py           # Data loading utilities
│   ├── metrics.py           # Metric reporting functions
│   ├── viz.py               # Visualization helpers
│   └── utils.py             # Utility functions (seeds, etc.)
├── tests/
│   └── smoke_test_plan.md   # Acceptance criteria
└── images/                  # Saved plots and visualizations
```

---

## 🎓 Learning Philosophy

This dojo follows a **guided discovery** approach:

- **70% Explanation, 30% Implementation** — Concepts come first
- **TODOs with Acceptance Criteria** — Clear goals, not full solutions
- **Self-Coding** — You implement to retain knowledge
- **Progressive Complexity** — Each notebook builds on the last

The goal isn't to copy-paste code—it's to build **durable intuition** that you'll carry into real projects.

---

## 📈 Expected Outcomes

After completing this dojo, you will:

- ✅ Understand when and why to use permutation importance
- ✅ Know when Ridge vs Lasso is appropriate
- ✅ Diagnose multicollinearity and interpret PCA results
- ✅ Generate SHAP explanations for linear and tree models
- ✅ Avoid data leakage through proper cross-validation
- ✅ Have a toolkit of interpretability techniques for production

---

## 🔬 Optional Extensions

Once you've mastered the basics, consider:

- **Stability Selection** — Which features survive across Lasso resamples?
- **Partial Dependence Plots** — Visualize feature effects on predictions
- **Permutation Importance Variance** — How stable are importance rankings?
- **Model Deployment** — Save models to Hugging Face or MLflow
- **Real-World Dataset** — Apply techniques to `data/raw/tennis_stats.csv`

---

## 📝 License

MIT License — feel free to use this dojo for learning and teaching.

---

## 🙏 Acknowledgments

This dojo is inspired by the need for practical, hands-on interpretability education. Special thanks to the scikit-learn, SHAP, and XGBoost communities for building the tools that make this possible.

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**

*Building interpretable AI, one concept at a time* 🚀

</div>

