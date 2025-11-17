"""
Data loading utilities for the ML Interpretability Dojo.

This module provides helper functions to load common datasets used throughout
the notebooks, ensuring consistent data preparation across experiments.
"""


def load_diabetes_df():
    """
    Return a pandas DataFrame for the sklearn diabetes dataset with feature names and target.
    
    The diabetes dataset is a classic regression dataset with 10 baseline variables
    (age, sex, BMI, blood pressure, etc.) and a quantitative measure of disease
    progression one year after baseline.
    
    Returns
    -------
    tuple : (X, y, feature_names)
        X : pandas.DataFrame
            Feature matrix with named columns
        y : pandas.Series
            Target variable (disease progression)
        feature_names : list
            List of feature names for reference
    
    Example
    -------
    >>> X, y, feature_names = load_diabetes_df()
    >>> print(f"Features: {X.shape}, Target: {y.shape}")
    >>> print(f"Feature names: {feature_names}")
    
    Notes
    -----
    This function uses sklearn's load_diabetes with as_frame=True to get
    a pandas DataFrame directly, which is more convenient for exploration
    and visualization.
    
    TODO: Implement this function
    - Import load_diabetes from sklearn.datasets
    - Use as_frame=True to get a DataFrame
    - Extract X (features), y (target), and feature names
    - Return all three as a tuple
    
    Acceptance:
    - Function returns (X, y, feature_names) tuple
    - X is a pandas DataFrame with 10 columns
    - y is a pandas Series
    - feature_names is a list of 10 strings
    """
    # === TODO (you code this) ===
    # Load the diabetes dataset using sklearn.datasets.load_diabetes
    # Hints: from sklearn.datasets import load_diabetes; use as_frame=True
    # Return X (DataFrame), y (Series), and feature_names (list)
    
    raise NotImplementedError("Implement load_diabetes_df()")


def load_tennis_stats(path="data/raw/tennis_stats.csv"):
    """
    Load the optional tennis statistics dataset if available.
    
    Parameters
    ----------
    path : str, default="data/raw/tennis_stats.csv"
        Path to the tennis stats CSV file
    
    Returns
    -------
    pandas.DataFrame or None
        DataFrame if file exists, None otherwise
    
    Notes
    -----
    This is an optional dataset. If the file doesn't exist, return None.
    The function should handle FileNotFoundError gracefully.
    """
    # === TODO (optional) ===
    # Load tennis_stats.csv if it exists
    # Hints: import pandas as pd; use try/except for FileNotFoundError
    
    raise NotImplementedError("Optional: implement load_tennis_stats()")

