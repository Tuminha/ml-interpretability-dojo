"""
Metric computation and reporting utilities.

This module provides functions to compute and display regression and classification
metrics in a clean, readable format for notebooks and reports.
"""


def regression_report(y_true, y_pred, label="test"):
    """
    Print RMSE, MAE, and R² neatly in one formatted line.
    
    This function computes common regression metrics and displays them
    in a human-readable format. Useful for quick comparisons across models.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    label : str, default="test"
        Label to include in the output (e.g., "train", "test", "val")
    
    Returns
    -------
    dict
        Dictionary containing computed metrics:
        - 'rmse': Root Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'r2': R-squared coefficient of determination
    
    Example
    -------
    >>> from sklearn.metrics import mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> metrics = regression_report(y_true, y_pred, label="validation")
    >>> # Prints: [validation] RMSE: 0.612, MAE: 0.500, R²: 0.957
    
    Notes
    -----
    RMSE penalizes large errors more than MAE, making it sensitive to outliers.
    R² measures the proportion of variance explained (1.0 = perfect, 0.0 = baseline).
    
    TODO: Implement this function
    - Import necessary metrics from sklearn.metrics
    - Compute RMSE (sqrt of mean_squared_error)
    - Compute MAE (mean_absolute_error)
    - Compute R² (r2_score)
    - Print formatted string: [label] RMSE: X.XXX, MAE: X.XXX, R²: X.XXX
    - Return dictionary with all three metrics
    
    Acceptance:
    - Function prints formatted metrics string
    - Returns dict with keys 'rmse', 'mae', 'r2'
    - All metrics are float values
    - Output format matches: [label] RMSE: X.XXX, MAE: X.XXX, R²: X.XXX
    """
    # === TODO (you code this) ===
    # Compute and print RMSE, MAE, R²
    # Hints: 
    #   from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    #   RMSE = sqrt(mean_squared_error(y_true, y_pred))
    #   Use f-strings for formatting
    #   Round to 3 decimal places
    
    raise NotImplementedError("Implement regression_report()")


def classification_report_basic(y_true, y_pred, label="test"):
    """
    Print basic classification metrics: accuracy, precision, recall, F1.
    
    Parameters
    ----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    label : str, default="test"
        Label for output
    
    Returns
    -------
    dict
        Dictionary with accuracy, precision, recall, f1
    
    Notes
    -----
    This is a simplified version. For multi-class, use sklearn's
    classification_report for per-class metrics.
    """
    # === TODO (optional) ===
    # Implement if needed for classification tasks
    # Hints: from sklearn.metrics import accuracy_score, precision_score, etc.
    
    raise NotImplementedError("Optional: implement classification_report_basic()")

