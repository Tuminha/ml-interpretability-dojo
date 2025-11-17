"""
Visualization utilities for the ML Interpretability Dojo.

This module provides helper functions for creating consistent, publication-ready
plots that are automatically saved to the images/ folder.
"""

import matplotlib.pyplot as plt
import os
from pathlib import Path


def get_images_dir():
    """
    Get the absolute path to the images directory.
    
    Returns
    -------
    Path
        Path object pointing to the images/ directory in the project root
    """
    # Get the project root (parent of src/)
    project_root = Path(__file__).parent.parent
    images_dir = project_root / "images"
    return images_dir


def barh(values, labels, title, save_path=None, figsize=(8, 6)):
    """
    Create a horizontal bar plot for feature importances or rankings.
    
    Horizontal bar plots are ideal for displaying feature importance because
    they allow easy reading of feature names and comparison of values.
    
    Parameters
    ----------
    values : array-like
        Values to plot (e.g., importances, coefficients)
    labels : array-like
        Labels for each bar (e.g., feature names)
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure. If None, saves to images/ folder with title-based filename
    figsize : tuple, default=(8, 6)
        Figure size (width, height)
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    
    Example
    -------
    >>> importances = [0.3, 0.25, 0.2, 0.15, 0.1]
    >>> features = ['age', 'bmi', 'bp', 's1', 's2']
    >>> fig = barh(importances, features, "Feature Importances")
    >>> # Saves to images/feature_importances.png
    
    Notes
    -----
    - Inverts y-axis so highest values appear at top
    - Automatically saves to images/ folder if save_path not provided
    - Creates images/ directory if it doesn't exist
    
    TODO: Implement this function
    - Create figure and axis with figsize
    - Use ax.barh() with values and labels
    - Invert y-axis: ax.invert_yaxis()
    - Set title, xlabel, ylabel
    - Tight layout: plt.tight_layout()
    - Save to images/ folder (create directory if needed)
    - Return figure object
    
    Acceptance:
    - Function creates horizontal bar plot
    - Y-axis is inverted (top = highest value)
    - Figure is saved to images/ folder
    - Returns matplotlib Figure object
    - Plot is readable with clear labels
    """
    # === TODO (you code this) ===
    # Create horizontal bar plot
    # Hints:
    #   fig, ax = plt.subplots(figsize=figsize)
    #   ax.barh(labels, values)
    #   ax.invert_yaxis()  # Top = highest
    #   os.makedirs('images', exist_ok=True)  # Create folder if needed
    #   save_path = save_path or f"images/{title.lower().replace(' ', '_')}.png"
    #   plt.savefig(save_path, dpi=150, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, values)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    
    # Get images directory and create if needed
    images_dir = get_images_dir()
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate save path if not provided
    if save_path is None:
        filename = title.lower().replace(' ', '_').replace('/', '_') + '.png'
        save_path = images_dir / filename
    else:
        # If relative path provided, make it relative to images dir
        if not os.path.isabs(save_path):
            save_path = images_dir / save_path
    
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"✓ Figure saved to: {save_path}")
    return fig


def save_figure(fig, filename, dpi=150):
    """
    Save a matplotlib figure to the images/ folder.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename (will be saved in images/ folder)
    dpi : int, default=150
        Resolution for saved figure
    
    Notes
    -----
    This is a convenience function to ensure all plots are saved consistently.
    Always use this or include save logic in your plotting functions.
    """
    # === TODO (optional helper) ===
    # Create images/ directory if needed
    # Save figure with high DPI
    # Hints: os.makedirs('images', exist_ok=True); fig.savefig(...)

    # Get images directory and create if needed
    images_dir = get_images_dir()
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Build full path
    save_path = images_dir / filename
    fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
    print(f"✓ Figure saved to: {save_path}")
    return fig