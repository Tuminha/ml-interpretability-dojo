"""
Utility functions for reproducibility and common tasks.

This module provides functions to set random seeds across different libraries,
ensuring reproducible results across runs and notebooks.
"""


def set_seed(seed=42):
    """
    Set random seeds for numpy, Python's random module, and optionally other libraries.
    
    Reproducibility is crucial in machine learning. Setting seeds ensures that
    random operations (data shuffling, train/test splits, model initialization)
    produce the same results across runs, making experiments comparable.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value (42 is a common choice, popularized by "The Hitchhiker's Guide")
    
    Returns
    -------
    None
    
    Example
    -------
    >>> set_seed(42)
    >>> # Now all random operations will be reproducible
    >>> import numpy as np
    >>> np.random.rand(5)  # Same values every time
    
    Notes
    -----
    This function sets seeds for:
    - numpy.random
    - Python's random module
    
    For scikit-learn, use random_state parameter in functions.
    For PyTorch, you may need additional torch.manual_seed() calls.
    
    TODO: Implement this function
    - Import random and numpy
    - Set random.seed(seed)
    - Set np.random.seed(seed)
    - Optionally set environment variable PYTHONHASHSEED for Python 3.7+
    
    Acceptance:
    - Function sets numpy random seed
    - Function sets Python random seed
    - No return value (or returns None)
    - Can be called at the start of notebooks for reproducibility
    """
    # === TODO (you code this) ===
    # Set random seeds for reproducibility
    # Hints:
    #   import random
    #   import numpy as np
    #   random.seed(seed)
    #   np.random.seed(seed)
    #   Optional: os.environ['PYTHONHASHSEED'] = str(seed)
    
    raise NotImplementedError("Implement set_seed()")


def get_random_state(seed=42):
    """
    Return a numpy RandomState object for fine-grained control.
    
    Sometimes you want a RandomState object instead of using the global
    numpy random state. This is useful when you need multiple independent
    random streams.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed
    
    Returns
    -------
    numpy.random.RandomState
        A RandomState object with the specified seed
    
    Example
    -------
    >>> rng = get_random_state(42)
    >>> values = rng.rand(10)  # Uses this specific RandomState
    """
    # === TODO (optional) ===
    # Return np.random.RandomState(seed)
    # Useful for advanced use cases
    
    raise NotImplementedError("Optional: implement get_random_state()")

