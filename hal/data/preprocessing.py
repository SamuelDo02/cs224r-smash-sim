import numpy as np


def preprocess_shield(array: np.ndarray) -> np.ndarray:
    """Preprocess the shield feature."""
    return array - 60.0
