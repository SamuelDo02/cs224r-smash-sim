import numpy as np

from hal.data.constants import STICK_XY_CLUSTER_CENTERS_V0


def one_hot_3d_fast_bugged(arr: np.ndarray) -> np.ndarray:
    """
    One-hot encode 3d array of raw button presses.

    Vectorized but slightly wrong; this takes the left-most 1 in each row instead of the newest button press, so buttons have a cardinal order of priority.

    Sets the last button to 1 if there are no 1s in the array.

    Args:
        arr (np.ndarray): Input array of shape (B, T, D) where B is the batch size,
                          T is the number of time steps, and D is the number of buttons + 1.

    Returns:
        np.ndarray: One-hot encoded array of the same shape (B, T, D).
    """
    # Find where 1s start in each row
    start_mask = np.concatenate([arr[:, :, 0:1], arr[:, :, 1:] > arr[:, :, :-1]], axis=2)
    # Compute cumulative sum to identify streaks
    cumsum = np.cumsum(start_mask, axis=2)
    streak_ids = cumsum * arr
    # Find the maximum streak ID for each row
    max_streak_ids = np.max(streak_ids, axis=2, keepdims=True)
    # Create a mask for the old streak in each row
    old_streak_mask = (streak_ids == max_streak_ids) & (streak_ids > 1) & (arr == 1)
    processed = arr * ~old_streak_mask
    # Handle rows with no 1s
    no_ones_mask = ~np.any(arr, axis=2)
    processed[no_ones_mask, -1] = 1

    return processed


def one_hot_2d_fast(arr: np.ndarray) -> np.ndarray:
    """
    One-hot encode 2D array of raw button presses.

    Vectorized but slightly wrong; this takes the left-most 1 in each row instead of the newest button press, so buttons have a cardinal order of priority.

    Sets the last button to 1 if there are no 1s in the array.

    Args:
        arr (np.ndarray): Input array of shape (T, D) where T is the number of time steps
                          and D is the number of buttons.

    Returns:
        np.ndarray: One-hot encoded array of the same shape (T, D).
    """
    # Find where 1s start in each row
    start_mask = np.concatenate([arr[:, 0:1], arr[:, 1:] > arr[:, :-1]], axis=-1)
    # Compute cumulative sum to identify streaks
    cumsum = np.cumsum(start_mask, axis=-1)
    streak_ids = cumsum * arr
    # Find the maximum streak ID for each row
    max_streak_ids = np.max(streak_ids, axis=-1, keepdims=True)
    # Create a mask for the old streak in each row
    old_streak_mask = (streak_ids == max_streak_ids) & (streak_ids > 1) & (arr == 1)
    # Process the array
    processed = arr * ~old_streak_mask
    # Handle rows with no 1s
    no_ones_mask = ~np.any(arr, axis=-1)
    processed[no_ones_mask, -1] = 1

    return processed


def get_closest_stick_xy_cluster_v0(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the closest point in STICK_XY_CLUSTER_CENTERS_V0 for given x and y values.

    Args:
        x (np.ndarray): (T,) X-coordinates in range [0, 1]
        y (np.ndarray): (T,) Y-coordinates in range [0, 1]

    Returns:
        np.ndarray: (T,) Indices of the closest cluster centers
    """
    point = np.stack((x, y), axis=-1)  # Shape: (T, 2)
    distances = np.sum((STICK_XY_CLUSTER_CENTERS_V0 - point[:, np.newaxis, :]) ** 2, axis=-1)
    return np.argmin(distances, axis=-1)
