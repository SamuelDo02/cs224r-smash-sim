from typing import Tuple

import torch

ORIGINAL_BUTTONS: Tuple[str, ...] = (
    "BUTTON_A",
    "BUTTON_B",
    "BUTTON_X",
    "BUTTON_Y",
    "BUTTON_Z",
    "BUTTON_D_UP",
)

JOYSTICK_POINTS = [
    # Center
    (0.5, 0.5),

    # Inner ring (radius = 0.25) at angles 0°, 90°, 180°, 270°
    (0.75, 0.5),   # 0°
    (0.5, 0.75),   # 90°
    (0.25, 0.5),   # 180°
    (0.5, 0.25),   # 270°

    # Outer ring (radius = 0.5) at angles 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
    (1.0000, 0.5000),   # 0°
    (0.8536, 0.8536),   # 45°
    (0.5000, 1.0000),   # 90°
    (0.1464, 0.8536),   # 135°
    (0.0000, 0.5000),   # 180°
    (0.1464, 0.1464),   # 225°
    (0.5000, 0.0000),   # 270°
    (0.8536, 0.1464),   # 315°
]

NUM_JOYSTICK_POSITIONS = len(JOYSTICK_POINTS)

def get_closest_joystick_point(x: float, y: float):
    """Find the closest predefined joystick point to the given coordinates, one hot encoded."""
    min_dist = float('inf')
    closest_point = (0.5, 0.5)  # Default to center
    
    for point in JOYSTICK_POINTS:
        dist = (x - point[0])**2 + (y - point[1])**2
        if dist < min_dist:
            min_dist = dist
            closest_point = point
            
    point_idx = JOYSTICK_POINTS.index(closest_point)
    one_hot = torch.zeros(NUM_JOYSTICK_POSITIONS)
    one_hot[point_idx] = 1
    return one_hot