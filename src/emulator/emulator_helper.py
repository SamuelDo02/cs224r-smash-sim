from emulator.constants import ORIGINAL_BUTTONS
import melee
from typing import Any, Dict, List

import numpy as np


def process_inputs(actions: np.ndarray) -> Dict[str, Any]:
    """
    Process the actions into a dictionary of controller inputs
    """
    print(f'Actions: {actions}')
    # Map continuous action values to controller inputs
    inputs = {
        "main_stick": [actions[0], actions[1]],  # Main stick X,Y
        "c_stick": [actions[2], actions[3]],     # C-stick X,Y
        "l_shoulder": actions[4],                # L trigger
        "r_shoulder": actions[5],                # R trigger
        "buttons": []                            # List to store pressed buttons
    }

    # Only process buttons if we have enough action dimensions
    if len(actions) >= 12:
        button_map = {
            6: ORIGINAL_BUTTONS[0],
            7: ORIGINAL_BUTTONS[1], 
            8: ORIGINAL_BUTTONS[2],
            9: ORIGINAL_BUTTONS[3],
            10: ORIGINAL_BUTTONS[4],
            11: ORIGINAL_BUTTONS[5]
        }

        # Add pressed buttons to list
        for idx, val in button_map.items():
            if actions[idx] > 0.5:
                inputs["buttons"].append(val)
    return inputs

def send_controller_inputs(controller: melee.Controller, inputs: Dict[str, Any]) -> None:
    """
    Press buttons and tilt analog sticks given a dictionary of array-like values (length T for T future time steps).

    Args:
        controller (melee.Controller): Controller object.
        inputs (Dict[str, Any]): Dictionary of controller inputs
    """
    print(f'Inputs: {inputs}')
    controller.tilt_analog(
        melee.Button.BUTTON_MAIN,
        inputs["main_stick"][0],
        inputs["main_stick"][1],
    )
    controller.tilt_analog(
        melee.Button.BUTTON_C,
        inputs["c_stick"][0],
        inputs["c_stick"][1],
    )

    shoulder_value = inputs.get("l_shoulder", 0)
    controller.press_shoulder(
        melee.Button.BUTTON_L,
        shoulder_value,
    )

    shoulder_value = inputs.get("r_shoulder", 0)
    controller.press_shoulder(
        melee.Button.BUTTON_R,
        shoulder_value,
    )

    buttons_to_press: List[str] = inputs.get("buttons", [])
    for button_str in ORIGINAL_BUTTONS:
        button = getattr(melee.Button, button_str.upper())
        if button_str in buttons_to_press:
            controller.press_button(button)
        else:
            controller.release_button(button)

    controller.flush()