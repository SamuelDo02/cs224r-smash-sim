import melee
import torch
from loguru import logger
from tensordict import TensorDict

from hal.data.schema import PYARROW_DTYPE_BY_COLUMN
from hal.training.config import EmbeddingConfig


def send_controller_inputs(controller: melee.Controller, inputs: TensorDict, idx: int = -1) -> None:
    """
    Press buttons and tilt analog sticks given a dictionary of array-like values (length T for T future time steps).

    Args:
        controller_inputs (Dict[str, torch.Tensor]): Dictionary of array-like values.
        controller (melee.Controller): Controller object.
        idx (int): Index in the arrays to send.
    """
    if idx >= 0:
        assert idx < len(inputs["main_stick_x"])

    controller.tilt_analog(
        melee.Button.BUTTON_MAIN,
        inputs["main_stick_x"][idx].item(),
        inputs["main_stick_y"][idx].item(),
    )
    controller.tilt_analog(
        melee.Button.BUTTON_C,
        inputs["c_stick_x"][idx].item(),
        inputs["c_stick_y"][idx].item(),
    )
    for button, state in inputs.items():
        if button.startswith("button") and button != "button_none" and state[idx].item() == 1:
            controller.press_button(getattr(melee.Button, button.upper()))
            logger.info(f"Pressed {button}")
            break
    controller.flush()


def mock_framedata_as_tensordict(seq_len: int) -> TensorDict:
    """Mock `seq_len` frames of gamestate data."""
    return TensorDict({k: torch.zeros(seq_len) for k in PYARROW_DTYPE_BY_COLUMN}, batch_size=(seq_len,))


def mock_preds_as_tensordict(embed_config: EmbeddingConfig) -> TensorDict:
    """Mock a single model prediction."""
    assert embed_config.num_buttons is not None
    assert embed_config.num_main_stick_clusters is not None
    assert embed_config.num_c_stick_clusters is not None
    return TensorDict(
        {
            "buttons": torch.zeros(embed_config.num_buttons),
            "main_stick": torch.zeros(embed_config.num_main_stick_clusters),
            "c_stick": torch.zeros(embed_config.num_c_stick_clusters),
        },
        batch_size=(),
    )


def share_and_pin_memory(tensordict: TensorDict) -> TensorDict:
    """Share and pin memory of a tensordict."""
    tensordict.share_memory_()

    cudart = torch.cuda.cudart()
    if cudart is None:
        return tensordict

    for tensor in tensordict.flatten_keys().values():
        assert isinstance(tensor, torch.Tensor)
        cudart.cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
        assert tensor.is_shared()
        assert tensor.is_pinned()

    return tensordict
