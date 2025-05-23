from typing import Any
from typing import Callable
from typing import Dict

import attr
from tensordict import TensorDict


@attr.s(auto_attribs=True)
class PostprocessConfig:
    """Configuration for how we convert model predictions to controller inputs."""

    transformation_by_controller_input: Dict[str, Callable[[TensorDict], Any]]
