import attr
import numpy as np
import numpy.typing as npt


@attr.s(auto_attribs=True, frozen=True)
class ModelOutputs:
    # v0
    main_stick: npt.NDArray[np.int_]
    c_stick: npt.NDArray[np.int_]
    buttons: npt.NDArray[np.int_]


@attr.s(auto_attribs=True, frozen=True)
class ModelInputs:
    # v0
    stage: npt.NDArray[np.int_]
    ego_character: npt.NDArray[np.int_]
    ego_action: npt.NDArray[np.int_]
    opponent_character: npt.NDArray[np.int_]
    opponent_action: npt.NDArray[np.int_]
    gamestate: npt.NDArray[np.float32]
