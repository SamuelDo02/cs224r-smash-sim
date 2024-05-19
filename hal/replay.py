from typing import Optional

import attr
import numpy as np


@attr.s(auto_attribs=True, frozen=True)
class PlayerState:
    """Stack player state attributes across replay frames."""
    character: str
    nickname: str
    pos_x: np.ndarray[float]
    pos_y: np.ndarray[float]
    percent: np.ndarray[float]
    shield: np.ndarray[float]
    stock: np.ndarray[int]
    facing: np.ndarray[bool]
    action: np.ndarray[int]
    invulnerable: np.ndarray[bool]
    jumps_left: np.ndarray[int]
    on_ground: np.ndarray[bool]
    ecb_right: Optional[np.ndarray[float]] = None
    ecb_left: Optional[np.ndarray[float]] = None
    ecb_top: Optional[np.ndarray[float]] = None
    ecb_bottom: Optional[np.ndarray[float]] = None
    speed_air_x_self: Optional[np.ndarray[float]] = None
    speed_y_self: Optional[np.ndarray[float]] = None
    speed_x_attack: Optional[np.ndarray[float]] = None
    speed_y_attack: Optional[np.ndarray[float]] = None
    speed_ground_x_self: Optional[np.ndarray[float]] = None


@attr.s(auto_attribs=True, frozen=True)
class Replay:
    id: int
    stage: str
    frame_count: int
    player1: PlayerState
    player2: PlayerState
