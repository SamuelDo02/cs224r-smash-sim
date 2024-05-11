from typing import Literal


DEVICES = Literal["cpu", "cuda", "mps"]
EVAL_MODE = Literal["cpu", "model"]
EVAL_STAGES = Literal["all", "fd", "bf", "ps", "dl", "fod", "ys"]
