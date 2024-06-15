from typing import List
from typing import Optional

from melee import GameState
from pyarrow import schema

REPLAY_PARQUET_SCHEMA = schema(
    [
        pyarrow.field("id", pyarrow.int64()),
        pyarrow.field("stage", pyarrow.string()),
        pyarrow.field("frame_count", pyarrow.int64()),
        pyarrow.field(
            "player1",
            pyarrow.struct(
                [
                    pyarrow.field("character", pyarrow.string()),
                    pyarrow.field("nickname", pyarrow.string()),
                    pyarrow.field("pos_x", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("pos_y", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("percent", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("shield", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("stock", pyarrow.list_(pyarrow.int64())),
                    pyarrow.field("facing", pyarrow.list_(pyarrow.bool_())),
                    pyarrow.field("action", pyarrow.list_(pyarrow.int64())),
                    pyarrow.field("invulnerable", pyarrow.list_(pyarrow.bool_())),
                    pyarrow.field("jumps_left", pyarrow.list_(pyarrow.int64())),
                    pyarrow.field("on_ground", pyarrow.list_(pyarrow.bool_())),
                    pyarrow.field("ecb_right", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("ecb_left", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("ecb_top", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("ecb_bottom", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("speed_air_x_self", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("speed_y_self", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("speed_x_attack", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("speed_y_attack", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("speed_ground_x_self", pyarrow.list_(pyarrow.float64())),
                ]
            ),
        ),
        pyarrow.field(
            "player2",
            pyarrow.struct(
                [
                    pyarrow.field("character", pyarrow.string()),
                    pyarrow.field("nickname", pyarrow.string()),
                    pyarrow.field("pos_x", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("pos_y", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("percent", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("shield", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("stock", pyarrow.list_(pyarrow.int64())),
                    pyarrow.field("facing", pyarrow.list_(pyarrow.bool_())),
                    pyarrow.field("action", pyarrow.list_(pyarrow.int64())),
                    pyarrow.field("invulnerable", pyarrow.list_(pyarrow.bool_())),
                    pyarrow.field("jumps_left", pyarrow.list_(pyarrow.int64())),
                    pyarrow.field("on_ground", pyarrow.list_(pyarrow.bool_())),
                    pyarrow.field("ecb_right", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("ecb_left", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("ecb_top", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("ecb_bottom", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("speed_air_x_self", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("speed_y_self", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("speed_x_attack", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("speed_y_attack", pyarrow.list_(pyarrow.float64())),
                    pyarrow.field("speed_ground_x_self", pyarrow.list_(pyarrow.float64())),
                ]
            ),
        ),
    ]
)


def process_slp_file(slp_file_path: str) -> None:
    """Process an SLP file and save the data to a Parquet file."""
    ...
