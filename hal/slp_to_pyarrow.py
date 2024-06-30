import multiprocessing as mp
import os
from typing import List
from typing import Optional

import attr
import melee
import pyarrow as pa
import pyarrow.dataset as ds
from loguru import logger


@attr.s(auto_attribs=True, frozen=True)
class FrameData:
    frame: int
    stage: str
    player1_character: Optional[str]
    player2_character: Optional[str]
    player1_x: Optional[float]
    player1_y: Optional[float]
    player2_x: Optional[float]
    player2_y: Optional[float]
    player1_percent: Optional[float]
    player2_percent: Optional[float]
    player1_stock: Optional[int]
    player2_stock: Optional[int]


def extract_frame_data(gamestate: melee.GameState) -> FrameData:
    return FrameData(
        frame=gamestate.frame,
        stage=gamestate.stage.name,
        player1_character=gamestate.players[1].character.name if 1 in gamestate.players else None,
        player2_character=gamestate.players[2].character.name if 2 in gamestate.players else None,
        player1_x=gamestate.players[1].x if 1 in gamestate.players else None,
        player1_y=gamestate.players[1].y if 1 in gamestate.players else None,
        player2_x=gamestate.players[2].x if 2 in gamestate.players else None,
        player2_y=gamestate.players[2].y if 2 in gamestate.players else None,
        player1_percent=gamestate.players[1].percent if 1 in gamestate.players else None,
        player2_percent=gamestate.players[2].percent if 2 in gamestate.players else None,
        player1_stock=gamestate.players[1].stock if 1 in gamestate.players else None,
        player2_stock=gamestate.players[2].stock if 2 in gamestate.players else None,
    )


def process_replay(replay_path: str) -> List[FrameData]:
    console = melee.Console(path=replay_path, is_dolphin=False, allow_old_version=True)
    try:
        console.connect()
    except Exception as e:
        print(f"Error connecting to console: {e}")
        return []

    try:
        gamestate = console.step()
    except Exception as e:
        logger.debug(f"{e}. Could not read gamestate from {replay_path}.")
        return None


def write_dataset(data: List[FrameData], output_dir: str) -> None:
    table = pa.Table.from_pylist([vars(frame) for frame in data])
    ds.write_dataset(table, output_dir, format="parquet", partitioning=["stage"])


def process_and_write(replay_path: str, output_dir: str) -> None:
    data = process_replay(replay_path)
    write_dataset(data, os.path.join(output_dir, os.path.basename(replay_path)))


def process_replays(replay_paths: List[str], output_dir: str) -> None:
    with mp.Pool() as pool:
        pool.starmap(process_and_write, [(path, output_dir) for path in replay_paths])


# if __name__ == "__main__":
# process_replays(replay_paths, output_path)
