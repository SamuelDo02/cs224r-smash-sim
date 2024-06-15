from typing import Optional
from typing import Tuple

import melee
import pyarrow

REPLAY_PARQUET_SCHEMA = pyarrow.schema(
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


def determine_player_ports(gamestate: melee.GameState) -> Tuple[Optional[int], Optional[int]]:
    """
    Determine the player ports based on the given gamestate.

    Args:
        gamestate (melee.GameState): The current game state.

    Returns:
        Tuple[Optional[int], Optional[int]]: The ports of player1 and player2.

    Examples:
        >>> from melee import GameState, Player
        >>> gamestate = GameState()
        >>> gamestate.players = {1: Player(), 2: None, 3: Player(), 4: None}
        >>> gamestate.players[1].character = "Fox"
        >>> gamestate.players[3].character = "Falco"
        >>> determine_player_ports(gamestate)
        (1, 3)

        >>> gamestate.players = {1: None, 2: None, 3: Player(), 4: None}
        >>> gamestate.players[3].character = "Falco"
        >>> determine_player_ports(gamestate)
        (3, None)

        >>> gamestate.players = {1: None, 2: None, 3: None, 4: None}
        >>> determine_player_ports(gamestate)
        (None, None)
    """
    player1_port = None
    player2_port = None

    for port, player in gamestate.players.items():
        if player is not None and player.character is not None:
            if player1_port is None:
                player1_port = port
            elif player2_port is None:
                player2_port = port

    return player1_port, player2_port


def slp_to_pyarrow_table(slp_file_path: str) -> pyarrow.Table:
    console = melee.Console(is_dolphin=False, path=slp_file_path)
    console.connect()

    data = {
        "id": [],
        "stage": [],
        "frame_count": [],
        "player1": {key: [] for key in REPLAY_PARQUET_SCHEMA.field("player1").type},
        "player2": {key: [] for key in REPLAY_PARQUET_SCHEMA.field("player2").type},
    }

    while True:
        gamestate = console.step()
        if gamestate is None:
            break

        player1_port, player2_port = determine_player_ports(gamestate)
        for player_port, player_key in [(player1_port, "player1"), (player2_port, "player2")]:
            if player_port is not None:
                player = gamestate.players[player_port]
                player_data = data[player_key]
                player_data["character"].append(player.character)
                player_data["nickname"].append(player.nickName)
                player_data["pos_x"].append(player.position.x)
                player_data["pos_y"].append(player.position.y)
                player_data["percent"].append(player.percent)
                player_data["shield"].append(player.shield_strength)
                player_data["stock"].append(player.stock)
                player_data["facing"].append(player.facing)
                player_data["action"].append(player.action)
                player_data["invulnerable"].append(player.invulnerable)
                player_data["jumps_left"].append(player.jumps_left)
                player_data["on_ground"].append(player.on_ground)
                player_data["ecb_right"].append(player.ecb_right)
                player_data["ecb_left"].append(player.ecb_left)
                player_data["ecb_top"].append(player.ecb_top)
                player_data["ecb_bottom"].append(player.ecb_bottom)
                player_data["speed_air_x_self"].append(player.speed_air_x_self)
                player_data["speed_y_self"].append(player.speed_y_self)
                player_data["speed_x_attack"].append(player.speed_x_attack)
                player_data["speed_y_attack"].append(player.speed_y_attack)
                player_data["speed_ground_x_self"].append(player.speed_ground_x_self)

    table = pyarrow.Table.from_pydict(data, schema=REPLAY_PARQUET_SCHEMA)
    return table


## use the libmelee docs here:
