from typing import Dict

import torch
import torch.nn as nn

from hal.data.constants import ACTION_BY_IDX
from hal.data.constants import BUTTON_BY_IDX
from hal.data.constants import CHARACTER_BY_IDX
from hal.data.constants import STAGE_BY_IDX
from hal.data.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.training.zoo.models.registry import Arch


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.stage_embed = nn.Embedding(len(STAGE_BY_IDX), 4)
        self.character_embed = nn.Embedding(len(CHARACTER_BY_IDX), 6)
        self.action_embed = nn.Embedding(len(ACTION_BY_IDX), 20)
        self.core = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.button_head = nn.Linear(hidden_size, len(BUTTON_BY_IDX))
        self.main_stick_head = nn.Linear(hidden_size, len(STICK_XY_CLUSTER_CENTERS_V0))
        self.c_stick_head = nn.Linear(hidden_size, len(STICK_XY_CLUSTER_CENTERS_V0))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        stage_embed = self.stage_embed(inputs["stage"])
        ego_character_embed = self.character_embed(inputs["ego_character"])
        opponent_character_embed = self.character_embed(inputs["opponent_character"])
        ego_action_embed = self.action_embed(inputs["ego_action"])
        opponent_action_embed = self.action_embed(inputs["opponent_action"])
        gamestate = inputs["gamestate"]

        hidden, cell = (
            torch.zeros(self.num_layers, 1, self.hidden_size),
            torch.zeros(self.num_layers, 1, self.hidden_size),
        )
        x = torch.cat(
            [
                stage_embed,
                ego_character_embed,
                opponent_character_embed,
                ego_action_embed,
                opponent_action_embed,
                gamestate,
            ],
            dim=-1,
        )
        x, (hidden, cell) = self.core(x, (hidden, cell))

        button_logits = self.button_head(hidden)
        main_stick_logits = self.main_stick_head(hidden)
        c_stick_logits = self.c_stick_head(hidden)

        button_pred = torch.argmax(button_logits, dim=2)
        main_stick_pred = torch.argmax(main_stick_logits, dim=2)
        c_stick_pred = torch.argmax(c_stick_logits, dim=2)

        return {
            "buttons": button_pred,
            "main_stick": main_stick_pred,
            "c_stick": c_stick_pred,
        }


Arch.register("lstm", make_net=LSTM, input_size=76, hidden_size=256, num_layers=2)
