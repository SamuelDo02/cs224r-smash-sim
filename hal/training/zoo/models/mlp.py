from typing import Iterable
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from hal.training.config import TrainConfig
from hal.training.utils import get_nembd_from_config


class MLPBC(nn.Module):
    """
    Simple MLP that predicts next action a from past states s.
    """

    def __init__(self, config: TrainConfig, hidden_size: int, n_layer: int = 4, dropout=0.1) -> None:
        super().__init__()
        data_config = config.data
        embed_config = config.embedding
        assert embed_config.num_buttons is not None
        assert embed_config.num_main_stick_clusters is not None
        assert embed_config.num_c_stick_clusters is not None
        self.n_embd = get_nembd_from_config(embed_config)
        self.max_length = data_config.input_len

        self.modules_by_name = nn.ModuleDict(
            dict(
                stage=nn.Embedding(embed_config.num_stages, embed_config.stage_embedding_dim),
                character=nn.Embedding(embed_config.num_characters, embed_config.character_embedding_dim),
                action=nn.Embedding(embed_config.num_actions, embed_config.action_embedding_dim),
                proj_in=nn.Linear(self.max_length * self.n_embd, hidden_size),
                mlp=nn.ModuleList(
                    [
                        layer
                        for _ in range(n_layer - 1)
                        for layer in [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size)]
                    ]
                ),
            )
        )
        self.button_head = nn.Linear(hidden_size, embed_config.num_buttons)
        self.main_stick_head = nn.Linear(hidden_size, embed_config.num_main_stick_clusters)
        self.c_stick_head = nn.Linear(hidden_size, embed_config.num_c_stick_clusters)

    def forward(
        self,
        inputs: TensorDict,
        hidden_in: Optional[Iterable[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[TensorDict, Iterable[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T, D = inputs["gamestate"].shape
        assert T > 0

        states = inputs[:, -self.max_length :].reshape(B, -1)  # concat states
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)

        return None, actions, None

    def get_action(self, states, actions, rewards, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [
                    torch.zeros(
                        (1, self.max_length - states.shape[1], self.state_dim),
                        dtype=torch.float32,
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            )
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0, -1]
