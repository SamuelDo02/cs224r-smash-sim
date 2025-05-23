import random
from typing import Optional
from typing import Sequence
from typing import cast

import numpy as np
from streaming import Stream
from streaming import StreamingDataset
from tensordict import TensorDict

from hal.constants import Player
from hal.constants import VALID_PLAYERS
from hal.preprocess.preprocessor import Preprocessor
from hal.preprocess.preprocessor import convert_ndarray_to_tensordict
from hal.preprocess.transformations import add_reward_to_episode
from hal.training.config import DataConfig


class HALStreamingDataset(StreamingDataset):
    def __init__(
        self,
        streams: Optional[Sequence[Stream]],
        local: Optional[str],
        remote: Optional[str],
        batch_size: int,
        shuffle: bool,
        data_config: DataConfig,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(streams=streams, local=local, remote=remote, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.preprocessor = Preprocessor(data_config=data_config)
        self.seq_len = self.preprocessor.seq_len
        self.data_config = data_config
        self.debug = debug

        if self.data_config.debug_repeat_batch:
            self.debug_sample = self._get_item(0)

    def _get_item(self, idx: int | slice | list[int] | np.ndarray) -> TensorDict:
        episode_features_by_name = super().__getitem__(idx)
        episode_features_by_name = add_reward_to_episode(episode_features_by_name)
        episode_td = convert_ndarray_to_tensordict(episode_features_by_name)

        player_perspective = "p1" if self.debug else cast(Player, random.choice(VALID_PLAYERS))
        episode_td = self.preprocessor.compute_returns(episode_td, player_perspective)
        sample_T = self.preprocessor.sample_from_episode(episode_td, debug=self.debug)

        inputs_T = self.preprocessor.preprocess_inputs(sample_T, player_perspective)
        targets_T = self.preprocessor.preprocess_targets(sample_T, player_perspective)

        inputs_L = self.preprocessor.offset_inputs(inputs_T)
        targets_L = self.preprocessor.offset_targets(targets_T)

        return TensorDict(
            {
                "inputs": inputs_L,
                "targets": targets_L,  # type: ignore
            },
            batch_size=(self.seq_len,),
        )

    def __getitem__(self, idx: int | slice | list[int] | np.ndarray) -> TensorDict:
        """Expects episode features to match data/schema.py."""
        if self.data_config.debug_repeat_batch:
            return self.debug_sample
        return self._get_item(idx)
