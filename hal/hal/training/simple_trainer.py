import argparse
import random

import numpy as np
import torch
from tensordict import TensorDict
from torch.nn import functional as F

from hal.training.config import TrainConfig
from hal.training.config import create_parser_for_attrs_class
from hal.training.config import parse_args_to_attrs_instance
from hal.training.distributed import auto_distribute
from hal.training.distributed import get_device_id
from hal.training.distributed import wrap_multiprocessing
from hal.training.streaming_dataloader import get_dataloaders
from hal.training.trainer import Trainer


class SimpleTrainer(Trainer):
    """
    Trains behavior cloning using cross-entropy loss on next-token prediction.
    """

    def loss(self, pred: TensorDict, target: TensorDict) -> TensorDict:
        loss_dict: TensorDict = TensorDict({})
        loss_fns = {
            "buttons": F.cross_entropy,
            "main_stick": F.cross_entropy,
            "c_stick": F.cross_entropy,
            "shoulder": F.cross_entropy,
            "analog_shoulder": F.cross_entropy,
            "shoulder_l": F.binary_cross_entropy_with_logits,
            "shoulder_r": F.binary_cross_entropy_with_logits,
        }

        for target_feature, loss_fn in loss_fns.items():
            if target_feature in pred and target_feature in target:
                frame_losses = loss_fn(pred[target_feature], target[target_feature])
                loss_dict[f"loss_{target_feature}"] = frame_losses

        return loss_dict

    def sum_losses(self, loss_by_head: TensorDict) -> torch.Tensor:
        return sum(v for k, v in loss_by_head.items() if k.startswith("loss"))


@auto_distribute
def main(train_config: TrainConfig) -> None:
    rank = get_device_id()
    seed = train_config.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader = get_dataloaders(train_config)
    trainer = SimpleTrainer(config=train_config, train_loader=train_loader, val_loader=val_loader)
    trainer.train_loop(train_loader, val_loader)


def parse_cli() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser = create_parser_for_attrs_class(TrainConfig, parser)
    args = parser.parse_args()
    return parse_args_to_attrs_instance(TrainConfig, args)


if __name__ == "__main__":
    config = parse_cli()
    # pass positional args and call wrapped fn; (kwargs not accepted)
    wrapped_train = wrap_multiprocessing(main, config)
    wrapped_train()
