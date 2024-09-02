import argparse
import random
from typing import Dict
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict
from torch.nn import functional as F
from torch.types import Number
from training.trainer import Trainer

from hal.training.config import TrainConfig
from hal.training.config import create_parser_for_attrs_class
from hal.training.config import parse_args_to_attrs_instance
from hal.training.dataloader import create_dataloaders
from hal.training.dataloader import create_tensordicts
from hal.training.distributed import auto_distribute
from hal.training.distributed import wrap_multiprocessing


class RecurrentTrainer(Trainer):
    """
    Trainer for deep models with recurrent blocks.
    """

    def loss_fn(self, pred: TensorDict, target: TensorDict) -> TensorDict:
        loss_dict: TensorDict = TensorDict({})

        # Calculate per-frame losses
        button_loss = F.cross_entropy(pred["buttons"], target["buttons"], reduction="none")
        main_stick_loss = F.cross_entropy(pred["main_stick"], target["main_stick"], reduction="none")
        c_stick_loss = F.cross_entropy(pred["c_stick"], target["c_stick"], reduction="none")

        # Log per-frame losses
        for t in range(button_loss.shape[1]):
            loss_dict[f"_button_loss_frame_{t}"] = button_loss[:, t].mean()
            loss_dict[f"_main_stick_loss_frame_{t}"] = main_stick_loss[:, t].mean()
            loss_dict[f"_c_stick_loss_frame_{t}"] = c_stick_loss[:, t].mean()

        # Calculate mean losses across all frames
        mean_button_loss = button_loss.mean()
        mean_main_stick_loss = main_stick_loss.mean()
        mean_c_stick_loss = c_stick_loss.mean()

        loss_dict["loss_button"] = mean_button_loss
        loss_dict["loss_main_stick"] = mean_main_stick_loss
        loss_dict["loss_c_stick"] = mean_c_stick_loss

        return loss_dict

    def train_op(self, inputs: TensorDict, targets: TensorDict) -> Dict[str, Number]:
        self.opt.zero_grad(set_to_none=True)

        # Warmup trajectory without calculating loss
        warmup_len = self.config.data.input_len
        target_len = self.config.data.target_len
        warmup_inputs = inputs[:, :warmup_len]
        _, hidden = self.model(warmup_inputs)

        # Teacher forcing
        preds = []
        for i in range(target_len):
            pred, hidden = self.model(inputs[:, warmup_len + i : warmup_len + i + 1], hidden)
            preds.append(pred)

        preds_td: TensorDict = torch.stack(preds)  # type: ignore
        targets_td = targets[:, warmup_len : warmup_len + target_len]

        loss_by_head = self.loss_fn(preds_td, targets_td)
        loss_total = torch.tensor(sum(v for k, v in loss_by_head.items() if k.startswith("loss")))
        loss_total.backward()
        self.opt.step()
        self.scheduler.step()

        loss_by_head["loss_total"] = loss_total
        metrics_dict = {f"train/{k}": v.item() for k, v in loss_by_head.items()}
        metrics_dict["lr"] = self.scheduler.get_lr()
        return metrics_dict

    def val_op(self, inputs: TensorDict, targets: TensorDict) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            # Warmup trajectory without calculating loss
            warmup_len = self.config.data.input_len
            target_len = self.config.data.target_len
            warmup_inputs = inputs[:, :warmup_len]
            _, hidden = self.model(warmup_inputs)

            # Teacher forcing
            preds = []
            for i in range(self.config.data.target_len):
                pred, hidden = self.model(inputs[:, warmup_len + i : warmup_len + i + 1], hidden)
                preds.append(pred)

            preds_td: TensorDict = torch.stack(preds)  # type: ignore
            targets_td = targets[:, warmup_len : warmup_len + target_len]

            loss_by_head = self.loss_fn(preds_td, targets_td)
            loss_total = torch.tensor(sum(v for k, v in loss_by_head.items() if k.startswith("loss")))

        loss_by_head["loss_total"] = loss_total
        metrics_dict = {f"val/{k}": v.item() for k, v in loss_by_head.items()}
        return metrics_dict


@auto_distribute
def main(
    rank: Optional[int],
    world_size: Optional[int],
    train_td: TensorDict,
    val_td: TensorDict,
    train_config: TrainConfig,
) -> None:
    rank = rank or 0
    seed = train_config.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader = create_dataloaders(train_td, val_td, train_config, rank=rank, world_size=world_size)
    trainer = RecurrentTrainer(config=train_config, train_loader=train_loader, val_loader=val_loader)
    trainer.train_loop(
        train_loader,
        val_loader,
        local_batch_size=train_config.local_batch_size,
        n_samples=train_config.n_samples,
        n_val_samples=train_config.n_val_samples,
        report_len=train_config.report_len,
        keep_ckpts=train_config.keep_ckpts,
    )


def parse_cli() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser = create_parser_for_attrs_class(TrainConfig, parser)
    args = parser.parse_args()
    return parse_args_to_attrs_instance(TrainConfig, args)


if __name__ == "__main__":
    config = parse_cli()
    train_data, val_data = create_tensordicts(config.data)
    # pass positional args and call wrapped fn; (kwargs not accepted)
    wrap_multiprocessing(main, config, train_data, val_data)
