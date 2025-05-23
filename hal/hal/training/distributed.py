import builtins
import functools
import os
import time
import traceback
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import torch
import torch.distributed
import tqdm as tqdm_module
from loguru import logger
from tqdm import tqdm

from hal.training.config import BaseConfig

# Disable Infiniband and P2P to ensure that NCCL does not try to use within single-node distributed training
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
# # Try adding explicit timeout configuration
# os.environ["NCCL_BLOCKING_WAIT"] = "1"
# os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
# os.environ["TORCH_DISTRIBUTED_DETAIL"] = "DEBUG"  # For more verbose logs


def barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{get_device_id()}"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def get_device_id() -> int:
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def is_master() -> bool:
    return get_device_id() == 0


def log_if_master(message: Any) -> None:
    if is_master():
        logger.info(message)


def trange(*args, **kwargs) -> Union[range, tqdm]:
    if not is_master():
        return range(*args)
    return tqdm_module.trange(*args, **kwargs)


def cuda_setup() -> None:
    torch.backends.cudnn.benchmark = True
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass


def print(*args, **kwargs) -> None:
    if is_master():
        builtins.print(*args, **kwargs)


def get_world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def maybe_wrap_model_distributed(
    m: torch.nn.Module,
) -> Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel]:
    if not torch.distributed.is_initialized():
        return m.to(get_device())
    return torch.nn.parallel.DistributedDataParallel(m.to(get_device_id()), device_ids=[get_device_id()])


def auto_distribute(f: Callable) -> Callable:
    """Automatically initialize process group for training loop."""

    @functools.wraps(f)
    def dist_wrapped(rank: Optional[int], world_size: Optional[int], *args):
        if rank is None or world_size is None:
            return f(*args)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        time.sleep(1)
        try:
            return f(*args)
        except Exception as e:
            logger.error(f"Error in distributed training: {e}\n{traceback.format_exc()}")
        finally:
            logger.info("destroy_process_group")
            torch.distributed.destroy_process_group()

    return dist_wrapped


def wrap_multiprocessing(main_fn: Callable, config: BaseConfig, *args: Any) -> Callable:
    """
    Initialize torch multiprocessing for training.
    """

    @functools.wraps(main_fn)
    def multiprocessing_wrapped():
        cuda_setup()
        device_count = torch.cuda.device_count()
        n_gpus = config.n_gpus
        assert n_gpus <= device_count, f"n_gpus={n_gpus}, only {device_count} gpus available!"
        if n_gpus == 1:
            return main_fn(None, None, config, *args)
        torch.multiprocessing.spawn(
            main_fn, args=(n_gpus, config, *args), nprocs=n_gpus, join=False, start_method="spawn"
        )

    @functools.wraps(main_fn)
    def dummy_wrapped():
        return main_fn(None, None, config, *args)

    device = get_device()
    if device.startswith("cuda"):
        return multiprocessing_wrapped
    else:
        return dummy_wrapped
