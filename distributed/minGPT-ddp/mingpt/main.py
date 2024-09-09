import os
import socket
import torch
from torch.utils.data import random_split
from torch.distributed import init_process_group, destroy_process_group
from model import GPT, GPTConfig, OptimizerConfig, create_optimizer
from trainer import Trainer, TrainerConfig
from char_dataset import CharDataset, DataConfig
from omegaconf import DictConfig
import hydra

import torch.multiprocessing as mp


def ddp_setup(global_rank: int, world_size: int):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_process_group(backend=backend, rank=global_rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(global_rank % 8)

def get_train_objs(gpt_cfg: GPTConfig, opt_cfg: OptimizerConfig, data_cfg: DataConfig):
    dataset = CharDataset(data_cfg)
    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size
    model = GPT(gpt_cfg)
    optimizer = create_optimizer(model, opt_cfg)
    
    return model, optimizer, train_set, test_set

def start_processes(
    host_rank: int,
    world_size: int,
    gpt_cfg: GPTConfig,
    opt_cfg: OptimizerConfig,
    data_cfg: DataConfig,
    trainer_cfg: TrainerConfig,
):
    processes = []
    for local_rank in range(8):
        global_rank = host_rank * 8 + local_rank
        p = mp.Process(target=train_process, args=(global_rank, world_size, gpt_cfg, opt_cfg, data_cfg, trainer_cfg))
        p.start()
        processes.append(p)
    return processes

def train_process(
    global_rank: int,
    world_size: int,
    gpt_cfg: GPTConfig,
    opt_cfg: OptimizerConfig,
    data_cfg: DataConfig,
    trainer_cfg: TrainerConfig,
):
    ddp_setup(global_rank, world_size)

    model, optimizer, train_data, test_data = get_train_objs(gpt_cfg, opt_cfg, data_cfg)
    trainer = Trainer(global_rank, trainer_cfg, model, optimizer, train_data, test_data)
    trainer.train()

    destroy_process_group()

@hydra.main(version_base=None, config_path=".", config_name="gpt2_train_cfg")
def main(cfg: DictConfig):

    gpt_cfg = GPTConfig(**cfg['gpt_config'])
    opt_cfg = OptimizerConfig(**cfg['optimizer_config'])
    data_cfg = DataConfig(**cfg['data_config'])
    trainer_cfg = TrainerConfig(**cfg['trainer_config'])

    world_size = int(os.environ["WORLD_SIZE"])
    host_rank = int(os.environ["HOST_RANK"])
    
    if os.environ.get("MASTER_ADDR") is None:
        os.environ["MASTER_ADDR"] = "localhost"
    if os.environ.get("MASTER_PORT") is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        os.environ["MASTER_PORT"] = str(port)

    processes = start_processes(
        host_rank=host_rank,
        world_size=world_size,
        gpt_cfg=gpt_cfg,
        opt_cfg=opt_cfg,
        data_cfg=data_cfg,
        trainer_cfg=trainer_cfg,
    )

if __name__ == "__main__":
    main()
