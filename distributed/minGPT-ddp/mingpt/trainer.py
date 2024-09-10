"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from dataclasses import dataclass, asdict
from collections import OrderedDict
from typing import Optional, Any, Dict
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.profiler import profile, record_function

from urllib.parse import urlparse
import fsspec
import io
import glob
import json
import time
from datetime import datetime

CHECKPOINT_DIR = "checkpoint/"

@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None
    save_every: int = None
    use_amp: bool = None
    profile: bool = None
    async_ckpts: bool = None

class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None, epoch=0, step=0):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.step = step

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "metadata": self.metadata()
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )
        self.load_metadata(state_dict["metadata"])
    
    def metadata(self):
        return {"epoch": self.epoch, "step": self.step}
    
    def load_metadata(self, metadata):
        self.epoch = metadata["epoch"]
        self.step = metadata["step"]

class Trainer:

    def __init__(self, global_rank: int, trainer_config: TrainerConfig, model, optimizer, train_dataset, test_dataset=None, profiler=None):
        self.config = trainer_config
        # set torchrun variables
        self.global_rank = global_rank
        self.local_rank = self.global_rank % 8
        # data stuff
        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset else None
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.to(self.local_rank)
        self.optimizer = optimizer        
        if self.config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        # load snapshot if available. only necessary on the first node.
        self.app_state = AppState(self.model, self.optimizer)
        self.writer = FileSystemWriter(path=CHECKPOINT_DIR, cache_staged_state_dict=True)
        self._load_checkpoint()
        # initialize train states
        # wrap with DDP. this step will synch model across all the processes.
        device_ids = [self.local_rank] if torch.cuda.is_available() else None
        self.model = DDP(module=self.model, device_ids=device_ids)
        self.checkpoint_future = None
        self.profiler = profiler
        
    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            drop_last=True,
            sampler=DistributedSampler(dataset)
        )

    def _save_checkpoint(self, epoch: int, step: int):
        if self.global_rank == 0:
            curr_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Starting async checkpoint at {curr_time}")

        self.checkpoint_future = dcp.async_save(
            state_dict={"app": self.app_state},
            checkpoint_id=f"{CHECKPOINT_DIR}ckpt_epoch{epoch}_step{step}",
            storage_writer=self.writer)

    def _load_checkpoint(self):
        # Find all files matching the pattern
        checkpoint_files = glob.glob(f"{CHECKPOINT_DIR}ckpt_epoch*_step*")
        checkpoint_files = [file for file in checkpoint_files if os.path.exists(os.path.join(file, ".metadata"))]

        # Sort the files based on the step number
        checkpoint_files.sort(key=lambda x: (int(x.split('_epoch')[1].split('_step')[0]), int(x.split('_step')[1])))

        # Get the file with the largest step number
        latest_checkpoint = checkpoint_files[-1] if checkpoint_files else None

        if latest_checkpoint:
            print(f"Loading checkpoint {latest_checkpoint}")

            dcp.load({"app": self.app_state}, checkpoint_id=latest_checkpoint)

        else:
            print("Checkpoint not found. Training model from scratch")
            return 

        print(f"Resuming training from checkpoint at epoch {self.app_state.epoch} step {self.app_state.step}")


    def _run_batch(self, source, targets, train: bool = True) -> float:
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.config.use_amp)):
            with record_function("FWD"):
                _, loss = self.model(source, targets)
        
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.config.use_amp: 
                with record_function("BWD"):
                    self.scaler.scale(loss).backward()
                with record_function("CLIP"):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                with record_function("OPT"):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                with record_function("BWD"):
                    loss.backward()
                with record_function("CLIP"):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                with record_function("OPT"):
                    self.optimizer.step()
        
        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        dataloader.sampler.set_epoch(epoch)
        start = time.time()
        for step, (source, targets) in enumerate(dataloader):
            if step < self.app_state.step:
                continue
            self.app_state.step = step

            step_type = "Train" if train else "Eval"
            if torch.cuda.is_available():
                source = source.to(self.local_rank)
                targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source, targets, train)

            if self.profiler:
                self.profiler.step()

            if step % 10 == 0:
                if self.global_rank == 0:
                    print(f"[GPU{self.global_rank}] Epoch {epoch} | Step {step} | {step_type} Loss {batch_loss:.5f} | Average Step Time {(time.time() - start)/100:.2f}s")
                start = time.time()

            if self.config.async_ckpts:
                if self.checkpoint_future is None or self.checkpoint_future.result():
                    self._save_checkpoint(epoch, step)


    def train(self):
        for epoch in range(int(self.app_state.epoch), self.config.max_epochs):
            self.app_state.epoch = epoch
            self._run_epoch(epoch, self.train_loader, train=True)
            # eval run
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)

            