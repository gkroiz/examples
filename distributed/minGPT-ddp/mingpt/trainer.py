"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from dataclasses import dataclass
from typing import Optional
import os
import io
import glob
import time
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.checkpoint as dcp
from torch.profiler import profile, record_function
from state_management import AppState, CustomWriter

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

class Trainer:

    def __init__(
        self,
        world_size: int,
        global_rank: int,
        trainer_config: TrainerConfig,
        model: torch.nn.Module,
        optimizer,
        train_dataset,
        test_dataset=None,
        profiler=None,
    ):
        self.config = trainer_config
        # set torchrun variables
        self.global_rank = global_rank
        self.local_rank = self.global_rank % 8
        self.world_size = world_size
        # data stuff
        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset else None
        self.model = model
        self.optimizer = optimizer        
        if self.config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.checkpoint_future = None
        self.profiler = profiler

    def _deferred_init(self):
        if torch.cuda.is_available():
            self.model = self.model.to(self.local_rank)
        device_ids = [self.local_rank] if torch.cuda.is_available() else None
        self.model = DDP(module=self.model, device_ids=device_ids)
        self.app_state = AppState(self.model, self.optimizer)
        self.writer = CustomWriter(path=CHECKPOINT_DIR, cache_staged_state_dict=True)
        self._load_checkpoint()

    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            drop_last=True,
            sampler=DistributedSampler(dataset=dataset, num_replicas=self.world_size, rank=self.global_rank)
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
        self._deferred_init()
        for epoch in range(int(self.app_state.epoch), self.config.max_epochs):
            self.app_state.epoch = epoch
            self._run_epoch(epoch, self.train_loader, train=True)
            # eval run
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)
