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
import pickle
import psutil
import gc

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.checkpoint as dcp
from torch.profiler import profile, record_function
from state_management import AppState, CustomWriter, offload_state_dict_to_cpu
from concurrent.futures import Future, ThreadPoolExecutor

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
    ckpt_method: bool = None
    debug: bool = None

class Trainer:

    def __init__(
        self,
        world_size: int,
        global_rank: int,
        trainer_config: TrainerConfig,
        train_dataset,
        test_dataset=None,
        model=None,
        optimizer=None,
        profiler=None,
        redis_client=None
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
        if self.config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.checkpoint_future = None
        self.profiler = profiler
        self.redis_client = redis_client
        self.app_state = AppState(model, optimizer)

    @property
    def model(self):
        return self.app_state.model
    
    @model.setter
    def model(self, model):
        self.app_state.model = model
    
    @property
    def optimizer(self):
        return self.app_state.optimizer
        
    @optimizer.setter
    def optimizer(self, optimizer):
        self.app_state.optimizer = optimizer

    def _deferred_init(self, model=None, optimizer=None):
        if self.model is None:
            self.model = model
        if self.optimizer is None:
            self.optimizer = optimizer

        if torch.cuda.is_available():
            self.model = self.model.to(self.local_rank)
        device_ids = [self.local_rank] if torch.cuda.is_available() else None
        self.model = DDP(module=self.model, device_ids=device_ids)
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

    def _redis_save(self, state_dict, epoch, step):
        checkpoint_key = f"ckpt_rank_{self.global_rank}"
        if self.redis_client.exists(checkpoint_key):
            self.redis_client.rename(checkpoint_key, checkpoint_key + "_backup")

        # Serialize and save the state_dict
        serialized_dict = pickle.dumps(state_dict)
        response = self.redis_client.set(checkpoint_key, serialized_dict)

        if not response:
            raise Exception("Failed to set checkpoint in Redis.")
        if self.redis_client.exists(checkpoint_key + "_backup"):
            self.redis_client.delete(checkpoint_key + "_backup")

        # Cleanup CPU memory
        del state_dict
        del serialized_dict
        return True

    def _save_checkpoint(self, epoch: int, step: int):
        if self.global_rank == 0:
            curr_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Starting async checkpoint at {curr_time}")
        if self.config.ckpt_method is None:
            return
        elif self.config.ckpt_method == "torch":
            self.checkpoint_future = dcp.async_save(
                state_dict={"app": self.app_state},
                checkpoint_id=f"{CHECKPOINT_DIR}ckpt_epoch{epoch}_step{step}",
                storage_writer=self.writer)
        elif self.config.ckpt_method == "redis":
            # Offload the state_dict to CPU memory before serializing & saving
            state_dict = offload_state_dict_to_cpu(self.app_state.state_dict())
            executor = ThreadPoolExecutor(max_workers=1)
            self.checkpoint_future = executor.submit(
                self._redis_save,
                state_dict,
                epoch,
                step,
            )
            del state_dict
            self.checkpoint_future.add_done_callback(lambda f: executor.shutdown(wait=False))
        else:
            raise ValueError("Invalid checkpoint method")

    def _redis_load(self):
        checkpoint_key = f"ckpt_rank_{self.global_rank}"
        if self.redis_client.exists(checkpoint_key):
            print(f"Loading checkpoint from redis {checkpoint_key}")
        elif self.redis_client.exists(checkpoint_key + "_backup"):
            print(f"Loading checkpoint from redis {checkpoint_key}_backup")
            checkpoint_key += "_backup"
        else:
            print("Checkpoint not found. Training model from scratch")
            return

        # Load and unserialize the state_dict
        serialized_dict = self.redis_client.get(checkpoint_key)
        state_dict = pickle.loads(serialized_dict)
        self.app_state.load_state_dict(state_dict)
        print(f"Resuming training from checkpoint at epoch {self.app_state.epoch} step {self.app_state.step}")

    def _load_checkpoint(self):
        # Find all files matching the pattern
        if self.config.ckpt_method is None:
            print("Checkpointing is disabled. Training model from scratch")
            return
        elif self.config.ckpt_method == "torch":
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
        elif self.config.ckpt_method == "redis":
            self._redis_load()
        else:
            raise ValueError("Invalid checkpoint method")

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
            if self.config.debug and self.global_rank == 0:
                print(f"Epoch {epoch} | Step {step} CPU RAM Used (GB): {psutil.virtual_memory()[3]/1000000000}")
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

            if self.checkpoint_future is None or self.checkpoint_future.result():
                self._save_checkpoint(epoch, step)
            else:
                print("self.checkpoint_future.result(): ", self.checkpoint_future.result())

    def train(self, model=None, optimizer=None):
        self._deferred_init(model, optimizer)
        for epoch in range(int(self.app_state.epoch), self.config.max_epochs):
            self.app_state.epoch = epoch
            self._run_epoch(epoch, self.train_loader, train=True)
            # eval run
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)
