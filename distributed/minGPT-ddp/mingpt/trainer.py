"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from dataclasses import dataclass
from typing import Optional
import os
import glob
import time
from contextlib import nullcontext
from datetime import datetime
import pickle
import psutil

import torch
from torch.utils.data import Dataset, DataLoader
from sampler import AdaptrDistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.checkpoint as dcp
from torch.profiler import record_function
from state_management import AppState, CustomWriter, offload_state_dict_to_cpu
from concurrent.futures import ThreadPoolExecutor

CHECKPOINT_DIR = "checkpoint/"


@dataclass
class TrainerConfig:
    max_epochs: int = None
    micro_batch_size: int = None
    global_batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None
    save_every: int = None
    use_amp: bool = None
    profile: bool = None
    ckpt_method: bool = None
    debug: bool = None
    training_strategy: str = None
    num_data_replicas: int = None
    data_replica_size: int = None


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
        redis_client=None,
        host_name: str = None,
    ):
        self.config = trainer_config
        self.host_name = host_name
        # set torchrun variables
        self.global_rank = global_rank
        self.local_rank = self.global_rank % 8
        self.world_size = world_size
        # data stuff
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        if self.config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.checkpoint_future = None
        self.profiler = profiler
        self.redis_client = redis_client
        self.app_state = AppState(
            model=model,
            optimizer=optimizer,
            world_size=world_size,
            global_rank=global_rank,
            micro_batch_size=self.config.micro_batch_size,
            global_batch_size=self.config.global_batch_size,
        )
        self.epoch = 0
        self.step = 0
        self.gradient_accumulation_steps = (
            self.config.global_batch_size // self.config.micro_batch_size
        )
        assert self.gradient_accumulation_steps > 0

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
        start = time.time()
        if self.model is None:
            self.model = model
        if self.optimizer is None:
            self.optimizer = optimizer

        if torch.cuda.is_available():
            self.model = self.model.to(self.local_rank)
        if self.config.training_strategy == "ddp":
            self.model = DDP(module=self.model)
        elif self.config.training_strategy == "fsdp":
            self.model = FSDP(self.model)
        elif self.config.training_strategy == "hsdp":
            from torch.distributed.device_mesh import init_device_mesh

            assert (
                self.config.num_data_replicas is not None
            ), "num_model_replicas must be set for HSDP"
            assert (
                self.config.data_replica_size is not None
            ), "num_data_replicas must be set for HSDP"
            device_mesh = init_device_mesh(
                "cuda" if torch.cuda.is_available() else "cpu",
                (self.config.num_data_replicas, self.config.data_replica_size),
            )
            self.model = FSDP(self.model, device_mesh=device_mesh)
        else:
            raise ValueError("Invalid training strategy")
        self.writer = CustomWriter(path=CHECKPOINT_DIR, cache_staged_state_dict=True)
        self._load_checkpoint()
        # Update epoch and step counts
        self.epoch = self.app_state.epoch
        self.step = self.app_state.step
        # Create the dataloaders. This requires information stored in checkpoint metadata
        self.train_loader = self._prepare_dataloader(
            self.train_dataset,
            self.app_state.step,
            self.app_state.prev_world_size,
            self.app_state.prev_global_rank,
            self.app_state.prev_global_batch_size,
        )
        self.test_loader = (
            self._prepare_dataloader(self.test_dataset, 0)
            if self.test_dataset
            else None
        )
        # Once dataloaders are created, we can update app_state to reflect
        if self.global_rank == 0:
            print(f"Deferred init took {time.time() - start:.2f}s")

    def _prepare_dataloader(
        self,
        dataset: Dataset,
        start_step=0,
        prev_world_size=None,
        prev_global_rank=None,
        prev_global_batch_size=None,
    ):
        return DataLoader(
            dataset,
            batch_size=self.config.micro_batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            drop_last=False,
            sampler=AdaptrDistributedSampler(
                dataset=dataset,
                num_replicas=prev_world_size
                if (prev_world_size is not None)
                else self.world_size,
                rank=prev_global_rank
                if (prev_global_rank is not None)
                else self.global_rank,
                drop_last=False,
                start_step=start_step,
                prev_global_batch_size=prev_global_batch_size,
            ),
        )

    def _redis_save(self, state_dict, epoch, step):
        if self.host_name is not None:
            checkpoint_key = f"ckpt_host_{self.host_name}_localrank_{self.local_rank}"
        else:
            checkpoint_key = f"ckpt_localrank_{self.local_rank}"
        if self.redis_client.exists(checkpoint_key):
            self.redis_client.rename(checkpoint_key, checkpoint_key + "_backup")

        # Serialize and save the state_dict
        serialized_dict = pickle.dumps(state_dict)
        response = self.redis_client.set(checkpoint_key, serialized_dict)

        if not response:
            raise Exception("Failed to set checkpoint in Redis.")

        # Wait for all ranks to finish saving the checkpoint before proceeding
        torch.distributed.barrier()

        if self.redis_client.exists(checkpoint_key + "_backup"):
            self.redis_client.delete(checkpoint_key + "_backup")

        # Cleanup CPU memory
        del state_dict
        del serialized_dict
        return True

    def _save_checkpoint(self, epoch: int, step: int):
        # Update app_state with the current epoch and step
        self.app_state.epoch = epoch
        self.app_state.step = step
        if self.global_rank == 0:
            curr_time = datetime.fromtimestamp(time.time()).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            print(f"Starting async checkpoint at {curr_time}")
        if self.config.ckpt_method is None:
            return
        elif self.config.ckpt_method == "torch":
            self.checkpoint_future = dcp.async_save(
                state_dict={"app": self.app_state},
                checkpoint_id=f"{CHECKPOINT_DIR}ckpt_epoch{epoch}_step{step}",
                storage_writer=self.writer,
            )
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
            self.checkpoint_future.add_done_callback(
                lambda f: executor.shutdown(wait=False)
            )
        else:
            raise ValueError("Invalid checkpoint method")

    def _redis_load(self):
        if self.host_name is not None:
            checkpoint_key = f"ckpt_host_{self.host_name}_localrank_{self.local_rank}"
        else:
            checkpoint_key = f"ckpt_localrank_{self.local_rank}"
        use_backup = torch.tensor(0)
        if self.redis_client.exists(checkpoint_key + "_backup"):
            print(f"Found checkpoint from redis {checkpoint_key}_backup")
            use_backup += 1
        elif self.redis_client.exists(checkpoint_key):
            print(f"Found checkpoint from redis {checkpoint_key}")
        else:
            print("Checkpoint not found. Training model from scratch")
            return

        # Check if any rank needs to use the backup
        torch.distributed.all_reduce(use_backup, op=torch.distributed.ReduceOp.SUM)
        if use_backup.item() > 0:
            checkpoint_key += "_backup"

            assert (
                use_backup.item() == self.world_size
            ), "Backup checkpoint exists on some ranks but not all"

        print("Checkpointing key: ", checkpoint_key)

        # Load and unserialize the state_dict
        serialized_dict = self.redis_client.get(checkpoint_key)
        state_dict = pickle.loads(serialized_dict)
        self.app_state.load_state_dict(state_dict)

        # Update model and optimizer with the loaded state_dict
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optim"])
        print(
            f"Resuming training from checkpoint at epoch {self.app_state.epoch} step {self.app_state.step}"
        )

    def _load_checkpoint(self):
        # Find all files matching the pattern
        if self.config.ckpt_method is None:
            print("Checkpointing is disabled. Training model from scratch")
            return
        elif self.config.ckpt_method == "torch":
            checkpoint_files = glob.glob(f"{CHECKPOINT_DIR}ckpt_epoch*_step*")
            checkpoint_files = [
                file
                for file in checkpoint_files
                if os.path.exists(os.path.join(file, ".metadata"))
            ]

            # Sort the files based on the step number
            checkpoint_files.sort(
                key=lambda x: (
                    int(x.split("_epoch")[1].split("_step")[0]),
                    int(x.split("_step")[1]),
                )
            )

            # Get the file with the largest step number
            latest_checkpoint = checkpoint_files[-1] if checkpoint_files else None

            if latest_checkpoint:
                print(f"Loading checkpoint {latest_checkpoint}")

                dcp.load({"app": self.app_state}, checkpoint_id=latest_checkpoint)

            else:
                print("Checkpoint not found. Training model from scratch")
                return

            print(
                f"Resuming training from checkpoint at epoch {self.app_state.epoch} step {self.app_state.step}"
            )
        elif self.config.ckpt_method == "redis":
            self._redis_load()
        else:
            raise ValueError("Invalid checkpoint method")

    def _run_batch(
        self, source, targets, is_accumulating: bool, train: bool = True
    ) -> float:
        #
        with torch.set_grad_enabled(train), torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=(self.config.use_amp)
        ):
            # According to https://huggingface.co/docs/accelerate/en/concept_guides/gradient_synchronization,
            # Forward step with FSDP and gradient accumulation should have no_sync() context manager
            with self.model.no_sync() if self.config.training_strategy == "fsdp" else nullcontext():
                with record_function("FWD"):
                    _, loss = self.model(source, targets)

        if train:
            if self.config.use_amp:
                with record_function("BWD"):
                    self.scaler.scale(loss).backward()
                with record_function("CLIP"):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_norm_clip
                    )
                with record_function("OPT"):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            else:
                # Divide loss by number of gradient accumulation steps
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                with self.model.no_sync() if self.config.training_strategy == "fsdp" else nullcontext():
                    with record_function("BWD"):
                        loss.backward()

                if not is_accumulating:
                    with record_function("CLIP"):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_norm_clip
                        )
                    with record_function("OPT"):
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        dataloader.sampler.set_epoch(epoch)
        start = time.time()
        step = self.step if train else 0
        step_loss = 0
        # Iterate through each batch in the dataloader
        for dataloader_iter, (source, targets) in enumerate(dataloader):
            # Boolean determining whether all gradients have accumulated for this step
            is_accumulating = (
                dataloader_iter + 1
            ) % self.gradient_accumulation_steps != 0

            if self.config.debug and self.global_rank == 0:
                step_type = "Train" if train else "Eval"
                print(
                    f"Epoch {epoch} | {step_type} Step {step} | Iter {dataloader_iter} CPU RAM Used (GB): {psutil.virtual_memory()[3]/1000000000}"
                )

            step_type = "Train" if train else "Eval"
            if torch.cuda.is_available():
                source = source.to(self.local_rank)
                targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source, targets, is_accumulating, train)
            step_loss += batch_loss

            if self.profiler:
                self.profiler.step()

            if step % 10 == 0 and not is_accumulating:
                if self.global_rank == 0:
                    print(
                        f"[GPU{self.global_rank}] Epoch {epoch} | Step {step} | {step_type} Loss {step_loss:.5f} | Average Step Time {(time.time() - start)/100:.2f}s"
                    )
                start = time.time()

            # Check if you can checkpoint, increment step counters, and reset step_loss
            if not is_accumulating:
                # Only checkpoint when training
                if train:
                    if (
                        self.checkpoint_future is None
                        or self.checkpoint_future.result()
                    ):
                        self._save_checkpoint(epoch, step)

                    # Update trainer step counter
                    self.step += 1

                # Update local step counter
                step += 1
                step_loss = 0

    def train(self, model=None, optimizer=None):
        self._deferred_init(model, optimizer)
        for epoch in range(int(self.epoch), self.config.max_epochs):
            self.epoch = epoch
            self._run_epoch(epoch, self.train_loader, train=True)
            # eval run
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)
