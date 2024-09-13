import os
from typing import Union
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._state_dict_utils import (
    _offload_state_dict_to_cpu,
)


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
        return {
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "metadata": self.metadata(),
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optim"])
        self.load_metadata(state_dict["metadata"])

    def metadata(self):
        return {"epoch": self.epoch, "step": self.step}

    def load_metadata(self, metadata):
        self.epoch = metadata["epoch"]
        self.step = metadata["step"]


class CustomWriter(FileSystemWriter):
    def __init__(
        self,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        cache_staged_state_dict: bool = False,
        overwrite: bool = True,
    ) -> None:
        super().__init__(
            path=path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
            cache_staged_state_dict=cache_staged_state_dict,
            overwrite=overwrite,
        )
        self.cache_staged_state_dict = cache_staged_state_dict

    def stage(self, state_dict):
        if not self.cache_staged_state_dict:
            return _offload_state_dict_to_cpu(state_dict, type_check=self.type_check)

        self.old_state_dict_cache = self.state_dict_cache
        self.state_dict_cache = _offload_state_dict_to_cpu(state_dict)
        self.old_state_dict_cache = None

        return self.state_dict_cache


def offload_state_dict_to_cpu(state_dict):
    return _offload_state_dict_to_cpu(state_dict)
