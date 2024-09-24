import math
from typing import Optional

import torch
from torch.utils.data import DistributedSampler, Dataset


class AdaptrDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        start_step: int = 0,
        prev_global_batch_size: int = -1,
    ) -> None:
        """DistributedSampler that can resume sampling from a previous state.
        Additional args:
        """

        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.prev_global_batch_size = prev_global_batch_size
        self.start_step = start_step

    def __iter__(self):
        # When resuming, we need to skip the indices that have already been used.
        # However, once the next epoch begins, we need to clear the used indices.
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        total_size = self.total_size
        num_samples = self.num_samples
        if self.start_step != 0:
            # remove indices that have already been used
            used_indices = []
            for replica in range(self.num_replicas):
                replica_indices = indices[replica : self.total_size : self.num_replicas]
                used_indices.extend(
                    replica_indices[: self.start_step * self.prev_global_batch_size]
                )

            # Remove used indices. Due to drop_last, only remove the first occurence of used_index in indices.
            for used_index in used_indices:
                indices.remove(used_index)
            # Update distributed sampler
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            super().__init__(
                self.dataset,
                self.num_replicas,
                self.rank,
                self.shuffle,
                self.seed,
                self.drop_last,
            )

            # Update num_samples
            if self.drop_last and len(indices) % self.num_replicas != 0:  # type: ignore[arg-type]
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                num_samples = math.ceil(
                    (len(indices) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
                )
            else:
                num_samples = math.ceil(len(indices) / self.num_replicas)  # type: ignore[arg-type]

            # Update total_size
            total_size = num_samples * self.num_replicas

            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[
                        :padding_size
                    ]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[:total_size]
            assert len(indices) == total_size

        # subsample
        indices = indices[self.rank : total_size : self.num_replicas]
        assert len(indices) == num_samples

        self.start_step = 0
        return iter(indices)
