from dataclasses import replace
from torch.utils.data.sampler import Sampler, BatchSampler
import torch
import numpy as np


class QueryBasedSampler(Sampler):
    def __init__(self, qid2idx_map, shuffle=True):
        self.qid2idx_map = qid2idx_map
        self.qids = list(qid2idx_map.keys())
        self.shuffle = shuffle

    @property
    def num_samples(self):
        return len(self.qid2idx_map)

    def __iter__(self):
        if self.shuffle:
            yield from torch.randperm(self.num_samples).tolist()
        else:
            yield from iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples


class QueryBasedBatchSampler(BatchSampler):

    def __init__(self, sampler, batch_size, max_len=32):
        self.sampler = sampler
        self.batch_size = batch_size
        self.max_len = max_len 

    def __iter__(self):
        for idx in self.sampler:
            qid = self.sampler.qids[idx]
            batch = self.sampler.qid2idx_map[qid]
            batch = torch.LongTensor(batch)
            batchs = torch.split(batch, self.max_len)
            for b in batchs:
                yield b.numpy().tolist()


    def __len__(self):
        return len(self.sampler)
