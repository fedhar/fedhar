import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from typing import List
import numpy as np
from util import data_process


class ATTDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        nums = [self.ys.count(i) for i in range(max(self.ys)+1)]
        self.class_num =len(nums)
        weight = [1/nums[y] for y in ys]
        self.sampler = WeightedRandomSampler(weight, num_samples=len(ys), replacement=True)

    def __getitem__(self, index):
        return data_process.to_tensor(self.xs[index]), self.ys[index]

    def __len__(self):
        return len(self.ys)

    def num4class(self):
        return [self.ys.count(i) for i in range(max(self.ys)+1)]


class MLoader:
    def __init__(self, dataloader:DataLoader, loop=True):
        self.dataloader = dataloader
        self.it = iter(dataloader)
        self.loop = loop

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            if self.loop:
                self.it = iter(self.dataloader)
                return next(self.it)
            else:
                return None

    def __len__(self):
        return self.dataloader.__len__()

def dataset_by_class(xs, ys) -> List[ATTDataset]:
    result = []
    for y in set(ys):
        index = [idx for idx in range(len(ys)) if ys[idx] == y]
        result.append(ATTDataset(
            [xs[idx] for idx in index], [ys[idx] for idx in index]
        ))
    return result