import torch
from util import data_process, data_util, train_utils
import numpy as np
import model.model as mod
import model.layer as layer
from torch.utils.data import DataLoader
from util.device import device, to_device
from typing import List
from abc import abstractmethod


class Client:
    def __init__(self, owner: data_process.DataOwner, model: mod.ATTModel, batch_size=256):
        self.owner = owner
        self.model = model
        self.batch_size = batch_size

    @abstractmethod
    def train_init(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_grad(self, *args, **kwargs) -> dict:
        pass


class Staff(Client):
    def __init__(self, owner: data_process.DataOwner, user_idx, samples, model: mod.ATTModel, batch_size):
        super(Staff, self).__init__(owner, model, batch_size)
        self.user_idx = user_idx
        self.samples = samples
        self.action_tup = self.owner.get_time_tup(user_idx, samples)

    def train_init(self, seconds=250, fresh=3, lap=0.):
        self.generator = self.generate(seconds, fresh, lap)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_score = train_utils.Metric()

    def generate(self, seconds, fresh, lap):
        while True:
            xs, ys = self.owner.choose_mul(self.user_idx, {action: seconds for action in self.action_tup.keys()},
                                           action_tup=self.action_tup, lap=lap)
            for idx in range(fresh):
                loader = data_util.MLoader(
                    DataLoader(data_util.ATTDataset(xs, ys), batch_size=self.batch_size, shuffle=True), loop=False)
                xys = next(loader)
                while xys is not None:
                    yield xys
                    xys = next(loader)

    def compute_grad(self, *args, **kwargs):
        self.model = self.model.to(device)
        self.model.train()
        xs, ys = next(self.generator)
        ys = to_device(ys)
        xs = to_device(xs)

        pred = self.model(xs)

        ce_loss = self.loss_fn(pred, ys)
        loss = ce_loss + self.model.regular()

        if kwargs['metric']:
            self.train_score.step(loss={'loss': loss})

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100, norm_type=2)

        grad_dict = {}
        for name, param in self.model.named_parameters():
            grad_dict[name] = torch.tensor(param.grad.data.clone().detach().cpu().tolist())

        self.model.zero_grad()
        return grad_dict
