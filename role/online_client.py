import torch
from util import data_process, data_util, train_utils
import numpy as np
import model.model as mod
import model.layer as layer
from torch.utils.data import DataLoader
from util.device import device, to_device
from typing import List
from abc import abstractmethod
import role.client


class OnlineClient(role.client.Client):
    def __init__(self, owner: data_process.DataOwner, user_idx, model: mod.ATTModel, batch_size):
        super(OnlineClient, self).__init__(owner, model, batch_size)
        self.user_idx = user_idx

    def generate(self, start, end, lap, cache_seconds, fresh):
        xs, ys = [], []
        action_seconds = {action: cache_seconds for action in self.owner.activity_tag.keys()}
        action_fresh = {action: 0. for action in self.owner.activity_tag.keys()}
        action_dict = self.owner.choose_mul(self.user_idx, action_seconds, lap=lap, return_dict=True)
        while True:
            while len(ys) < self.batch_size:
                time_len = np.random.choice(range(start, end), size=1)[0]
                samples = time_len*1000 // self.owner.time_window
                action = np.random.choice(list(self.owner.activity_tag.keys()), size=1)[0]
                action_fresh[action] += time_len
                if action_fresh[action] > fresh * cache_seconds:
                    action_dict[action] = self.owner.choose(self.user_idx, action, cache_seconds, lap=lap)
                    action_fresh[action] = 0.

                start_time = np.random.choice(range(len(action_dict[action][1])), size=1)[0]
                end_time = (start_time + samples) % len(action_dict[action][1])
                if start_time <= end_time:
                    x = action_dict[action][0][start_time: end_time]
                    y = action_dict[action][1][start_time: end_time]
                else:
                    x = action_dict[action][0][:end_time] + action_dict[action][0][start_time:]
                    y = action_dict[action][1][:end_time] + action_dict[action][1][start_time:]

                xs += x
                ys += y

            loader = data_util.MLoader(DataLoader(data_util.ATTDataset(xs, ys), batch_size=self.batch_size,
                                                  shuffle=False, drop_last=True), loop=False)
            xys = next(loader)
            while xys is not None:
                yield xys
                xys = next(loader)

            index = (len(ys) // self.batch_size) * self.batch_size
            xs = xs[index:]
            ys = ys[index:]

    @abstractmethod
    def compute_grad(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_init(self, *args, **kwargs):
        pass

class OnlineSemiClient(OnlineClient):
    def __init__(self, owner: data_process.DataOwner, user_idx, model: mod.ATTModel, batch_size, near_fn=None):
        super(OnlineSemiClient, self).__init__(owner, user_idx, model, batch_size)
        self.near_fn = near_fn
        if near_fn is None:
            self.near_fn = layer.ProbCrossEntropy(torch.nn.MSELoss())

    def train_init(self, start, end, lap, cache_seconds, fresh):
        self.generator = self.generate(start, end, lap, cache_seconds, fresh)
        self.train_score = train_utils.Metric()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_grad(self, metric=True):
        self.model = self.model.to(device)

        xs, _ys = next(self.generator)
        xs = to_device(xs)
        preds = self.model(xs)
        near_loss = self.near_fn(preds)

        if metric:
            self.train_score.reset()
            self.train_score.step(loss={'loss': near_loss})

        near_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1, norm_type=2)

        grad_dict = {}
        for name, param in self.model.named_parameters():
            grad_dict[name] = torch.tensor(param.grad.data.clone().detach().cpu().tolist())

        self.model.zero_grad()

        return grad_dict