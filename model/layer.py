import torch
import torch.nn as nn
from torch.nn import Module

class Attention(Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.weight = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh(), nn.Softmax(dim=1))
        self.record = False
        self.record_value = None

    def forward(self, inputs :torch.Tensor):
        assert len(inputs.shape) == 3
        weight = self.weight(inputs)
        outputs = (inputs * weight).sum(dim=1)
        if self.record:
            if self.record_value is None:
                self.record_value = torch.tensor(weight.mean(dim=[0,2]).clone().detach().cpu().tolist())
            else:
                self.record_value = (torch.tensor(weight.mean(dim=[0,2]).clone().detach().cpu().tolist()) + self.record_value) / 2
        return outputs

    def record_on(self):
        self.record_value = None
        self.record = True

    def record_off(self):
        self.record_value = None
        self.record = False

class MultiClassCrossEntropy:
    def __init__(self, T :int):
        self.T = T

    def __call__(self, logits :torch.Tensor, labels :torch.Tensor):
        outputs = torch.log_softmax(logits/self.T, dim=1)
        labels = torch.softmax(labels/self.T, dim=1)
        outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
        outputs = -torch.mean(outputs, dim=0, keepdim=False)
        return outputs

class ProbCrossEntropy:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, logits :torch.Tensor):
        return self.loss_fn(logits[1:], logits[:-1])


