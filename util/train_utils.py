from model.model import ATTModel
import torch
from torch.utils.data import DataLoader
from typing import List, Callable
from util.device import to_device
import numpy as np
import time
from copy import deepcopy
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


def acc_score(y_pred, y_true: torch.Tensor, average=True, return_list=False):
    if average:
        acc = {}
        for cl in y_true.unique():
            index = [idx for idx in range(len(y_true)) if y_true[idx] == cl]
            acc[int(cl)] = accuracy_score(y_pred=y_pred[index], y_true=y_true[index])
        if return_list:
            return torch.tensor(list(acc.values())).mean(), acc
        else:
            return torch.tensor(list(acc.values())).mean()
    else:
        return accuracy_score(y_pred=y_pred, y_true=y_true)


class Metric_Item:
    def __init__(self, func: Callable, name: str = None, **kwargs):
        name = name if name is not None else func.__name__
        self.tuple = (name, func, kwargs)

    def __getitem__(self, item):
        return self.tuple[item]

class Metric:
    def __init__(self, items :List[Metric_Item] = None):
        self.preds :List[torch.Tensor]= []
        self.labels :List[torch.Tensor]= []
        self.losses :List[dict]= []
        self.items = items if items is not None else []
        self.record = (len(self.items) != 0)


    def step(self, labels :torch.Tensor = None, preds :torch.Tensor = None, loss :[torch.Tensor, dict] = None):
        if self.record:
            labels = self.process_data(labels)
            preds = self.process_data(preds)
            self.labels.append(labels)
            self.preds.append(preds)

        if loss is not None:
            if type(loss) == dict:
                loss = {n: self.process_data(p) for n, p in loss.items()}
            else:
                loss ={'loss': self.process_data(loss)}
            self.losses.append(loss)

    def process_data(self, tensor):
        if tensor is None:
            pass
        elif type(tensor) == torch.Tensor:
            tensor = torch.tensor(tensor.clone().detach().cpu().tolist())
        elif type(tensor) == np.ndarray:
            tensor = torch.from_numpy(tensor)
        else:
            tensor = torch.tensor(tensor)

        return tensor


    def value(self):
        if len(self.losses) == 0 and len(self.labels) == 0:
            raise Exception('has already read.')

        result = {}

        if len(self.losses) >0:
            for key in self.losses[0].keys():
                result[key] = torch.tensor([dic[key] for dic in self.losses]).mean()

        if self.record:
            preds = torch.cat(self.preds, dim=0)
            labels = torch.cat(self.labels, dim=0)
            for name, func, kwargs in self.items:
                result[name] = func(y_pred=preds.reshape(preds.shape[0], -1), y_true=labels.reshape(labels.shape[0], -1), **kwargs)

        self.reset()
        return result

    def reset(self):
        self.preds.clear()
        self.labels.clear()
        self.losses.clear()


def eval(model : ATTModel, eval_loader: DataLoader, score: Metric) -> dict:
    model.eval()
    for xs, ys in eval_loader:
        ys = to_device(ys)
        xs = to_device(xs)

        pred = model(xs)

        ce_loss = torch.nn.functional.cross_entropy(pred, ys)
        loss = ce_loss + model.regular()

        score.step(loss={'eval_loss': loss}, labels=ys, preds=pred.max(dim=1)[1])

    model.train()
    return score.value()

def interpolate(xs: List[torch.Tensor], preds_t: torch.Tensor, k):
    sensor_xs = []
    for sensor in range(len(xs)):
        dev_xs = []
        for dev in range(len(xs[sensor])):
            x1 = xs[sensor][dev][:-1]
            x2 = xs[sensor][dev][1:]
            x = k * x1 + (1-k) * x2
            dev_xs.append(x)
        sensor_xs.append(dev_xs)

    inter_ys = (k * preds_t[:-1] + (1-k) * preds_t[1:])

    return sensor_xs, inter_ys

def show_result(*results, decimal=5, end='\n'):
    def show(p, decimal, func=None):
        if type(p) == torch.Tensor or type(p) == np.float64 or type(p) == float:
            if func is None:
                print(('{:.' + str(decimal) + 'f}').format(float(p)), end='')
            else:
                print(func(p, decimal), end='')
        elif type(p) == list or type(p) == tuple:
            print('(', end='')
            for pi in p:
                show(pi, decimal=decimal, func=func)
                print(',', end='')
            print(')', end='')
        elif type(p) == dict:
            if 'func' in p.keys():
                func = p.pop('func')
            print('{', end='')
            for key, value in p.items():
                show(key, decimal=decimal, func=func)
                print(':', end='')
                show(value, decimal=max(decimal-2, 2), func=func)
                print(',', end='')
            print('}', end='')
        else:
            print(p, end='')

    for result in results:
       show(result, decimal=decimal+2)
    print(end=end)

class Early_Stop:
    def __init__(self, model :nn.Module, patience :int, max_well :bool, min_change :float = 1e-5, return_best = True, verbose = True, allow_nan = False):
        self.model = model
        self.patience = patience
        self.max_well = max_well
        self.min_change = min_change
        self.return_best = return_best
        self.epoch = 0
        self.verbose = verbose
        self.allow_nan = allow_nan

        self.worst_value = (-1. if max_well else 1.) * float('inf')
        self.best_value = self.worst_value
        self.args = None
        self.best = None

    def __call__(self, value, args=None, can_record=True, stop=False) -> bool:
        if np.isnan(value):  # value is nan
            if self.allow_nan:
                value = self.worst_value
            else:
                self.stop(info='by counter nan')
                return True

        if not can_record and self.best is None:
            return False

        self.epoch = self.epoch + 1

        if self.better(value, self.best_value, delta=self.min_change) and can_record: #value is better than best
            self.record_best(value, args)
            return False

        if self.epoch < self.patience and stop == False:
            return False
        else:
            self.stop()
            return True

    def record_best(self, value, *args):
        self.best_value = value
        self.args = args
        if self.return_best:
            self.best = deepcopy(self.model.state_dict())
        self.epoch = 0

    def better(self, a, b, delta = 0.):
        if self.max_well:
            return a - b - delta > 0
        else:
            return a + delta - b < 0

    def stop(self, info = ''):
        if self.return_best and self.best != None:
            self.model.load_state_dict(self.best)
        if self.verbose:
            print('early stopping {} ...'.format(info))

class ValueStopper:
    def __init__(self, value, patience :int, max_well :bool, verbose=True):
        self.value = value
        self.patience = patience
        self.max_well = max_well
        self.epoch = 0
        self.verbose = verbose

    def __call__(self, value, inc=1) -> bool:
        self.epoch = self.epoch + inc

        if self.better(value, self.value): #value is better than best
            self.epoch = 0
            return False

        if self.epoch < self.patience:
            return False
        else:
            self.stop()
            return True

    def better(self, a, b, delta = 0.):
        if self.max_well:
            return a - b - delta > 0
        else:
            return a + delta - b < 0

    def stop(self, info = ''):
        if self.verbose:
            print('stopping {} ...'.format(info))

    def state_dict(self):
        return {
            'value': self.value, 'patience': self.patience, 'max_well': self.max_well, 'epoch': self.epoch,
        }

    def load_state_dict(self, state):
        self.value = state['value']
        self.patience = state['patience']
        self.max_well = state['max_well']
        self.epoch = state['epoch']