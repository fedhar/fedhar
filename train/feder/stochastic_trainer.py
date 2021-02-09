import numpy as np
from util import utils, data_process, train_utils, data_util
from abc import abstractmethod
from train import trainer

class StochasticTrainer(trainer.Trainer):
    def __init__(self, sto_num, **kwargs):
        trainer.Trainer.__init__(self,
            **{key: value for key, value in kwargs.items() if key in trainer.Trainer.__init__.__code__.co_varnames})

        self.sto_num = sto_num

    def train_step(self, ep, verbose, show_idx, log_idx):
        staff_gl = []
        for staff in self.staff_list:
            grad_dict = self.sup_compute(staff, verbose)
            staff_gl.append(grad_dict)
        if verbose:
            train_utils.show_result(ep, [staff.train_score.value()['loss'] for staff in self.staff_list], )
        self.sever.staff_gl += staff_gl

        users = np.random.choice(len(self.client_list), size=self.sto_num, replace=False)
        for _ in range(self.agg):
            client_gl = []
            for idx in users:
                grad_dict = self.semi_compute(self.client_list[idx], metric=verbose)
                client_gl.append(grad_dict)
            if verbose:
                train_utils.show_result(ep, [self.client_list[idx].train_score.value()['loss'] for idx in users])
            self.sever.client_gl += client_gl

        self.semi_update(ep)
        self.log_step(verbose, show_idx, log_idx, ep)

    @abstractmethod
    def sup_compute(self, staff, metric) -> dict:
        pass

    @abstractmethod
    def semi_compute(self, client, metric) -> dict:
        pass

    @abstractmethod
    def semi_update(self, ep):
        pass

    @abstractmethod
    def log_step(self, verbose, show_idx, log_idx, ep):
        pass
