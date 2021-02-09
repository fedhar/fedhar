from train import trainer, database
from train.personlize import personalize_trainer as pt
import torch

class DirftPT(pt.PersonalizedTrainer):
    def __init__(self, near_semi, **kwargs):
        pt.PersonalizedTrainer.__init__(self,
            **{key: value for key, value in kwargs.items() if key in pt.PersonalizedTrainer.__init__.__code__.co_varnames})
        self.near_semi = near_semi
        self.lr = kwargs['lr'] if 'lr' in kwargs.keys() else None
        self.momentum = kwargs['momentum'] if 'momentum' in kwargs.keys() else 0.9

    def create_read_optimizer(self):
        if self.lr is None:
            optimizer = torch.optim.Adam(self.model.parameters(), betas=(self.momentum, 0.99), weight_decay=0.0005)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), betas=(self.momentum, 0.99), weight_decay=0.0005, lr=self.lr)

        if self.checkpoint.get('optimizer') is not None:
            optimizer.load_state_dict(self.checkpoint.get('optimizer'))
        return optimizer

    def sup_compute(self, staff, metric) -> dict:
        return staff.compute_grad(metric=metric)

    def semi_compute(self, client, metric) -> dict:
        return client.compute_grad(metric=metric)

    def semi_update(self, ep):
        self.sever.mean_update(semi=self.near_semi)

    def log_step(self, verbose, show_idx, log_idx, ep):
        self.train_log(verbose, show_idx, log_idx, ep)

class UciDriftPT(DirftPT, database.Uci):
    def __init__(self, **kwargs):
        DirftPT.__init__(self, **kwargs)
        database.Uci.__init__(self, **kwargs)

    def create_database(self, **kwargs):
        self.make_owner()
        return self

class GleamDriftPT(DirftPT, database.Gleam):
    def __init__(self, **kwargs):
        DirftPT.__init__(self, **kwargs)
        database.Gleam.__init__(self, **kwargs)

    def create_database(self, **kwargs):
        self.make_owner()
        return self

class RealworldDriftPT(DirftPT, database.Realworld):
    def __init__(self, **kwargs):
        DirftPT.__init__(self, **kwargs)
        database.Realworld.__init__(self, **kwargs)

    def create_database(self, **kwargs):
        self.make_owner()
        return self