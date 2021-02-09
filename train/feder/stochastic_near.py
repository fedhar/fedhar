from util.device import device
import role.online_client as onc
import role.sever as sev
from train import database
from train.feder import stochastic_trainer
from model import att_model


class StochasticNearTrainer(stochastic_trainer.StochasticTrainer):
    def __init__(self, semi=0.2, near_fn=None, **kwargs):
        stochastic_trainer.StochasticTrainer.__init__(self, **kwargs)
        self.semi = semi
        self.near_fn = near_fn

    def make_clients(self):
        client_list = []
        for user_idx in range(self.db.owner.user_count):
            if user_idx not in self.staff_user:
                client = onc.OnlineSemiClient(self.db.owner, user_idx, self.model,
                                              batch_size=self.batch_size * self.batch_amp, near_fn=self.near_fn)
                client.train_init(self.min_second, self.max_second, self.lap, self.db.semi_seconds, self.db.semi_fresh)
                client_list.append(client)

        return client_list

    def make_sever(self):
        sever = sev.NearSever(self.model)
        sever.train_init(self.optimizer)

        return sever

    def create_read_model(self):
        model = att_model.RealworldATTModel(self.db.owner.class_num, self.db.owner.sensor_dim, self.db.owner.num_in_sensors,
                                       self.db.owner.devices, series_dropout=0.2, device_dropout=0.2, sensor_dropout=0.2,
                                       predict_dropout=0.4).float().to(device)
        if self.checkpoint.get('model') is not None:
            model.load_state_dict(self.checkpoint.get('model'))

        return model

    def sup_compute(self, staff, metric):
        return staff.compute_grad(metric=metric)

    def semi_compute(self, client, metric):
        return client.compute_grad(metric=metric)

    def semi_update(self, ep):
        if ep <= 400:
            semi = (ep / 400) * self.semi
        else:
            semi = self.semi

        self.sever.mean_update(semi=semi)

    def log_step(self, verbose, show_idx, log_idx, ep):
        self.train_log(verbose, show_idx, log_idx, ep)

class RealworldStochasticNearTrainer(StochasticNearTrainer, database.Realworld):
    def __init__(self, **kwargs):
        StochasticNearTrainer.__init__(self, **kwargs)
        database.Realworld.__init__(self, **kwargs)

    def create_database(self, **kwargs):
        self.make_owner()
        return self


class UciStochasticNearTrainer(StochasticNearTrainer, database.Uci):
    def __init__(self, **kwargs):
        StochasticNearTrainer.__init__(self, **kwargs)
        database.Uci.__init__(self, **kwargs)

    def create_database(self, **kwargs):
        self.make_owner()
        return self

class GleamStochasticNearTrainer(StochasticNearTrainer, database.Gleam):
    def __init__(self, **kwargs):
        StochasticNearTrainer.__init__(self, **kwargs)
        database.Gleam.__init__(self, **kwargs)

    def create_database(self, **kwargs):
        self.make_owner()
        return self