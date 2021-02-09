from util import train_utils
import role.online_client as onc
import role.sever as sev
from tqdm import tqdm
from train import trainer


class NearTrainer(trainer.Trainer):
    def __init__(self, **kwargs):
        super(NearTrainer, self).__init__(
            **{key: value for key, value in kwargs.items() if key in trainer.Trainer.__init__.__code__.co_varnames})

    def make_clients(self):
        client_list = []
        for user_idx in range(self.db.owner.user_count):
            if user_idx not in self.staff_user:
                client = onc.OnlineSemiClient(self.db.owner, user_idx, self.model,
                                              batch_size=self.batch_size * self.batch_amp)
                client.train_init(self.min_second, self.max_second, self.lap, self.db.semi_seconds, self.db.semi_fresh)
                client_list.append(client)

        return client_list

    def make_sever(self):
        sever = sev.NearSever(self.model)
        sever.train_init(self.optimizer)

        return sever

    def train(self, verbose, show_idx, log_idx):
        epit = range(self.start_epoch + 1, self.epochs + 1)
        epit = tqdm(epit) if verbose else epit
        for ep in epit:
            staff_gl = []
            for staff in self.staff_list:
                grad_dict = staff.compute_grad(metric=verbose)
                staff_gl.append(grad_dict)
            if verbose:
                train_utils.show_result(ep, [staff.train_score.value()['loss'] for staff in self.staff_list], )
            self.sever.staff_gl += staff_gl

            for _ in range(self.agg):
                client_gl = []
                for idx, client in enumerate(self.client_list):
                    grad_dict = client.compute_grad(metric=verbose)
                    client_gl.append(grad_dict)
                if verbose:
                    train_utils.show_result(ep, [client.train_score.value()['near'] for client in self.client_list])
                self.sever.client_gl += client_gl

            semi = 0.04

            self.sever.mean_update(semi=semi)

            self.train_log(verbose, show_idx, log_idx, ep)