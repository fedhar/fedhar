import numpy as np
from util import utils, data_process, train_utils, data_util
import torch
import role.client as cli
from torch.utils.data import DataLoader
from abc import abstractmethod
from tqdm import tqdm


class Trainer:
    def __init__(self, batch_size, min_second, max_second, lap, log_path, save_path, epochs, max_run, agg, batch_amp,
                 sup_batch_size, eval_seed):
        self.max_run = max_run
        self.batch_size = batch_size
        self.min_second = min_second
        self.max_second = max_second
        self.lap = lap
        self.save_path = save_path
        self.log_path = log_path
        self.epochs = epochs
        self.agg = agg
        self.batch_amp = batch_amp
        self.sup_batch_size = sup_batch_size
        self.eval_seed = eval_seed

        self.checkpoint = self.make_checkpoint()

        self.run_idx = None
        self.db = None
        self.staff_user = None
        self.model = None
        self.optimizer = None
        self.start_epoch, self.record = None, None

        self.eval_loader_list = None
        self.eval_score = None
        self.staff_list = None
        self.client_list = None
        self.sever = None

    def make_checkpoint(self):
        return utils.CheckPoint(self.log_path)

    def start(self, **kwargs):
        if not self.check_run():
            return

        self.db = self.create_database(**kwargs)

        while self.check_run():
            self.run_idx = self.create_read_run()
            self.staff_user = self.sample_read_staff(**kwargs)
            self.model = self.create_read_model()
            self.optimizer = self.create_read_optimizer()
            self.start_epoch, self.record = self.create_read_args()

            self.eval_loader_list = self.make_eval_loaders()
            self.sever = self.make_sever()
            self.staff_list = self.make_staffs()
            self.client_list = self.make_clients()
            self.eval_score = self.make_score()

            self.before_train(kwargs['verbose'])
            self.train(verbose=kwargs['verbose'], show_idx=kwargs['show_idx'], log_idx=kwargs['log_idx'],
                       tqdm_verbose=kwargs['tqdm_verbose'] if 'tqdm_verbose' in kwargs.keys() else None)
            self.save()

    def check_run(self):
        run_idx = self.checkpoint.get('run_idx')

        return run_idx is None or run_idx < self.max_run

    def create_read_run(self):
        run_idx = self.checkpoint.get('run_idx')
        if run_idx is None:
            run_idx = 0
            self.checkpoint.update(run_idx=run_idx)
            self.checkpoint.write()

        return run_idx

    def make_score(self):
        return train_utils.Metric([
                train_utils.Metric_Item(train_utils.acc_score, name='eval_acc', average=False),
                train_utils.Metric_Item(train_utils.f1_score, name='f1', average='macro'),
            ])

    def sample_read_staff(self, **kwargs):
        staff_user = self.checkpoint.get('staff_user')
        if staff_user is None:
            if 'staff_user' in kwargs.keys() and kwargs['staff_user'][self.run_idx] is not None:
                staff_user = kwargs['staff_user'][self.run_idx]
            else:
                staff_user = np.random.choice(range(self.db.owner.user_count), size=self.db.staff_num,
                                              replace=False).tolist()
            self.checkpoint.update(staff_user=staff_user)

        return staff_user

    def create_read_args(self):
        if self.checkpoint.get('start') is None:
            record, start = [], 0
            self.checkpoint.update(record=record, start=start)
        else:
            start, record, = self.checkpoint.gets('start', 'record')

        return start, record

    @abstractmethod
    def create_read_model(self):
        pass

    def create_read_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.99), weight_decay=0.0005)
        if self.checkpoint.get('optimizer') is not None:
            optimizer.load_state_dict(self.checkpoint.get('optimizer'))
        return optimizer

    def make_eval_loaders(self):
        eval_loader_list = []
        for idx in range(self.db.owner.user_count):
            eval_loader = DataLoader(
                data_util.ATTDataset(
                    *self.db.owner.choose_mul(idx, {key: 600 for key in self.db.owner.activity_tag.keys()},
                                              seed=self.eval_seed)), batch_size=256)
            eval_loader_list.append(eval_loader)

        return eval_loader_list

    def make_staffs(self):
        staff_list = []
        for idx, staff_idx in enumerate(self.staff_user):
            staff = cli.Staff(self.db.owner, staff_idx, self.db.sup_samples, self.model, self.sup_batch_size)
            staff.train_init(seconds=self.db.sup_seconds, fresh=self.db.sup_fresh, lap=0.)
            staff_list.append(staff)

        return staff_list

    def eval(self):
        result_list = [train_utils.eval(self.model, self.eval_loader_list[user_idx], self.eval_score)
                       for user_idx in range(self.db.owner.user_count) if user_idx not in self.staff_user]
        mean_value = {key: np.mean([result[key] for result in result_list]) for key in result_list[0].keys()}

        return result_list, mean_value

    def save(self):
        save = self.make_save_dict()
        utils.write_pkl(self.save_path.format(self.run_idx), save)
        self.run_idx = self.run_idx + 1
        self.checkpoint.clear()
        self.checkpoint.update(run_idx=self.run_idx)
        self.checkpoint.write()

    def make_save_dict(self):
        return {'model': self.model.cpu().state_dict(), 'staff_users': self.staff_user, 'record': self.record}

    @abstractmethod
    def make_clients(self):
        pass

    @abstractmethod
    def make_sever(self):
        pass

    @abstractmethod
    def create_database(self, **kwargs):
        pass

    def before_train(self, verbose):
        pass

    def train(self, verbose, show_idx, log_idx, tqdm_verbose):
        epit = range(self.start_epoch + 1, self.epochs + 1)
        epit = tqdm(epit) if verbose or tqdm_verbose else epit
        for ep in epit:
            flag = self.train_step(ep, verbose, show_idx, log_idx)
            if flag:
                break

    @abstractmethod
    def train_step(self, ep, verbose, show_idx, log_idx):
        pass


    def train_log(self, verbose, show_idx, log_idx, ep, **kwargs):
        if ep % show_idx == 0:
            result_list, mean_value = self.eval()
            if verbose:
                for result in result_list:
                    train_utils.show_result(ep, result)
                train_utils.show_result(ep, '****mean: ', mean_value)
            self.record.append([ep, mean_value, result_list])

        if ep % log_idx == 0:
            self.checkpoint.update(start=ep, model=self.model.state_dict(), record=self.record,
                                   optimizer=self.optimizer.state_dict(), **kwargs)
            self.checkpoint.write()
