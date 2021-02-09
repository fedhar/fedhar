from util import utils, data_process, train_utils, data_util
from train import trainer, database
from model import att_model
from util.device import device
from role import online_client as onc
from torch.utils.data import DataLoader
from role import sever as sev
from abc import abstractmethod
import os
import torch

class PersonalizedTrainer(trainer.Trainer):
    def __init__(self, batch_size, min_second, max_second, lap, log_path, epochs, max_run, batch_amp,
                 sup_batch_size, eval_seed, model_path, user):
        trainer.Trainer.__init__(self, batch_size=batch_size, min_second=min_second, max_second=max_second,
                                 lap=lap, log_path=log_path, save_path=None, epochs=epochs, max_run=max_run,
                                 agg=None, batch_amp=batch_amp, sup_batch_size=sup_batch_size, eval_seed=eval_seed)

        self.model_path = model_path
        self.db_user = -1 # 真实号
        self.user = user # 虚拟号 1~
        self.stopper = None

    def make_checkpoint(self):
        return None

    def check_run(self):
        for idx in range(self.max_run):
            if os.path.exists(self.log_path.format(idx)):
                checkpoint = utils.CheckPoint(self.log_path.format(idx))
                start_epoch = checkpoint.get('start')
                if start_epoch is not None and start_epoch >= self.epochs:
                    continue
                else:
                    self.run_idx = idx  # 继续任务
                    self.checkpoint = checkpoint
                    return True
            else:
                self.run_idx = idx  # 新建任务
                self.checkpoint = utils.CheckPoint(self.log_path.format(self.run_idx))
                self.checkpoint.update(run_idx=self.run_idx)
                self.checkpoint.write()
                return True

        return False

    def create_read_run(self):
        start_epoch = self.checkpoint.get('start')
        start_epoch = 0 if start_epoch is None else start_epoch
        print('start {}-th run from epoch :{}'.format(self.run_idx, start_epoch))
        return self.run_idx

    def sample_read_staff(self, **kwargs):
        staff_user = self.checkpoint.get('staff_user')
        if staff_user is None:
            staff_user = utils.read_pkl(self.model_path)['staff_users']
            self.checkpoint.update(staff_user=staff_user)

        cs = []
        for idx in range(self.db.owner.user_count):
            if idx not in staff_user:
                cs.append(idx)
        self.db_user = cs[self.user]

        return staff_user

    def create_read_model(self):
        model = att_model.RealworldATTModel(self.db.owner.class_num, self.db.owner.sensor_dim,
                                            self.db.owner.num_in_sensors,
                                            self.db.owner.devices, series_dropout=0.2, device_dropout=0.2,
                                            sensor_dropout=0.2,
                                            predict_dropout=0.4).float().to(device)
        if self.checkpoint.get('model') is not None:
            model.load_state_dict(self.checkpoint.get('model'))
        else:
            state = utils.read_pkl(self.model_path)['model']
            model.load_state_dict(state)

        torch.cuda.empty_cache()
        return model

    def make_eval_loaders(self):
        eval_loader = DataLoader(
            data_util.ATTDataset(
                *self.db.owner.choose_mul(self.db_user, {key: 600 for key in self.db.owner.activity_tag.keys()},
                                          seed=self.eval_seed)), batch_size=256)

        return [eval_loader]

    def make_sever(self):
        sever = sev.NearSever(self.model)
        sever.train_init(self.optimizer)

        return sever

    def make_clients(self):
        client = onc.OnlineSemiClient(self.db.owner, self.db_user, self.model, batch_size=self.batch_size * self.batch_amp)
        client.train_init(self.min_second, self.max_second, self.lap, self.db.semi_seconds, self.db.semi_fresh)

        return [client]

    def before_train(self, verbose):
        self.create_read_stopper(verbose)

    def create_read_stopper(self, verbose):
        if self.checkpoint.get('stopper') is not None:
            self.stopper = train_utils.ValueStopper(None, None, None, verbose=verbose)
            self.stopper.load_state_dict(self.checkpoint.get('stopper'))
        else:
            result = self.eval()
            self.stopper = train_utils.ValueStopper(value=result['eval_acc'], patience=200, max_well=True, verbose=verbose)

    def train_step(self, ep, verbose, show_idx, log_idx):
        if ep == 1:
            self.train_log(verbose, show_idx, log_idx, 0)

        staff_gl = []
        for staff in self.staff_list:
            grad_dict = self.sup_compute(staff, verbose)
            staff_gl.append(grad_dict)
        self.sever.staff_gl += staff_gl

        grad_dict = self.semi_compute(self.client_list[0], metric=verbose)
        self.sever.client_gl += [grad_dict]

        self.semi_update(ep)

        self.log_step(verbose, show_idx, log_idx, ep)

        if ep % show_idx == 0:
            return self.stopper(self.record[-1][1]['eval_acc'], inc=show_idx)

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

    def train_log(self, verbose, show_idx, log_idx, ep, **kwargs):
        if ep % show_idx == 0:
            result = self.eval()
            if verbose:
                train_utils.show_result(ep,'****eval: ', result)
            self.record.append([ep, result])

        if ep % log_idx == 0:
            self.checkpoint.update(start=ep, model=self.model.cpu().state_dict(), record=self.record,
                                   optimizer=self.optimizer.state_dict(), stopper=self.stopper.state_dict(), **kwargs)
            self.checkpoint.write()

    def eval(self):
        return train_utils.eval(self.model, self.eval_loader_list[0], self.eval_score)

    def save(self):
        self.checkpoint.update(start=self.epochs)
        self.checkpoint.write()
        self.checkpoint = utils.CheckPoint(self.log_path.format(self.run_idx))