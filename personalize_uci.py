from train import database
from train.personlize import drift
import warnings
import os
import numpy as np
from util import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

print('pid: ', os.getpid())

dataset = database.Uci.NAME
path = Path.online_uci.format('all_all')
eval_seed = 4
time_window = int(2.5 * 1000)
allow_rate = 0.8
freq_rate = 1.
min_second = int(np.ceil(time_window / 1000))
max_second = 600
batch_size = 24  # 60s
sup_seconds = 600
sup_fresh = 200
semi_seconds = 1200
semi_fresh = 200
log_idx = 50
sup_batch_size = 128
batch_amp = 1
max_run = 2

staff_num = database.Uci.STAFF_1
samples = database.Uci.SAMPLES_1


def drift_train(epochs, user, verbose, show_idx, near_semi=0.2, lr=None, momentum=0.9):
    name = 'new_drift_{}-{}-{}'.format(near_semi, lr, momentum)
    drift.UciDriftPT(
        user=user, batch_size=batch_size, min_second=min_second, max_second=max_second, lap=0.,
        model_path=Path.feder.format(dataset=dataset, staff_num=staff_num,
                                     samples=samples, model='stochastic_near', name='0_1'),
        log_path=Path.personalize.format(dataset=dataset, staff_num=staff_num, samples=samples, user=user, name=name + '_{}'),
        epochs=epochs, max_run=max_run, batch_amp=batch_amp,
        sup_batch_size=sup_batch_size,
        eval_seed=eval_seed, staff_num=staff_num, sup_fresh=sup_fresh,
        sup_seconds=sup_seconds,
        sup_samples=samples, semi_seconds=semi_seconds, semi_fresh=semi_fresh,
        time_window=time_window, allow_rate=allow_rate,
        freq_rate=freq_rate, data_path=path, near_semi=near_semi, lr=lr, momentum=momentum). \
        start(verbose=verbose, show_idx=show_idx, log_idx=log_idx)


if __name__ == '__main__':
    epoch = 5000
    show_idx = 1
    search_args = {
        'near_semi': [0.01, 0.1, 0.2],
        'lr_momentum': [(None, 0.9), (None, 0.), (1e-4, 0.9)],
    }

    for user in range(database.Uci.CLIENTS - database.Uci.STAFF_3):
        for near_semi in search_args['near_semi']:
            for lr, momentum in search_args['lr_momentum']:
                drift_train(epoch, user, verbose=True, show_idx=show_idx, near_semi=near_semi, lr=lr, momentum=momentum)