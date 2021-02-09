from train import database
from train.feder import stochastic_near
from util import utils, Path
import os
import numpy as np
from tqdm import tqdm
from model import layer
import warnings

warnings.filterwarnings("ignore")

print('pid: ', os.getpid())

dataset = database.Realworld.NAME
path = Path.online_realworld.format('all_sel')
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
show_idx = 20
log_idx = 100
sup_batch_size = 128
batch_amp = 1
agg=20
max_run = 3
staff_user = None

sto_num = 5


def stochastic_near_train(staff_num, samples, epochs, staff_user, verbose=True):
    name = 'stochastic_near'
    stochastic_near.RealworldStochasticNearTrainer(
        sto_num=sto_num, batch_size=batch_size, min_second=min_second, max_second=max_second, lap=0.,
        log_path=Path.feder.format(dataset=dataset, staff_num=staff_num,
                                   samples=samples,
                                   model=name, name='2log'),
        save_path=Path.feder.format(dataset=dataset, staff_num=staff_num,
                                    samples=samples,
                                    model=name, name='2{}'),
        epochs=epochs, max_run=max_run, agg=agg, batch_amp=batch_amp,
        sup_batch_size=sup_batch_size,
        eval_seed=eval_seed, staff_num=staff_num, sup_fresh=sup_fresh,
        sup_seconds=sup_seconds,
        sup_samples=samples, semi_seconds=semi_seconds, semi_fresh=semi_fresh,
        time_window=time_window, allow_rate=allow_rate,
        freq_rate=freq_rate, data_path=path). \
        start(staff_user=staff_user, verbose=verbose, show_idx=show_idx, log_idx=log_idx)

if __name__ == '__main__':
    epochs = 2000
    seeds = [16, 16, 16, 22, 202, 4, 1751, 25, 7]
    idx = 0
    for staff_num in [database.Realworld.STAFF_1, database.Realworld.STAFF_2, database.Realworld.STAFF_3]:
        for samples in [database.Realworld.SAMPLES_1, database.Realworld.SAMPLES_2, database.Realworld.SAMPLES_3]:
            np.random.seed(seeds[idx])
            staff_user = [np.random.choice(database.Realworld.CLIENTS, size=staff_num, replace=False).tolist()] * max_run
            stochastic_near_train(staff_num=staff_num, samples=samples, epochs=epochs, staff_user = staff_user, verbose=False)