from train import database
from train.feder import stochastic_near
from util import utils, Path
import os
from model import  layer
import numpy as np
from tqdm import tqdm
import warnings

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
show_idx = 20
log_idx = 50
sup_batch_size = 128
batch_amp = 1
agg = 40
max_run = 1
staff_user = None

sto_num = 9


def stochastic_near_train(staff_num, samples, epochs, staff_user, verbose=True):
    name = 'stochastic_near'
    stochastic_near.UciStochasticNearTrainer(
        sto_num=sto_num, batch_size=batch_size, min_second=min_second, max_second=max_second, lap=0.,
        log_path=Path.feder.format(dataset=dataset, staff_num=staff_num,
                                   samples=samples,
                                   model=name, name='7log'),
        save_path=Path.feder.format(dataset=dataset, staff_num=staff_num,
                                    samples=samples,
                                    model=name, name='7{}'),
        epochs=epochs, max_run=max_run, agg=agg, batch_amp=batch_amp,
        sup_batch_size=sup_batch_size,
        eval_seed=eval_seed, staff_num=staff_num, sup_fresh=sup_fresh,
        sup_seconds=sup_seconds,
        sup_samples=samples, semi_seconds=semi_seconds, semi_fresh=semi_fresh,
        time_window=time_window, allow_rate=allow_rate,
        freq_rate=freq_rate, data_path=path). \
        start(staff_user=staff_user, verbose=verbose, show_idx=10, log_idx=log_idx)

if __name__ == '__main__':
    epochs = 2000
    seeds = [201, 3, 23, 202, 202, 2002, 25, 5, 103]
    idx = 0
    for staff_num in [database.Uci.STAFF_1, database.Uci.STAFF_2, database.Uci.STAFF_3]:
        for samples in [database.Uci.SAMPLES_1, database.Uci.SAMPLES_2, database.Uci.SAMPLES_3]:
            np.random.seed(seeds[idx])
            staff_user = [np.random.choice(database.Uci.CLIENTS, size=staff_num, replace=False).tolist()] * max_run
            stochastic_near_train(staff_num=staff_num, samples=samples, epochs=epochs, staff_user=staff_user, verbose=False)