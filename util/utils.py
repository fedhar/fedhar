import pickle
import os
import shutil
import numpy as np


def read_pkl(path: str):
    f = open(path, 'rb+')
    data = pickle.load(f)
    return data


def write_pkl(path: str, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()


def read(path: str):
    with open(path, 'r') as f:
        data = f.read()

    return data

def create_dir(path, is_clear):
    if os.path.exists(path):
        if is_clear:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def sample_staffs(max_run, clients, staff_num, seed):
    staff_user = []
    while len(staff_user) < max_run:
        np.random.seed(seed)
        staff = np.random.choice(clients, size=staff_num, replace=False).tolist()
        staff.sort()
        if staff not in staff_user:
            staff_user.append(staff)
        seed = seed + 1

    return staff_user


class CheckPoint:
    def __init__(self, path, info=None):
        self.path = path
        if os.path.exists(self.path):
            self.save = self.load(info=info)
        else:
            self.save = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.save[key] = value

    def load(self, info):
        if info is None:
            print('load {}'.format(self.path))
        else:
            print(info)

        return read_pkl(self.path)

    def write(self, log=None):
        if log is not None:
            print(log)

        write_pkl(self.path, self.save)

    def gets(self, *args):
        return [self.save[arg] for arg in args]

    def get(self, key):
        if key in self.save.keys():
            return self.save[key]
        else:
            return None

    def clear(self):
        self.save = {}
