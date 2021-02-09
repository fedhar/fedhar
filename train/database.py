from abc import abstractmethod
from util import data_process


class Database:
    def __init__(self, staff_num, sup_seconds, sup_fresh, sup_samples, semi_seconds, semi_fresh, time_window,
                 allow_rate, freq_rate, data_path):
        self.staff_num = staff_num
        self.sup_seconds = sup_seconds
        self.sup_fresh = sup_fresh
        self.sup_samples = sup_samples
        self.semi_seconds = semi_seconds
        self.semi_fresh = semi_fresh
        self.time_window = time_window
        self.allow_rate = allow_rate
        self.freq_rate = freq_rate
        self.data_path = data_path
        self.owner = None

    @abstractmethod
    def make_owner(self):
        pass


class Realworld(Database):
    NAME = 'realworld'
    SECCONDS_PER_USER = 5302
    SAMPLES_PER_USRE = 2118  # 2.5s
    CLIENTS = 15

    SAMPLES_1 = 1200
    SAMPLES_2 = 1600
    SAMPLES_3 = 2000

    STAFF_1 = 3
    STAFF_2 = 4
    STAFF_3 = 5

    def __init__(self, **kwargs):
        super(Realworld, self).__init__(**{key: value for key, value in kwargs.items() if key in Database.__init__.__code__.co_varnames})

    def make_owner(self):
        self.owner = data_process.FftDataOwner(path=self.data_path, freq_rate=self.freq_rate,
                                               time_window=self.time_window,
                                               allow_rate=self.allow_rate, expand_last=True)

class Uci(Database):
    NAME = 'uci'
    SECCONDS_PER_USER = 472
    SAMPLES_PER_USRE = 187
    CLIENTS = 30

    SAMPLES_1 = 100
    SAMPLES_2 = 140
    SAMPLES_3 = 180

    STAFF_1S = 2
    STAFF_2S = 3
    STAFF_1 = 4
    STAFF_2 = 6
    STAFF_3 = 8

    def __init__(self, **kwargs):
        super(Uci, self).__init__(**{key: value for key, value in kwargs.items() if key in Database.__init__.__code__.co_varnames})

    def make_owner(self):
        self.owner = data_process.FftDataOwner(path=self.data_path, freq_rate=self.freq_rate,
                                               time_window=self.time_window,
                                               allow_rate=self.allow_rate, expand_last=True)

class Gleam(Database):
    NAME = 'gleam'
    SECCONDS_PER_USER = 1649
    SAMPLES_PER_USRE = 658
    CLIENTS = 38

    SAMPLES_1 = 300
    SAMPLES_2 = 500
    SAMPLES_3 = 700

    STAFF_1 = 4
    STAFF_2 = 6
    STAFF_3 = 8
    def __init__(self, **kwargs):
        super(Gleam, self).__init__(**{key: value for key, value in kwargs.items() if key in Database.__init__.__code__.co_varnames})

    def make_owner(self):
        self.owner = data_process.FftDataOwner(path=self.data_path, freq_rate=self.freq_rate,
                                               time_window=self.time_window,
                                               allow_rate=self.allow_rate, expand_last=True)