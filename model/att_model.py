import model.model as mod
from model.layer import Attention
from torch.nn import Dropout, Sequential, Linear, LeakyReLU, BatchNorm1d, Module


class BaseATTModel(mod.ATTModel):
    SERIES_OUT = 256
    DEVICE_OUT = 64
    SENSOR_OUT = 32

    def __init__(self, class_num, sensor_dim, num_in_sensors, devices, series_dropout=0., device_dropout=0., sensor_dropout=0., predict_dropout=0.):
        super(BaseATTModel, self).__init__(class_num, sensor_dim, num_in_sensors, devices, series_dropout, device_dropout, sensor_dropout, predict_dropout)

    def attention4series(self, in_feature, out_feature, num):
        attention = Attention(out_feature)
        return Sequential(
            # BatchNorm1d(num),
            Linear(in_feature, 64), Dropout(self.series_dropout), LeakyReLU(),
            Linear(64, 128), Dropout(self.series_dropout), LeakyReLU(),
            Linear(128, out_feature), Dropout(self.series_dropout), LeakyReLU(),
            attention, Dropout(self.series_dropout)
        ), attention

    def attention4device(self, in_feature, out_feature):
        attention = Attention(out_feature)
        return Sequential(
            Linear(in_feature, out_feature), Dropout(self.device_dropout), LeakyReLU(),
            attention, Dropout(self.device_dropout),
        ), attention

    def attention4sensor(self, input_feature, out_feature):
        attention = Attention(out_feature)
        return Sequential(
            Linear(input_feature, out_feature), Dropout(self.sensor_dropout), LeakyReLU(),
            attention, Dropout(self.sensor_dropout)
        ), attention

    def merge_module(self, input_feature, out_feature) -> Module:
        return Sequential(
            Linear(input_feature, 16), Dropout(self.predict_dropout), LeakyReLU(),
            Linear(16, out_feature)
        )


class RealworldATTModel(BaseATTModel):
    def __init__(self, *args, **kwargs):
        super(RealworldATTModel, self).__init__(*args, **kwargs)


class UciATTModel(BaseATTModel):
    def __init__(self, *args, **kwargs):
        super(UciATTModel, self).__init__(*args, **kwargs)


class GleamATTModel(BaseATTModel):
    def __init__(self, *args, **kwargs):
        super(GleamATTModel, self).__init__(*args, **kwargs)