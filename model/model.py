from torch import nn
import torch
from torch import stack, cat
from torch.nn import Linear, Sequential, LeakyReLU, Module, Dropout, BatchNorm1d
from model.layer import Attention
from typing import Tuple, Dict
from abc import ABCMeta, abstractmethod

class ATTModel(Module):
    IMP4SENSORS = 1
    IMP4DEVICES = 2
    IMP4SERIES = 3

    def __init__(self, class_num, sensor_dim, num_in_sensors, devices, series_dropout=0., device_dropout=0., sensor_dropout=0., predict_dropout=0.):
        super(ATTModel, self).__init__()

        self.series_dropout = series_dropout
        self.device_dropout = device_dropout
        self.sensor_dropout = sensor_dropout
        self.predict_dropout = predict_dropout

        self.series_attention_layers = []
        self.device_attention_layers = []

        self.device_attentions: Dict[str, Attention] = {}
        self.series_attentions: Dict[str, Dict[str, Attention]] = {}

        for sensor in sensor_dim.keys():
            dev_layers = []
            self.series_attentions[sensor] = {}
            for dev in devices:
                series_layer, self.series_attentions[sensor][dev] = self.attention4series(sensor_dim[sensor], self.SERIES_OUT, num_in_sensors[sensor])
                self.add_module(name='{}_{}_attention'.format(sensor, dev), module=series_layer)
                dev_layers.append(series_layer)

            self.series_attention_layers.append(dev_layers)
            device_attention, self.device_attentions[sensor] = self.attention4device(self.SERIES_OUT, self.DEVICE_OUT)

            self.add_module(name='{}_attention'.format(sensor), module=device_attention)
            self.device_attention_layers.append(device_attention)

        self.sensor_attention_layer, self.sensor_attention = self.attention4sensor(self.DEVICE_OUT, self.SENSOR_OUT)
        self.predictor = self.merge_module(self.SENSOR_OUT, class_num)

    def forward(self, input: list):
        sensor_out = []
        for sensor_idx, sensor_in in enumerate(input):
            dev_out = []
            for dev_idx, dev_in in enumerate(sensor_in):
                dev_x = self.series_attention_layers[sensor_idx][dev_idx](dev_in)
                dev_out.append(dev_x)

            dev_out = stack(dev_out, dim=1)
            dev_out = self.device_attention_layers[sensor_idx](dev_out)
            sensor_out.append(dev_out)

        sensor_out = stack(sensor_out, dim=1)
        sensor_out = self.sensor_attention_layer(sensor_out)
        label = self.predictor(sensor_out)

        return label

    @abstractmethod
    def attention4series(self, in_feature, out_feature, num):
        pass

    @abstractmethod
    def attention4device(self, in_feature, out_feature):
        pass

    @abstractmethod
    def attention4sensor(self, input_feature, out_feature):
        pass

    @abstractmethod
    def merge_module(self, input_feature, out_feature) -> Module:
        pass

    def record_on(self, codes):
        if self.IMP4SENSORS in codes:
            self.sensor_attention.record_on()
        if self.IMP4DEVICES in codes:
            for attention in self.device_attentions.values():
                attention.record_on()
        if self.IMP4SERIES in codes:
            for series_attentions in self.series_attentions.values():
                for attention in series_attentions.values():
                    attention.recode_on()

    def record_off(self, codes):
        if self.IMP4SENSORS in codes:
            self.sensor_attention.record_off()
        if self.IMP4DEVICES in codes:
            for attention in self.device_attentions.values():
                attention.record_off()
        if self.IMP4SERIES in codes:
            for series_attentions in self.series_attentions.values():
                for attention in series_attentions.values():
                    attention.recode_off()

    def get_record(self, code, sensor=None, dev=None, mean=True):
        if code == self.IMP4SENSORS:
            recode_value = self.sensor_attention.record_value
            return recode_value
        elif code == self.IMP4DEVICES:
            if sensor is None:
                if mean:
                    return torch.stack([attention.record_value for attention in self.device_attentions.values()], dim=0).mean(dim=0)
                else:
                    return {key: attention.record_value for key, attention in self.device_attentions.items()}
            else:
                return self.device_attentions[sensor].record_value
        elif code == self.IMP4SERIES:
            return self.series_attentions[sensor][dev].record_value

    def regular(self):
        return 0.
