from util import utils, Path
import numpy as np

activity_tag = {'walking': 0, 'upstairs': 1, 'downstairs': 2, 'sitting': 3, 'standing': 4, 'laying': 5}
sensor_freq = {'acc': 50, 'gyr': 50}
sensor_dim = {'acc': 3, 'gyr': 3}
locations = ['waist']

stamp = 1

def get_subject(group) -> np.ndarray:
    subj = utils.read(Path.raw_uci_subject.format(group=group))
    subj = [value.strip() for value in subj.split('\n') if value.strip() != '']
    subj = np.array(subj, dtype=int)

    return subj


def get_data(loc, group, sensor, axis) -> np.ndarray:
    data = utils.read(Path.raw_uci.format(group=group, sensor=sensor, axis=axis, loc=loc))
    data = [value.strip() for value in data.split('\n') if value.strip() != '']
    for idx in range(len(data)):
        data[idx] = [value.strip() for value in data[idx].split(' ') if value.strip() != '']
    data = np.array(data, dtype=float)

    return data


def get_label(group) -> np.ndarray:
    label = utils.read(Path.raw_uci_label.format(group=group))
    label = np.array([value.strip() for value in label.split('\n') if value.strip() != ''], dtype=int)

    return label - 1


def no_lap_index(data: np.ndarray):
    data_0 = data[:-1, 64:]
    data_1 = data[1:, :64]
    return np.nonzero((data_0 != data_1).any(axis=1))[0]

def flatten(data :np.ndarray):
    xs = []
    xs += data[0].tolist()

    for idx in range(1, len(data)):
        if np.all(data[idx-1][64:] == data[idx][:64]): # lap
            xs += data[idx][64:].tolist()
        else: # no lap
            xs += data[idx].tolist()

    return xs


def get_time(length, freq):
    result = []
    global stamp
    for idx in range(length):
        result.append(stamp)
        stamp = stamp + int(1000 // freq)

    return result


def extract_sensor(sensor_dict, index):
    sensor_csv_dict = {sensor: {} for sensor in sensor_freq.keys()}
    timeline = []
    for sensor, values in sensor_dict.items():
        data = []
        for value in values:
            data.append(flatten(value[index]))

        if len(timeline) == 0:  # initiate
            timeline = get_time(len(data[0]), sensor_freq[sensor])
        data.insert(0, timeline)
        data = np.array(data).swapaxes(0, 1)
        sensor_csv_dict[sensor][locations[0]] = data

    return sensor_csv_dict, (min(timeline), max(timeline))

def extract_action(label, start, sensor_dict):
    action_dict = {}
    num_dict = {}
    for y in np.unique(label):
        index = np.nonzero(label == y)[0] + start

        action = [key for key, value in activity_tag.items() if value == y][0]
        action_dict[action], num_dict[action] = extract_sensor(sensor_dict, index)

    return action_dict, num_dict


def extract(subject, label, sensor_dict):
    userset = []
    userlen = []
    for user in np.unique(subject):
        index = np.nonzero(subject == user)[0]
        start = min(index)
        end = max(index) + 1
        if (end - start) != len(index):
            raise Exception()

        action_dict, num_dict = extract_action(label[index], start, sensor_dict)
        userset.append(action_dict)
        userlen.append(num_dict)

    return userset, userlen


def save(name):
    userset = []
    userlen = []
    for group in ['train', 'test']:
        subject = get_subject(group)
        label = get_label(group)
        sensor_dict = {}
        for sensor, loc in [('acc', 'total'), ('gyr', 'body')]:
            sensor_dict[sensor] = []
            for axis in ['x', 'y', 'z']:
                sensor_dict[sensor].append(get_data(loc, group, sensor, axis))

        uset, ulen = extract(subject, label, sensor_dict)
        userset += uset
        userlen += ulen

    utils.write_pkl(Path.online_uci.format(name), [userset, userlen, activity_tag, sensor_freq, sensor_dim, locations])

save('all_all')
