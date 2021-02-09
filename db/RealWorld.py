import pandas as pd
import zipfile
from util import utils, Path

sensor_freq = {'acc': 49.95, 'gyr': 49.95, 'mag': 49.95}
sensor_dim = {'acc': 3, 'gyr': 3, 'mag': 3}
locations = ['chest', 'head', 'waist']
activity_tag = {'running': 0, 'standing': 1, 'climbingup': 2, 'lying': 3, 'sitting': 4, 'walking': 5, 'climbingdown': 6, 'jumping': 7}
window_size = 2.5

value_length = {'acc': 120, 'gyr': 120, 'mag': 120}


def cutoff(user_idx, action: str):
    sensor_csv_dict = {sensor: {} for sensor in sensor_freq.keys()}
    start_time = []
    end_time = []

    for sensor in sensor_freq.keys():
        with zipfile.ZipFile(Path.raw_realworld.format(user_idx=user_idx, sensor=sensor, action=action), 'r') as z:
            for loc in locations:
                csv_file_name = Path.raw_realworld_csv.format(action=action, sensor=sensor, loc=loc)
                if csv_file_name in z.namelist():
                    df = pd.read_csv(z.open(csv_file_name))
                    start_time.append(min(df['attr_time'].values))
                    end_time.append(max(df['attr_time'].values))
                    sensor_csv_dict[sensor][loc] = df
                else:
                    sensor_csv_dict[sensor][loc] = None

    start_time = max(start_time)
    end_time = min(end_time)
    for sensor in sensor_freq.keys():
        for loc in locations:
            if sensor_csv_dict[sensor][loc] is not None:
                df: pd.DataFrame = sensor_csv_dict[sensor][loc]
                df = df[['attr_time', 'attr_x', 'attr_y', 'attr_z']]
                df = df[(df['attr_time'] >= start_time) & (df['attr_time'] <= end_time)]
                df = df.sort_values(by = 'attr_time', ascending=True)
                sensor_csv_dict[sensor][loc] = df.values

    return sensor_csv_dict, (start_time, end_time)

def save(name):
    userset = []
    userlen = []
    for user_idx in range(1, 16):
        action_dict = {}
        num_dict = {}
        for action in activity_tag.keys():
            action_dict[action], num_dict[action] = cutoff(user_idx, action)
        userlen.append(num_dict)
        userset.append(action_dict)
    utils.write_pkl(Path.online_realworld.format(name), [userset, userlen, activity_tag, sensor_freq, sensor_dim, locations])

save('all_sel')