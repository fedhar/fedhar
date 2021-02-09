root = '/root/project/fos'

# raw data root
raw_realworld_root = root + '/data/raw/realworld/'
raw_realworld =  root + '/data/raw/realworld/proband{user_idx}/data/{sensor}_{action}_csv.zip'
raw_realworld_csv = '{sensor}_{action}_{loc}.csv'

raw_uci =  root + '/data/raw/UCI HAR Dataset/{group}/Inertial Signals/{loc}_{sensor}_{axis}_{group}.txt'
raw_uci_label =  root + '/data/raw/UCI HAR Dataset/{group}/y_{group}.txt'
raw_uci_subject =  root + '/data/raw/UCI HAR Dataset/{group}/subject_{group}.txt'

raw_gleam =  root + '/data/raw/GLEAM/{user_idx}/{user_idx}_sensorData.csv'
raw_gleam_label =  root + '/data/raw/GLEAM/{user_idx}/{user_idx}_annotate.csv'


# train data
train_realworld =  root + '/data/train/realworld/{}.pkl'
online_realworld =  root + '/data/train/realworld/online/{}.pkl'

online_uci =  root + '/data/train/uci/{}.pkl'

online_gleam =  root + '/data/train/gleam/{}.pkl'

# drift result for sup-online
drift_sup_online =  root + '/result/drift/sup-online/{}.pkl'
drift_sup_online_continue =  root + '/result/drift/sup-online/continue/{}.pkl'

# feder result
feder = root + '/result/feder/{dataset}/{staff_num} clients/{samples} samples/{model}/{name}.pkl'
feder_csv =  root + '/result/feder/csv/{}.csv'

# personalize
personalize =  root + '/result/personalize/{dataset}/{staff_num} clients/{samples} samples/{user}/{name}.pkl'
temp_personalize_csv =  root + '/result/personalize/{dataset}/csv/{dataset}_{name}.csv'

# arg
arg = root + '/result/arg/{dataset}/{staff_num} clients/{samples} samples/{arg}/{name}.pkl'

# loss
loss = root + '/result/loss/{dataset}/{staff_num} clients/{samples} samples/{name}.pkl'

# attention imp
att_imp =  root + '/result/attention/{}.pkl'