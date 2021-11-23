"""
Contains a set of utility function to process data
"""

from __future__ import print_function

import csv
import datetime
import numpy as np
import h5py
import argparse
from sklearn.model_selection import train_test_split

start_date = '2016-11-07'
end_date = '2021-11-05'
date_format = '%Y-%m-%d'
start_datetime = datetime.datetime.strptime(start_date, date_format)
end_datetime = datetime.datetime.strptime(end_date, date_format)
number_datetime = (end_datetime - start_datetime).days + 1

exclude_set = set()

target_list = ['ADSK', 'CAG', 'BSX', 'ROK', 'AIG', 'XRAY', 'HBI', 'HPE', 'NSC', 'BIO', 'C', 'DAL', 'BBWI', 'GD', 'IPGP', 'PVH', 'INTU', 'EXR', 'CTSH', 'BWA', 'PEP', 'KIM', 'DD', 'ALK', 'UAA', 'CAT', 'PSA', 'DLR', 'ISRG', 'SJM', 'JNPR', 'UHS', 'ORLY', 'CBRE']

target_list_2 = ['FOX', 'FISV', 'EXPE', 'FAST', 'ESRX', 'DLTR', 'CTSH', 'CSCO', 'QCOM', 'PCLN', 'CELG',
                 'AMGN', 'WFM', 'WDC', 'NVDA', 'STX']


def normalize(x):
    """ Create a universal normalization function across close/open ratio

    Args:
        x: input of any shape

    Returns: normalized data

    """
    return (x - 1) * 100

def create_dataset(filepath, dic, datatype='stocks'):
    """ create the raw dataset from all_stock_5yr.csv. The data is Open,High,Low,Close,Volume

    Args:
        path: path of all_stocks_5yr.csv

    Returns:
        history: numpy array of size (N, number_day, 6),
        abbreviation: a list of company abbreviation where index map to name
        params['numVar']: date, open, close, high, low, volume
    """
    assert datatype in ['stocks', 'crypto'], "data can be stocks or crypto"
    params = dic[datatype] # num = number of companies, numVar: number of variables to consider, i=index of companyname
    history = np.empty(shape=(params['num'], number_datetime, params['numVar']), dtype=float)
    abbreviation = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first_row = next(reader)
        current_company = None
        current_company_index = -1
        current_date = None
        current_date_index = None
        previous_day_data = None
        for row in reader:
            if row[params['i']] in exclude_set:
                continue
            if row[params['i']] != current_company:
                current_company_index += 1
                # initialize
                if current_date != None and (current_date - end_datetime).days != 1:
                    print(row[params['i']])
                    print(current_date)
                assert current_date is None or (current_date - end_datetime).days == 1, \
                    'Previous end date is not 2021-11-05'
                current_date = start_datetime
                current_date_index = 0
                date = datetime.datetime.strptime(row[0], date_format)
                if (date - start_datetime).days != 0:
                    print(row[params['i']])
                    print(current_date)
                    exclude_set.add(row[params['i']])
                    current_date = end_datetime + datetime.timedelta(days=1)
                    continue
                assert (date - start_datetime).days == 0, 'Start date is not 2016-11-07'
                try:
                    if row[params['i']-1] == '':
                        row[params['i'] - 1] = 0
                    data = np.array(list(map(float, row[1:params['i']])))
                except:
                    print(row[params['i']])
                    assert False
                history[current_company_index][current_date_index] = data
                previous_day_data = data

                current_company = row[params['i']]
                abbreviation.append(current_company)
            else:
                date = datetime.datetime.strptime(row[0], date_format)
                # missing date, loop to the date difference is 0
                while (date - current_date).days != 0:
                    history[current_company_index][current_date_index] = previous_day_data.copy()
                    current_date += datetime.timedelta(days=1)
                    current_date_index += 1
                # miss data
                try:
                    data = np.array(list(map(float, row[1:params['i']])))
                except:
                    data = previous_day_data.copy()
                history[current_company_index][current_date_index] = data
                previous_day_data = data

            current_date += datetime.timedelta(days=1)
            current_date_index += 1
    write_to_h5py(history, abbreviation, filepath=f'utils/datasets/{datatype}_history.h5')


def write_to_h5py(history, abbreviation, filepath='utils/datasets/stocks_history.h5'):
    """ Write a numpy array history and a list of string to h5py

    Args:
        history: (N, timestamp, 6)
        abbreviation: a list of stock abbreviations

    Returns:

    """
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('history', data=history)
        abbr_array = np.array(abbreviation, dtype=object)
        string_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("abbreviation", data=abbr_array, dtype=string_dt)


def create_target_dataset(target_list=target_list, datatype='stocks', filepath='utils/datasets/stocks_history_target.h5'):
    """ Create 16 company history datasets

    Args:
        target_list:
        filepath:

    Returns:

    """
    if datatype == 'stocks':
        history_all, abbreviation_all = read_stock_history()
    elif datatype == 'crypto':
        # history_all, abbreviation_all = read_crypto_history()
        pass
    history = None
    for target in target_list:
        print(target)
        data = np.expand_dims(history_all[abbreviation_all.index(target)], axis=0)
        if history is None:
            history = data
        else:
            history = np.concatenate((history, data), axis=0)
    write_to_h5py(history, target_list, filepath=f'utils/datasets/{datatype}_history_target.h5')


def read_stock_history(filepath='utils/datasets/stocks_history.h5'):
    """ Read data from extracted h5

    Args:
        filepath: path of file

    Returns:
        history:
        abbreviation:

    """
    with h5py.File(filepath, 'r') as f:
        history = f['history'][:]
        abbreviation = f['abbreviation'][:].tolist()
        abbreviation = [abbr.decode('utf-8') for abbr in abbreviation]
    return history, abbreviation


def index_to_date(index):
    """

    Args:
        index: the date from start-date (2016-11-07)

    Returns:

    """
    return (start_datetime + datetime.timedelta(index)).strftime(date_format)


def date_to_index(date_string):
    """

    Args:
        date_string: in format of '2016-11-07'

    Returns: the days from start_date: '2016-11-07'

    >>> date_to_index('2016-11-07')
    0
    >>> date_to_index('2016-11-06')
    -1
    >>> date_to_index('2016-11-09')
    2
    """
    return (datetime.datetime.strptime(date_string, date_format) - start_datetime).days


def create_optimal_imitation_dataset(history, training_data_ratio=0.8, is_normalize=True):
    """ Create dataset for imitation optimal action given future observations
        label the dataset with the index of largest close/open ratio stock out of the 16 for each day

    Args:
        history: size of (num_stocks, T, num_features) contains (open, high, low, close)
        training_data_ratio: the ratio of training data

    Returns: un-normalized close/open ratio with size (T, num_stocks), labels: (T,)
             split the data according to training_data_ratio

    """
    num_stocks, T, num_features = history.shape
    cash_history = np.ones((1, T, num_features))
    history = np.concatenate((cash_history, history), axis=0)
    close_open_ratio = np.transpose(history[:, :, 3] / history[:, :, 0])
    if is_normalize:
        close_open_ratio = normalize(close_open_ratio)
    #labels will be the index of largest close/open ratio stock out of the 16 every day
    labels = np.argmax(close_open_ratio, axis=1)
    num_training_sample = int(T * training_data_ratio)
    return (close_open_ratio[:num_training_sample], labels[:num_training_sample]), \
           (close_open_ratio[num_training_sample:], labels[num_training_sample:])


def create_imitation_dataset(history, window_length, training_data_ratio=0.8, is_normalize=True):
    """ Create dataset for imitation optimal action given past observations

    Args:
        history: size of (num_stocks, T, num_features) contains (open, high, low, close)
        window_length: length of window as feature
        training_data_ratio: for splitting training data and validation data
        is_normalize: whether to normalize the data

    Returns: close/open ratio of size (num_samples, num_stocks, window_length)

    """
    num_stocks, T, num_features = history.shape
    cash_history = np.ones((1, T, num_features))
    history = np.concatenate((cash_history, history), axis=0)
    close_open_ratio = history[:, :, 3] / history[:, :, 0]
    if is_normalize:
        close_open_ratio = normalize(close_open_ratio)
    Xs = []
    Ys = []
    for i in range(window_length, T):
        obs = close_open_ratio[:, i - window_length:i]
        label = np.argmax(close_open_ratio[:, i:i+1], axis=0)
        Xs.append(obs)
        Ys.append(label)
    Xs = np.stack(Xs)
    Ys = np.concatenate(Ys)
    num_training_sample = int(T * training_data_ratio)
    return (Xs[:num_training_sample], Ys[:num_training_sample]), \
           (Xs[num_training_sample:], Ys[num_training_sample:])

def get_train_val_test_split_h5(h5_file):
    """

    Args:
        h5_file:

    Returns:

    """
    with h5py.File(h5_file, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
    return train_test_split(X, Y, test_size=0.1, random_state=42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", '-d', help="data stocks or crypto",default="stocks", choices=['stocks', 'crypto'])
    parser.add_argument("--filePath", '-f', help= "file path of csv file", default='utils/datasets/all_stocks_5yr.csv')
    parser.add_argument("--dataset", '-ds', help= "dataset type", default='create_dataset', choices=['create_dataset', 'create_target'])
    args = parser.parse_args()

    start_date = '2016-11-07' if args.data == 'stocks' else '2017-08-20'
    end_date = '2021-11-05'
    date_format = '%Y-%m-%d'
    start_datetime = datetime.datetime.strptime(start_date, date_format)
    end_datetime = datetime.datetime.strptime(end_date, date_format)
    number_datetime = (end_datetime - start_datetime).days + 1
    if args.data == 'stocks':
        exclude_set = set()

        target_list = ['ADSK', 'CAG', 'BSX', 'ROK', 'AIG', 'XRAY', 'HBI', 'HPE', 'NSC', 'BIO', 'C', 'DAL', 'BBWI', 'GD', 'IPGP', 'PVH', 'INTU', 'EXR', 'CTSH', 'BWA', 'PEP', 'KIM', 'DD', 
                        'ALK', 'UAA', 'CAT', 'PSA', 'DLR', 'ISRG', 'SJM', 'JNPR', 'UHS', 'ORLY', 'CBRE']

        target_list_2 = ['FOX', 'FISV', 'EXPE', 'FAST', 'ESRX', 'DLTR', 'CTSH', 'CSCO', 'QCOM', 'PCLN', 'CELG',
                 'AMGN', 'WFM', 'WDC', 'NVDA', 'STX']
    elif args.data == 'crypto':
        exclude_set = set()
        target_list = ['BTC-USD','ETH-USD','USDT-USD', 'XRP-USD', 'BNB-USD', 'ADA-USD', 'SOL1-USD', 'HEX-USD','DOT1-USD']
    dic = {'stocks': {'numVar': 6, "num": 503, 'i':7}, 'crypto':{'numVar':6, 'num':9, 'i':7}}
    if args.dataset == 'create_dataset':
        create_dataset(args.filePath, dic, args.data)
    else:
        create_target_dataset(target_list, args.data, args.filePath)

