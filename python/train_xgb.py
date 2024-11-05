"""
Train an xgboost model to fit pod coefficients (pod_coefs_sample*.txt),
based on timestep and parameter (from uq_log*.json).
"""
import argparse
from glob import glob
import numpy as np
from copy import copy
import hjson as json
import matplotlib.pyplot as plt
from os import makedirs
from os.path import isfile
from sklearn.utils import shuffle
import xgboost as xgb

def get_inputs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uq-config",
        "-c",
        default="config.jsonc", 
        type=str,
        help="name of config file used for uq jobs",
    )
    parser.add_argument(
        "--xgboost-params",
        default="params_xgb.jsonc", 
        type=str,
        help="name of config file used for xgboost training",
    )
    parser.add_argument(
        "--pod-dir",
        default="pod_data/", 
        type=str,
        help="/path/to/dir containing pod_coefs_sample*.txt",
    )
    parser.add_argument(
        "--num-epoch",
        "-ne",
        default=500, 
        type=int,
        help="iterations for xgboost training",
    )
    parser.add_argument(
        "--save-model",
        default=False,
        action="store_true",
        help="save xgb model as bin file",
    )
    return parser.parse_args()

def find_uqlog_names(uq_config):
    """
    Find names of log json files for each app's uncertainties
    This is given in the uq_log config file

    Parameters
    ----------
    uq_config : ordereddict
        uq-toolkit config file. Has sub-dict entry for each app in hierarchy.

    Returns
    -------
    app_log_list : list of ordereddicts
        each entry has list of uncertain parameter values for the 
        corresponding moose app.
    """
    app_log_list = []

    # hjson uses ordered dicts, so ordering is consistent every time
    for key_i in uq_config["apps"]:
        uq_log_fname = f'{uq_config["apps"][key_i]["uq_log_name"]}.json'
        with open(uq_log_fname, "r") as f:
            app_log = json.load(f)
        app_log_list.append(app_log)
    
    return app_log_list

def find_uqparam_names(log_list, sample):
    """
    Extract uncertain parameters as keys from uq log files.

    Parameters
    ----------
    log_list : list of ordereddicts
        contains uq_log_*.json file data created by uq-toolkit
    sample : str
        can be any sample string, the keys that we extract will be the same

    Returns
    -------
    key_name_list : list of lists of strs
        each sublist contains keys corresponding to values of UQ params
    """
    key_name_list = [None] * len(log_list)
    for i, app_log in enumerate(log_list):
        key_name_list[i] = []
        for key in app_log[sample].keys():
            key_name_list[i].append(key)
    return key_name_list


if __name__ == "__main__":
    args = get_inputs()

    sample_names = [
        s.split("/")[-1].split('_')[-1].split('.')[0] for s in glob(f"{args.pod_dir}/pod_coefs_sample*")
    ]

    with open(args.uq_config) as f:
        uq_config = json.load(f)

    app_log_list = find_uqlog_names(uq_config)
    app_key_name_list = find_uqparam_names(app_log_list, "sample0")

    dataset_coefs_pertime = dict.fromkeys(sample_names)
    x = []
    y = []
    for sample_i in sample_names:
        coefs_data = np.loadtxt(f"{args.pod_dir}/pod_coefs_{sample_i}.txt", skiprows=1)
        time_arr = coefs_data[:, 0]
        dataset_coefs_pertime[sample_i] = coefs_data[:, 1:]
        for t_index, time_i in enumerate(time_arr):
            # x_arr is [t, uq_0, uq_1, ..., uq_n]
            # y is [pod_coef_0, ..., pod_coef_n]
            x_i = [time_i]
            y_i = []

            for app_log, key_list in zip(app_log_list, app_key_name_list):
                for key in key_list:
                    if isinstance(app_log[sample_i][key], list):
                        x_i.extend(app_log[sample_i][key])
                    else:
                        x_i.append(app_log[sample_i][key])
            x.append(x_i)
            y_i = dataset_coefs_pertime[sample_i][t_index]
            y.append(y_i)

    # randomly shuffle and then split data into train/test
    x, y = shuffle(x, y)
    x = np.array(x)
    y = np.array(y)
    split_size = int(np.ceil(0.8*x.shape[0]))
    x_train = x[:split_size]
    y_train = y[:split_size]
    x_test = x[split_size:]
    y_test = y[split_size:]
    
    # read model hyperparams from file, or set to default
    if isfile(args.xgboost_params):
        with open(args.xgboost_params, "r") as f:
            params = json.load(f)
    else:
        params = {
            "max_depth": 2, # depth of tree
            "eta": 0.01, # learning rate
            "min_child_weight" : 0.010, # default is 1
            "subsample": 0.5, # randomly subsample dataset by this fraction
            "gamma" : 0.0 # default is 0, loss needed to make new leaf
        }

    # add data to xgb native format
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test) # not needed for now
    evallist = [(dtrain, 'train'), (dtest, 'eval')]

    bst = xgb.train(params, dtrain, args.num_epoch, evallist)
    ypred = bst.predict(dtest)
    # output info on error computed on test set
    test_error = abs(ypred - y_test)
    print(f"test error. mean: {test_error.mean()}, max: {test_error.max()}")
    if args.save_model:
        bst.save_model('xgb_model.bin')
