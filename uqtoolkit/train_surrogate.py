import argparse
import logging
from glob import glob
from os import makedirs
from os.path import isfile

import hjson as json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import torch
from autoemulate.core.compare import AutoEmulate
from autoemulate.emulators import GaussianProcess
from uqtoolkit import SurrogateCLI

logger = logging.getLogger(__name__)


def get_inputs():
    parser = SurrogateCLI()
    parser.add_argument(
        "--surrogate-params",
        default="gpr_params.jsonc",
        type=str,
        help="name of config file used for surrogate model training",
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
        help="save model",
    )
    parser.add_argument(
        "--load-model",
        default=False,
        action="store_true",
        help="load model",
    )
    parser.add_argument(
        "--steady-state",
        default=False,
        action="store_true",
        help="data is not time-dependent",
    )
    parser.add_argument(
        "--pod-coef",
        default=0,
        type=int,
        help="POD coeff to train",
    )
    return parser.parse_args()


def find_uqlog_names(uq_config, results_dir="."):
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
        with open(f"{results_dir}/{uq_log_fname}", "r") as f:
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


def find_uqparam_human_names(uq_config, results_dir="."):
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
    human_names = []

    # hjson uses ordered dicts, so ordering is consistent every time
    for key_i in uq_config["apps"]:
        uq_params = uq_config["apps"][key_i]["uncertain-params"]
        for param_i in uq_params.values():
            print(param_i)
            human_names.append(param_i["human_name"])

    return human_names

def load_data_timedependent(sample_names, params, app_log_list, app_key_name_list, pod_dir):
    dataset_coefs_pertime = dict.fromkeys(sample_names)
    x = []
    y = []
    for sample_i in sample_names:
        coefs_data = np.loadtxt(f"{pod_dir}/pod_coefs_{sample_i}.txt", skiprows=1)
        time_arr = coefs_data[:, 0]
        dataset_coefs_pertime[sample_i] = coefs_data[:, 1:]
        for t_index, time_i in enumerate(time_arr):
            # x_arr is [t, uq_0, uq_1, ..., uq_n]
            # y is [pod_coef_0, ..., pod_coef_n]
            # all data should be [0,1], so we apply some normalisations
            if "time" in params["input-normalisations"]:
                time_i = time_i / params["input-normalisations"]["time"]

            x_i = [time_i]
            y_i = []

            for app_log, key_list in zip(app_log_list, app_key_name_list):
                for key in key_list:
                    # if "Current" in key:
                    if isinstance(app_log[sample_i][key], list):
                        x_i.extend(app_log[sample_i][key])
                    else:
                        x_i.append(app_log[sample_i][key])

                    # for offset_param_name, offset_value in params[
                    #     "input-offset"
                    # ].items():
                    #     if offset_param_name in key:
                    #         x_i[-1] = x_i[-1] - offset_value

                    # for norm_param_name, norm_value in params[
                    #     "input-normalisations"
                    # ].items():
                    #     if norm_param_name in key:
                    #         x_i[-1] = x_i[-1] / norm_value
            x.append(x_i)
            y_i = [dataset_coefs_pertime[sample_i][t_index][1]]
            y.append(y_i)
    return x, y

def load_data_steady(sample_names, params, app_log_list, app_key_name_list, pod_dir):
    dataset_coefs_pertime = dict.fromkeys(sample_names)
    x = []
    y = []
    for sample_i in sample_names:
        coefs_data = np.loadtxt(f"{pod_dir}/pod_coefs_{sample_i}.txt", skiprows=1)
        dataset_coefs_pertime[sample_i] = coefs_data
        x_i = []
        y_i = []
        # x_arr is [t, uq_0, uq_1, ..., uq_n]
        # y is [pod_coef_0, ..., pod_coef_n]
        for app_log, key_list in zip(app_log_list, app_key_name_list):
            for key in key_list:
                if isinstance(app_log[sample_i][key], list):
                    x_i.extend(app_log[sample_i][key])
                else:
                    x_i.append(app_log[sample_i][key])

                # for offset_param_name, offset_value in params[
                #     "input-offset"
                # ].items():
                #     if offset_param_name in key:
                #         x_i[-1] = x_i[-1] - offset_value

                # for norm_param_name, norm_value in params[
                #     "input-normalisations"
                # ].items():
                #     if norm_param_name in key:
                #         x_i[-1] = x_i[-1] / norm_value
        x.append(x_i)
        y_i = dataset_coefs_pertime[sample_i]
        y.append(y_i)
    return x, y

if __name__ == "__main__":
    args = get_inputs()
    logging.basicConfig(filename="train_surrogate.log", level=logging.INFO)

    # read all existing sample names
    with open(f"{args.path_to_samples}/complete_samples.txt", "r") as f:
        file_lines = f.read()
        sample_names = file_lines.strip().split("\n")

    # read model hyperparams from file, or set to default
    if isfile(args.surrogate_params):
        with open(args.surrogate_params, "r") as f:
            params = json.load(f)
    else:
        raise AssertionError(f"not found {args.surrogate_params}")
    logger.info(f"surrogate hyperparams {params}")

    app_log_list = find_uqlog_names(args.uq_config, results_dir=args.path_to_samples)
    app_key_name_list = find_uqparam_names(app_log_list, "sample0")

    dataset_coefs_pertime = dict.fromkeys(sample_names)
    if args.steady_state:
        data_func = load_data_steady
    else:
        data_func = load_data_timedependent
        
    x, y = data_func(sample_names, params, app_log_list, app_key_name_list, args.pod_dir)

    # randomly shuffle and then split data into train/test
    TRAIN_FRACTION = 0.8
    x, y = shuffle(x, y, random_state=63)
    x = np.array(x)
    y = np.array(y)[:,args.pod_coef:args.pod_coef+1]
    assert y.size > 0, "target array is empty"
    # print(y.size)
    split_size = int(np.ceil(TRAIN_FRACTION * x.shape[0]))
    x_train = x[:split_size]
    y_train = y[:split_size]
    x_test = x[split_size:]
    y_test = y[split_size:]
    num_out_features = y.shape[1]
    # add different statistics to log file
    logger.info(f"input feature names values {app_key_name_list}")
    for feature in ["min", "max", "mean", "std"]:
        func_to_check = getattr(np, feature)
        logger.info(f"input feature {feature} values {func_to_check(x_train, axis=0)}")

    if args.load_model:
        best = AutoEmulate.load_model(f"{args.pod_dir}/ae_best_{args.pod_coef}")
    else:
        ae = AutoEmulate(x_train, y_train, log_level="progress_bar", models=[GaussianProcess])
        print(ae.summarise())
        best_result = ae.best_result()
        best = best_result.model
        print("Model with id: ", best_result.id, " performed best: ", best_result.model_name)
    if args.save_model:
        best_result_filepath = ae.save(best, f"{args.pod_dir}/ae_best_{args.pod_coef}", use_timestamp=True)
    y_pred_ae = best.predict(torch.tensor(x_test).to(torch.float32)).mean.numpy().squeeze()
    print(y_pred_ae)
    print("autoemulate error")
    print(np.c_[y_test.squeeze(), y_pred_ae, abs(y_test.squeeze() - y_pred_ae)])

    y_pred_train_all_ae = []
    y_pred_train_all_std_ae = []
    for x_train_i, y_train_i in zip(x_train, y_train):
        y_pred_ae = best.predict(torch.tensor(x_train_i[None]).to(torch.float32))
        y_pred_train_all_ae.append(y_pred_ae.mean.numpy())
        y_pred_train_all_std_ae.append(y_pred_ae.variance.numpy()**0.5)

    y_pred_test_all_ae = []
    y_pred_test_all_std_ae = []
    for x_test_i, y_test_i in zip(x_test, y_test):
        y_pred_ae = best.predict(torch.tensor(x_test_i[None]).to(torch.float32))
        y_pred_test_all_ae.append(y_pred_ae.mean.numpy())
        y_pred_test_all_std_ae.append(y_pred_ae.variance.numpy()**0.5)

    # shape is [N_sample, 1, N_features], so we slice axis=1 
    y_pred_train_all_ae = np.array(y_pred_train_all_ae)[:,0]
    y_pred_train_all_std_ae = np.array(y_pred_train_all_std_ae)[:,0]
    y_pred_test_all_ae = np.array(y_pred_test_all_ae)[:,0]
    y_pred_test_all_std_ae = np.array(y_pred_test_all_std_ae)[:,0]

    errbar_kwargs = dict(ls="none", alpha=0.2)

    plt.close("all")
    i = 0
    fig, ax = plt.subplots(ncols=2, sharey=True)
    # autoemulate
    ax[0].scatter(y_train[:,i], y_pred_train_all_ae[:,i], c="red", s=10)
    ax[0].errorbar(y_train[:,i], y_pred_train_all_ae[:,i], yerr=y_pred_train_all_std_ae[:,i]*2, c="red", **errbar_kwargs)
    # identity
    ax[0].scatter(y_train[:,i], y_train[:,i], c="black", s=1)
    # autoemulate
    ax[1].scatter(y_test[:,i], y_pred_test_all_ae[:,i], c="red", s=10)
    ax[1].errorbar(y_test[:,i], y_pred_test_all_ae[:,i], yerr=y_pred_test_all_std_ae[:,i]*2, c="red", **errbar_kwargs)
    # identity
    ax[1].scatter(y_test[:,i], y_test[:,i], c="black", s=1)
    # formatting
    ax[0].set_ylabel("predicted")
    ax[0].set_xlabel("true")
    ax[1].set_xlabel("true")
    plt.savefig(f"{args.plot_dir}/gpr-calibration-{args.pod_coef}.png", dpi=300)

