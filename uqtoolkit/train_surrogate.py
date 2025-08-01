"""
Train an xgboost model to fit pod coefficients (pod_coefs_sample*.txt),
based on timestep and parameter (from uq_log*.json).
"""
import argparse
import logging
from glob import glob
from os import makedirs
from os.path import isfile

import hjson as json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.utils import shuffle
from skops.io import dump, get_untrusted_types, load

logger = logging.getLogger(__name__)


def get_inputs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path-to-samples",
        default=".",
        type=str,
        help="/path/to/samples basedir",
    )
    parser.add_argument(
        "--uq-config",
        "-c",
        default="config.jsonc",
        type=str,
        help="name of config file used for uq jobs",
    )
    parser.add_argument(
        "--surrogate-params",
        default="gpr_params.jsonc",
        type=str,
        help="name of config file used for surrogate model training",
    )
    parser.add_argument(
        "--pod-dir",
        default=None,
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

def duplicate_kernels(kernel, num_duplicates=1):
    kernel_out = kernel
    for i in range(num_duplicates-1):
        kernel_out = combine_kernels(kernel_out, kernel)
    return kernel_out        

def combine_kernels(kernel_orig, kernel_to_add, combine_method="add"):
    if combine_method == "add":
        out_kernel = kernel_orig + kernel_to_add
    else:
        out_kernel = kernel_orig * kernel_to_add
    return out_kernel


def sklearn_gpr(x_train, y_train, x_test, y_test, hyperparams):
    # set-up kernels based on information in the json file
    kernel = None
    for i, (name, props) in enumerate(hyperparams["kernels"].items()):
        if name == "RBF":
            if hyperparams["anisotropic-kernel"] and len(props["length_scale"]) == 1:
                props["length_scale"] = props["length_scale"]*x_train.shape[1]
        if i == 0:
            kernel = getattr(kernels, name)(**props)
            try:
                kernel = duplicate_kernels(kernel, hyperparams["num-kernels"][name])
            except KeyError:
                pass
        else:
            kernel_i = getattr(kernels, name)(**props)
            try:
                kernel_i = duplicate_kernels(kernel_i, hyperparams["num-kernels"][name])
            except KeyError:
                pass
            kernel = combine_kernels(kernel, getattr(kernels, name)(**props), hyperparams["kernel-combine"])


    # use skops to save/load gpr model in a safe way
    model_found = False
    if "model-file" in hyperparams.keys():
        model_file = hyperparams["model-file"]
        if isfile(model_file) and args.load_model:
            model_found = True
            unknown_types = get_untrusted_types(file=model_file)
            print("loading gpr")
            gpr = load(model_file, trusted=unknown_types)
    # if no model available, or --load-model not provided, then we fit the GPR
    if not model_found:
        gpr = GPR(kernel=kernel, **hyperparams["GPR-params"]).fit(x_train, y_train)
    
    # log the outputs
    score_train = gpr.score(x_train, y_train)
    score_test = gpr.score(x_test, y_test)

    logger.info(f"kernel {gpr.kernel_}")
    logger.info(
        f"alpha min: {abs(gpr.alpha_).min()}, mean {abs(gpr.alpha_).mean()}, max: {abs(gpr.alpha_).max()}"
    )
    logger.info(f"log_marginal_likelihood {gpr.log_marginal_likelihood_value_}")
    return gpr, score_train, score_test

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

                    for offset_param_name, offset_value in params[
                        "input-offset"
                    ].items():
                        if offset_param_name in key:
                            x_i[-1] = x_i[-1] - offset_value

                    for norm_param_name, norm_value in params[
                        "input-normalisations"
                    ].items():
                        if norm_param_name in key:
                            x_i[-1] = x_i[-1] / norm_value
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

                for offset_param_name, offset_value in params[
                    "input-offset"
                ].items():
                    if offset_param_name in key:
                        x_i[-1] = x_i[-1] - offset_value

                for norm_param_name, norm_value in params[
                    "input-normalisations"
                ].items():
                    if norm_param_name in key:
                        x_i[-1] = x_i[-1] / norm_value
        x.append(x_i)
        y_i = dataset_coefs_pertime[sample_i]
        y.append(y_i)
    return x, y

if __name__ == "__main__":
    args = get_inputs()
    logging.basicConfig(filename="train_surrogate.log", level=logging.INFO)
    
    # by default, POD data will be in the results dir. 
    # But in the nextflow pipeline, we will write to a temporary cache directory.
    if args.pod_dir is None:
        POD_DIR = f"{args.path_to_samples}/pod_data"
    else:
        POD_DIR = args.pod_dir
    
    # make plotting directory
    makedirs(f"{args.path_to_samples}/plots", exist_ok=True)

    # read all existing sample names
    with open(f"{args.path_to_samples}/complete_samples.txt", "r") as f:
        file_lines = f.read()
        sample_names = file_lines.strip().split("\n")

    with open(args.uq_config) as f:
        uq_config = json.load(f)

    # read model hyperparams from file, or set to default
    if isfile(args.surrogate_params):
        with open(args.surrogate_params, "r") as f:
            params = json.load(f)
    else:
        raise AssertionError(f"not found {args.surrogate_params}")
    logger.info(f"surrogate hyperparams {params}")

    app_log_list = find_uqlog_names(uq_config, results_dir=args.path_to_samples)
    app_key_name_list = find_uqparam_names(app_log_list, "sample0")

    dataset_coefs_pertime = dict.fromkeys(sample_names)
    if args.steady_state:
        data_func = load_data_steady
    else:
        data_func = load_data_timedependent
        
    x, y = data_func(sample_names, params, app_log_list, app_key_name_list, POD_DIR)

    # randomly shuffle and then split data into train/test
    TRAIN_FRACTION = 0.8
    x, y = shuffle(x, y)
    x = np.array(x)
    y = np.array(y)[:,args.pod_coef:args.pod_coef+1]
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

    gpr, score_train, score_test = sklearn_gpr(x_train, y_train, x_test, y_test, params)
    if "model-file" in params.keys() and args.save_model:
        dump(gpr, f'{POD_DIR}/my_gpr_podcoef{args.pod_coef}.skops')

    final_result = f"scores: train {score_train:5f}, test {score_test:5f}"
    print(final_result)
    logger.info(final_result)

    y_pred_train_all = []
    y_pred_train_all_std = []
    for x_train_i, y_train_i in zip(x_train, y_train):
        y_pred, y_std = gpr.predict(x_train_i.reshape(1, -1), return_std=True)
        y_pred_train_all.append(y_pred.reshape(-1, num_out_features))
        y_pred_train_all_std.append(y_std.reshape(-1, num_out_features))

    y_pred_test_all = []
    y_pred_test_all_std = []
    for x_test_i, y_test_i in zip(x_test, y_test):
        y_pred, y_std = gpr.predict(x_test_i.reshape(1, -1), return_std=True)
        y_pred_test_all.append(y_pred.reshape(-1, num_out_features))
        y_pred_test_all_std.append(y_std.reshape(-1, num_out_features))

    # shape is [N_sample, 1, N_features], so we slice axis=1 
    y_pred_train_all = np.array(y_pred_train_all)[:,0]
    y_pred_test_all = np.array(y_pred_test_all)[:,0]
    y_pred_train_all_std = np.array(y_pred_train_all_std)[:,0]
    y_pred_test_all_std = np.array(y_pred_test_all_std)[:,0]

    errbar_kwargs = dict(ls="none", alpha=0.2)

    plt.close("all")
    i = 0
    fig, ax = plt.subplots(ncols=2, sharey=True)
    ax[0].set_title(fr"Train ($R^2={score_train:4f}$)")
    ax[0].scatter(y_train[:,i], y_pred_train_all[:,i], c="blue", s=10)
    ax[0].errorbar(y_train[:,i], y_pred_train_all[:,i], yerr=y_pred_train_all_std[:,i]*2, **errbar_kwargs)
    ax[0].scatter(y_train[:,i], y_train[:,i], c="black", s=1)
    ax[0].set_ylabel("predicted")
    ax[0].set_xlabel("true")
    ax[1].set_xlabel("true")
    ax[1].set_title(fr"Test ($R^2={score_test:4f}$)")
    ax[1].scatter(y_test[:,i], y_pred_test_all[:,i], c="blue", s=10)
    ax[1].errorbar(y_test[:,i], y_pred_test_all[:,i], yerr=y_pred_test_all_std[:,i]*2, c="blue", **errbar_kwargs)
    ax[1].scatter(y_test[:,i], y_test[:,i], c="black", s=1)
    plt.savefig(f"{args.path_to_samples}/plots/gpr-calibration-{args.pod_coef}.png", dpi=300)

