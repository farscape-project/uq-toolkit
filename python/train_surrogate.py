"""
Train an xgboost model to fit pod coefficients (pod_coefs_sample*.txt),
based on timestep and parameter (from uq_log*.json).
"""
import argparse
import logging
from glob import glob
from os.path import isfile

import hjson as json
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
        help="save model",
    )
    parser.add_argument(
        "--load-model",
        default=False,
        action="store_true",
        help="load model",
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


def sklearn_gpr(x_train, y_train, x_test, y_test, hyperparams):
    # set-up kernels based on information in the json file
    kernel = None
    for i, (name, props) in enumerate(hyperparams["kernels"].items()):
        if name == "RBF":
            if hyperparams["anisotropic-kernel"] and len(props["length_scale"]) == 1:
                props["length_scale"] = props["length_scale"]*x_train.shape[1]
        if kernel is None:
            kernel = getattr(kernels, name)(**props)
        else:
            if hyperparams["kernel-combine"] in ["add", "plus"]:
                kernel = kernel + getattr(kernels, name)(**props)
            else:
                kernel = kernel * getattr(kernels, name)(**props)

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
        if "model-file" in hyperparams.keys() and args.save_model:
            dump(gpr, hyperparams["model-file"])
    
    # log the outputs
    score_train = gpr.score(x_train, y_train)
    score_test = gpr.score(x_test, y_test)

    logger.info(f"kernel {gpr.kernel_}")
    logger.info(
        f"alpha min: {abs(gpr.alpha_).min()}, mean {abs(gpr.alpha_).mean()}, max: {abs(gpr.alpha_).max()}"
    )
    logger.info(f"log_marginal_likelihood {gpr.log_marginal_likelihood_value_}")
    return gpr, score_train, score_test


if __name__ == "__main__":
    args = get_inputs()
    logging.basicConfig(filename="train_surrogate.log", level=logging.INFO)

    sample_names = [
        s.split("/")[-1].split("_")[-1].split(".")[0]
        for s in glob(f"{args.pod_dir}/pod_coefs_sample*")
    ]

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
    x = []
    y = []
    for sample_i in sample_names:
        coefs_data = np.loadtxt(f"{args.pod_dir}/pod_coefs_{sample_i}.txt", skiprows=1)
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

                    for norm_param_name, norm_value in params[
                        "input-normalisations"
                    ].items():
                        if norm_param_name in key:
                            x_i[-1] = x_i[-1] / norm_value
            x.append(x_i)
            y_i = [dataset_coefs_pertime[sample_i][t_index][1]]
            y.append(y_i)

    # randomly shuffle and then split data into train/test
    TRAIN_FRACTION = 0.8
    x, y = shuffle(x, y)
    x = np.array(x)
    y = np.array(y)
    split_size = int(np.ceil(TRAIN_FRACTION * x.shape[0]))
    x_train = x[:split_size]
    y_train = y[:split_size]
    x_test = x[split_size:]
    y_test = y[split_size:]

    gpr, score_train, score_test = sklearn_gpr(x_train, y_train, x_test, y_test, params)

    logger.info(f"scores: train {score_train:5f}, test {score_test:5f}")
