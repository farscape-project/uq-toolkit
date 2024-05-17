import argparse
from copy import copy
import os
from shutil import copytree

import numpy as np
import hjson as json
import pyhit
import moosetree

import uq_sampler
from launcher import UQLauncher
from moose_tools import parse_moose_to_param_dict, setup_new_moose_input
from uq_logger import UncertaintyLogger

"""
Steps: 
1. read tabular data
2. fit polynomials and save coefficients to object orientated structure
3. send somewhere
4. sample
5. 

"""

def get_inputs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c",
        default="config.jsonc", 
        type=str,
        help="which json file to read?",
    )
    parser.add_argument("--copy-off",
        default=False,
        action="store_true",
        help="Do not copy dir",
    )
    return parser.parse_args()


def perturb_params(param_dict, sampler_dict) -> dict:
    perturbed_param_dict = dict.fromkeys(param_dict.keys()) # should be app-names
    for app_i in param_dict.keys():
        perturbed_param_dict[app_i] = dict.fromkeys(param_dict[app_i])
        for key_i in param_dict[app_i]:
            # get random sample from distribution defined in json
            perturbed_param_dict[app_i][key_i] = sampler_dict[app_i][key_i].get_sample(param_dict[app_i][key_i])
            # print(f"{key_i} is", param_dict[app_i][key_i], "sample is", perturbed_param_dict[app_i][key_i])
    return perturbed_param_dict

if __name__ == "__main__":
    args = get_inputs()
    with open(args.config, "r") as f:
        config = json.load(f)
    # setup helper classes
    launcher = UQLauncher(config["launcher"], config["template_launcher_script"])

    baselinedir_abs_path = os.path.join(config["paths"]["workdir"],config["paths"]["baseline_dir"])
    # parser for reading MOOSE input files
    moose_input_obj_list = []
    # moose_params_list = []
    # distrib_dict_list = []
    app_name_list = config["apps"].keys()
    uq_history = dict.fromkeys(app_name_list)
    moose_params = dict.fromkeys(app_name_list)
    distrib_dict = dict.fromkeys(app_name_list)
    for fname in app_name_list:
        uq_history[fname] = UncertaintyLogger()
        moose_input_obj_list.append(pyhit.load(f"{baselinedir_abs_path}/{fname}"))
        print(config["paths"])

        # find uncertain parameters in MOOSE input file corresponding to json
        moose_params_i, distrib_dict_i = parse_moose_to_param_dict(config["apps"][fname]["uncertain-params"], config["paths"], moose_input_obj_list[-1])
        moose_params[fname] = moose_params_i
        distrib_dict[fname] = distrib_dict_i


    print(distrib_dict)
    # sampler_dict = uq_sampler.setup_sample_dict(distrib_dict)
    sampler_dict = uq_sampler.setup_spacefilling_sampler(moose_params, distrib_dict, config["num_samples"])
    
    for sample_i in range(config["num_samples"]):
        sample_string = f"sample{sample_i}"

        new_dir = os.path.join(config["paths"]["workdir"], sample_string)
        if args.copy_off:
            pass
        else:
            copytree(baselinedir_abs_path, new_dir)

        # sample parameters
        perturbed_param_dict = perturb_params(moose_params, sampler_dict)
        for app_i in app_name_list:
            uq_history[app_i].record_sample(sample_string, perturbed_param_dict[app_i])

        # send sample parameters to corresponding locations in moose input file
        for (moose_input_obj, app_i) in zip(moose_input_obj_list, app_name_list):
            print(sample_string, app_i)
            setup_new_moose_input(config["apps"][app_i]["uncertain-params"], perturbed_param_dict[app_i], baselinedir_abs_path, new_dir, app_i, moose_input_obj)
    """   
        launcher.append_to_scheduler(new_dir)
    launcher.write_launcher(f"launcher.sh")
    """
    for app_i in app_name_list:
        uq_history[app_i].write_logger(f"{config['apps'][app_i]['uq_log_name']}.json")

