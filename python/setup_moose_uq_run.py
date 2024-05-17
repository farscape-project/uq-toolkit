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
    perturbed_param_dict = {}
    for key_i in param_dict:
        # get random sample from distribution defined in json
        perturbed_param_dict[key_i] = sampler_dict[key_i].get_sample(param_dict[key_i])
        print(f"{key_i} is", param_dict[key_i], "sample is", perturbed_param_dict[key_i])
    return perturbed_param_dict

if __name__ == "__main__":
    args = get_inputs()
    with open(args.config, "r") as f:
        config = json.load(f)
    # setup helper classes
    uq_history = UncertaintyLogger()#moose_params, config["baseline_dir"])
    launcher = UQLauncher(config["launcher"], config["template_launcher_script"])

    baselinedir_abs_path = os.path.join(config["workdir"],config["baseline_dir"])
    # parser for reading MOOSE input files
    moose_input_obj = pyhit.load(f"{baselinedir_abs_path}/{config['moose-input']}")

    # find uncertain parameters in MOOSE input file corresponding to json
    moose_params, distrib_dict = parse_moose_to_param_dict(config, moose_input_obj)

    sampler_dict = uq_sampler.setup_sample_dict(distrib_dict)
    
    for sample_i in range(config["num_samples"]):
        sample_string = f"sample{sample_i}"

        new_dir = os.path.join(config["workdir"], sample_string)
        if args.copy_off:
            pass
        else:
            copytree(baselinedir_abs_path, new_dir)

        # sample parameters
        perturbed_param_dict = perturb_params(moose_params, sampler_dict)
        uq_history.record_sample(sample_string, perturbed_param_dict)

        # send sample parameters to corresponding locations in moose input file
        setup_new_moose_input(config["uncertain-params"], perturbed_param_dict, baselinedir_abs_path, new_dir, config['moose-input'], moose_input_obj)
        launcher.append_to_scheduler(new_dir)
    launcher.write_launcher(f"launcher.sh")
    uq_history.write_logger(f"{config['uq_log_name']}.json")
    

