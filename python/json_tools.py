from copy import copy
from os import path
import hjson as json
import numpy as np

def parse_json_to_param_dict(uq_config, json_input_file):
    param_dict = {}
    distrib_dict = {}

    for param_i in uq_config:
        # setup dict key to be same as MOOSE, for simplicity
        moose_path = f'/{param_i}'
        param_json = json_input_file[param_i]

        distrib_dict[moose_path] = copy(uq_config[param_i]["distribution"])
        
        if uq_config[param_i]["type"] == "value":
            param_dict[moose_path] = copy(float(param_json))
        else:
            raise NotImplementedError
    return param_dict, distrib_dict

def setup_new_json_input(config, perturbed_param_dict, basedir_abs, sample_dir_abs, moose_input_fname, moose_input_obj):

    for param_i in config:
        moose_path = f'/{param_i}'

        if config[param_i]["type"] == "value":
            moose_input_obj[param_i] = perturbed_param_dict[moose_path]
        else:
            raise NotImplementedError
            
    with open(f"{sample_dir_abs}/{moose_input_fname}", 'w', encoding='utf-8') as f:
        json.dump(moose_input_obj, f, ensure_ascii=False, indent=4)
    return None

