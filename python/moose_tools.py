from copy import copy
from os import path
import moosetree
import pyhit
import numpy as np

def parse_moose_to_param_dict(uq_config, path_config, moose_input):
    param_dict = {}
    distrib_dict = {}
    for key_i in uq_config:
        for param_i in uq_config[key_i]:
            moose_path = f'/{key_i}/{param_i}'
            param_moose = moosetree.find(moose_input, func=lambda n: n.fullpath == moose_path)
            
            distrib_dict[moose_path] = copy(uq_config[key_i][param_i]["distribution"])
            if uq_config[key_i][param_i]["type"] == "csv":
                data = np.loadtxt(path.join(path_config['workdir'], path_config['baseline_dir'],param_moose["data_file"]), delimiter=",")
                x = data[0]
                y = data[1]
            elif uq_config[key_i][param_i]["type"] == "xy":
                x = np.array(param_moose["x"].split(), dtype=float)
                y = np.array(param_moose["y"].split(), dtype=float)
            elif uq_config[key_i][param_i]["type"] == "value":
                value_name = uq_config[key_i][param_i]["value_name"]
                param_dict[moose_path] = copy(float(param_moose[value_name]))
            else:
                raise NotImplementedError
            
            if "fit_poly" in uq_config[key_i][param_i].keys():
                coeffs = np.polyfit(x,y, deg=uq_config[key_i][param_i]["fit_poly"]["deg"])
                param_dict[moose_path] = copy(coeffs)
    
    return param_dict, distrib_dict

def setup_new_moose_input(config, perturbed_param_dict, basedir_abs, sample_dir_abs, moose_input_fname, moose_input_obj):

    for key_i in config:
        for param_i in config[key_i]:
            moose_path = f'/{key_i}/{param_i}'
            param_moose = moosetree.find(moose_input_obj, func=lambda n: n.fullpath == moose_path)

            if config[key_i][param_i]["type"] == "csv":
                data = np.loadtxt(path.join(basedir_abs, param_moose["data_file"]), delimiter=",")
                x = data[0]
                if "fit_poly" in config[key_i][param_i].keys():
                    y_out = perturbed_poly(x, perturbed_param_dict[moose_path])
                    out_path = f"{sample_dir_abs}/{param_moose['data_file']}"
                    np.savetxt(out_path, np.c_[x, y_out].T)
                
            elif config[key_i][param_i]["type"] == "xy":
                x = np.array(param_moose["x"].split(), dtype=float)
                y = np.array(param_moose["y"].split(), dtype=float)
                if "fit_poly" in config[key_i][param_i].keys():
                    y_out = perturbed_poly(x, perturbed_param_dict[moose_path])
                    param_moose["y"] = " ".join(map(str, y_out.tolist()))
                else:
                    raise NotImplementedError
            elif config[key_i][param_i]["type"] == "value":
                value_name = config[key_i][param_i]["value_name"]
                param_moose[value_name] = perturbed_param_dict[moose_path]
            else:
                raise NotImplementedError
            
    pyhit.write(f"{sample_dir_abs}/{moose_input_fname}", moose_input_obj)
    return None

def perturbed_poly(x, coeffs):
    """
    Take a perturbed set of polynomial coefficients, and compute y for a given series of x
    """
    deg_list = np.arange(0, coeffs.size, 1)[::-1]
    # # set fine sampling for x axis, and use this to sample function
    # x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = np.sum(coeffs*(x[:,None]**deg_list[None,:]), axis=-1)
    return y_fit
