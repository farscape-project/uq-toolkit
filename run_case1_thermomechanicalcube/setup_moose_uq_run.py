from copy import copy
import os
from shutil import copytree

import numpy as np
import hjson as json
import pyhit
import moosetree

"""
Steps: 
1. read tabular data
2. fit polynomials and save coefficients to object orientated structure
3. send somewhere
4. sample
5. 

"""

def perturbed_poly(x, coeffs):
    deg_list = np.arange(0, coeffs.size, 1)[::-1]
    print(deg_list)
    print(coeffs)
    # set fine sampling for x axis, and use this to sample function
    # x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = np.sum(coeffs*(x[:,None]**deg_list[None,:]), axis=-1)

    return y_fit

def parse_moose_to_param_dict(config):
    param_dict = {}
    for key_i in config["uncertain-params"]:
        for param_i in config["uncertain-params"][key_i]:
            moose_path = f'/{key_i}/{param_i}'
            param_moose = moosetree.find(root, func=lambda n: n.fullpath == moose_path)
            # print(param_i, list(param_moose.params()))
            
            if config["uncertain-params"][key_i][param_i]["type"] == "csv":
                data = np.loadtxt(os.path.join(config['workdir'], config['baseline_dir'],param_moose["data_file"]), delimiter=",")
                x = data[0]
                y = data[1]
            elif config["uncertain-params"][key_i][param_i]["type"] == "xy":
                x = np.array(param_moose["x"].split(), dtype=float)
                y = np.array(param_moose["y"].split(), dtype=float)
            elif config["uncertain-params"][key_i][param_i]["type"] == "value":
                param_dict[moose_path] = copy(float(param_moose["value"]))
            else:
                raise NotImplementedError
            
            if "fit_poly" in config["uncertain-params"][key_i][param_i].keys():
                coeffs = np.polyfit(x,y, deg=config["uncertain-params"][key_i][param_i]["fit_poly"]["deg"])
                param_dict[moose_path] = copy(coeffs)
    
    return param_dict

def perturb_params(param_dict) -> dict:
    perturbed_param_dict = {}
    for key_i in param_dict:
        if type(param_dict[key_i]) is float:
            perturbed_param_dict[key_i] = param_dict[key_i] * np.random.normal(1, 0.01)
        elif type(param_dict[key_i]) is np.ndarray:
            perturbed_param_dict[key_i] = param_dict[key_i] * np.random.normal(1, 0.01, size=param_dict[key_i].shape)
        else:
            raise TypeError(f"incorrect type for {param_dict[key_i]} is {type(param_dict[key_i])}")
    return perturbed_param_dict

def setup_new_moose_input(config, perturbed_param_dict, basedir_abs, sample_dir_abs, moose_input):

    for key_i in config:
        for param_i in config[key_i]:
            moose_path = f'/{key_i}/{param_i}'
            param_moose = moosetree.find(root, func=lambda n: n.fullpath == moose_path)
            
            if config[key_i][param_i]["type"] == "csv":
                data = np.loadtxt(os.path.join(basedir_abs, param_moose["data_file"]), delimiter=",")
                x = data[0]
                if "fit_poly" in config[key_i][param_i].keys():
                    y_out = perturbed_poly(x, perturbed_param_dict[moose_path])
                    out_path = f"{sample_dir_abs}/{param_moose['data_file']}"
                    print("writing perturbed file to ", out_path)
                    np.savetxt(out_path, np.c_[x, y_out].T)
                
            elif config[key_i][param_i]["type"] == "xy":
                x = np.array(param_moose["x"].split(), dtype=float)
                y = np.array(param_moose["y"].split(), dtype=float)
                if "fit_poly" in config[key_i][param_i].keys():
                    y_out = perturbed_poly(x, perturbed_param_dict[moose_path])
                    param_moose["y"] = " ".join(map(str, y_out.tolist()))
                    print(y_out)
                else:
                    print(config[key_i][param_i].keys())
                    raise NotImplementedError
            elif config[key_i][param_i]["type"] == "value":
                param_moose["value"] = perturbed_param_dict[moose_path]
            else:
                raise NotImplementedError
            
            # if "fit_poly" in config["uncertain-params"][key_i][param_i].keys():
            #     coeffs = np.polyfit(x,y, deg=config["uncertain-params"][key_i][param_i]["fit_poly"]["deg"])
            #     param_dict[moose_path] = copy(coeffs)
    pyhit.write(f"{sample_dir_abs}/{moose_input}", root)
    return None

class UQLauncher:
    def __init__(self, launcher_type, launcher_script):
        self.launcher_type = launcher_type
        self.launch_string = self._get_launcher_string()
        assert os.path.exists(f"basedir/{launcher_script}"), "template launcher script does not exist"
        self.launcher_script = launcher_script
        self.scheduler_text = self._setup_scheduler()

    def _get_launcher_string(self):
        if self.launcher_type == "bash":
            return "bash "
        elif self.launcher_type == "slurm":
            return "sbatch "
        elif self.launcher_type == "lsf":
            return "bsub < "
        else:
            raise NotImplementedError

    def _setup_scheduler(self):
        # TODO: Add slurm and others
        scheduler_text = []
        if self.launcher_type == "bash":
            # note that when using mpi, each simulation is run in parallel, but the batch itself is run in serial
            scheduler_text.append("#!/bin/bash\n\n")
            pass
        else:
            raise NotImplementedError
        return scheduler_text

    def append_to_scheduler(self, newdir):
        self.scheduler_text.append(f"cd {newdir}\n")
        self.scheduler_text.append(f"{self.launch_string} {self.launcher_script} \n")
        self.scheduler_text.append(f"cd -\n")

    def write_launcher(self, launcher_name):
        if self.launcher_type == "bash":
            self._write_bash_launcher(launcher_name)

    def _write_bash_launcher(self, launcher_name):
        with open(launcher_name, "w") as f:
            [f.write(line_i) for line_i in self.scheduler_text]

class UncertaintyLogger:
    def __init__(self, initial_dict, initial_name):
        self.out_dict = {}
        self.record_sample(initial_name, initial_dict)
        pass

    def record_sample(self, sample_name, sample_dict):
        self.out_dict[sample_name] = {}
        for key_i, value_i in sample_dict.items():
            value_to_store = copy(value_i)
            if isinstance(value_to_store, np.ndarray):
                value_to_store = value_to_store.tolist()
            self.out_dict[sample_name][key_i] = value_to_store

    def write_logger(self, logname):
        with open(logname, 'w', encoding='utf-8') as f:
            json.dump(self.out_dict, f, ensure_ascii=False, indent=4)
        

if __name__ == "__main__":
    tracker_dict = {}
    with open("config.jsonc", "r") as f:
        config = json.load(f)

    baselinedir_abs_path = os.path.join(config["workdir"],config["baseline_dir"])
    # parser for reading MOOSE input files
    root = pyhit.load(f"{baselinedir_abs_path}/{config['moose-input']}")

    # find uncertain parameters in MOOSE input file corresponding to json
    moose_params = parse_moose_to_param_dict(config)
    print(moose_params)
    uq_history = UncertaintyLogger(moose_params, config["baseline_dir"])

    # setup class to make script for running simulations
    launcher = UQLauncher(config["launcher"], config["template_launcher_script"])
    
    for sample_i in range(config["num_samples"]):
        sample_string = f"sample{sample_i}"

        new_dir = os.path.join(config["workdir"], sample_string)
        # copytree(baselinedir_abs_path, new_dir)

        # sample parameters
        perturbed_param_dict = perturb_params(moose_params)
        uq_history.record_sample(sample_string, perturbed_param_dict)

        # send sample parameters to corresponding locations in moose input file
        setup_new_moose_input(config["uncertain-params"], perturbed_param_dict, baselinedir_abs_path, new_dir, config['moose-input'])
        launcher.append_to_scheduler(new_dir)
    launcher.write_launcher("launcher.sh")
    uq_history.write_logger("uq_log.json")
    

