from copy import copy
import hjson as json
import numpy as np

class UncertaintyLogger:
    def __init__(self):
        self.out_dict = {}
        # self.record_sample(initial_name, initial_dict)
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
        