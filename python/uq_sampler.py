from abc import ABC, abstractmethod, abstractproperty
import numpy as np

class SampleDistributionBase(ABC):

    @abstractmethod
    def get_sample(self, baseline_value):
        pass
  
    def _get_size(self, baseline_value):
        size = 1
        if type(baseline_value) is np.ndarray:
            size = baseline_value.shape
        return size
    

class SampleDistributions(SampleDistributionBase):
    def __init__(self, distribution_string, val_0, val_1):
        self.distribution = getattr(np.random, distribution_string)
        self.val_0 = val_0
        self.val_1 = val_1
        pass

    def get_sample(self, baseline_value):
        return self.distribution(self.val_0, self.val_1, size=self._get_size(baseline_value)).squeeze()
    
class SampleRelativeDistributions(SampleDistributionBase):
    def __init__(self, distribution_string, val_0, val_1):
        self.distribution = getattr(np.random, distribution_string)
        self.val_0 = val_0
        self.val_1 = val_1
        pass

    def get_sample(self, baseline_value):
        return baseline_value * self.distribution(self.val_0, self.val_1, size=self._get_size(baseline_value)).squeeze()

def setup_sample_dict(distribution_dict):
    sample_dict = dict.fromkeys(distribution_dict.keys())

    for key_i in sample_dict.keys():
        distribution_name = distribution_dict[key_i]["name"]
        distribution_args = distribution_dict[key_i]["args"]
        distribution_is_relative = distribution_dict[key_i]["fraction"]
        if distribution_is_relative:
            sample_dict[key_i] = SampleRelativeDistributions(distribution_name, *distribution_args)
        else:
            sample_dict[key_i] = SampleDistributions(distribution_name, *distribution_args)
    
    return sample_dict
