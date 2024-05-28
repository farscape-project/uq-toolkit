from abc import ABC, abstractmethod
import numpy as np
from warnings import warn
from scipy.stats import qmc
from UQpy import distributions, sampling

class SampleDistributionBase(ABC):

    @abstractmethod
    def get_sample(self, baseline_value):
        pass
  
    def _get_size(self, baseline_value):
        size = 1
        if type(baseline_value) is np.ndarray:
            size = baseline_value.shape
        return size
    
    def _random_sample(self, baseline_value):
        return self.distribution(self.val_0, self.val_1, size=self._get_size(baseline_value)).squeeze()


class SampleDistributions(SampleDistributionBase):
    def __init__(self, distribution_string, val_0, val_1):
        self.distribution = getattr(np.random, distribution_string)
        self.val_0 = val_0
        self.val_1 = val_1
        pass

    def get_sample(self, baseline_value):
        return self._random_sample(baseline_value)
    
class SampleRelativeDistributions(SampleDistributionBase):
    def __init__(self, distribution_string, val_0, val_1):
        self.distribution = getattr(np.random, distribution_string)
        if distribution_string == "normal" and val_0 == 0:
            warn("This will multiply baseline value by normal distribution with 0 mean. Do you mean mean = 1?")
        self.val_0 = val_0
        self.val_1 = val_1
        pass

    def get_sample(self, baseline_value):
        return baseline_value * self._random_sample(baseline_value)
    
class DatasetSampler:
    def __init__(self, data):
        self.data = data.squeeze()
        self.sampled_row = 0
        pass

    def get_sample(self, baseline_value):
        data_out = self.data[self.sampled_row]
        self.sampled_row += 1
        return data_out

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


def setup_spacefilling_sampler(main_param_dict, main_distribution_dict, n_samples):

    upper_bound = []
    lower_bound = []
    num_features = 0
    for app_i in main_distribution_dict.keys():
        distribution_dict = main_distribution_dict[app_i]
        param_dict = main_param_dict[app_i]
        # get distribution for uncertain param in each app. 
        # Assumes uniform distribution. Get lower and upper bounds for scaling LHS output
        for key_i in distribution_dict.keys():
            l_bound_frac = distribution_dict[key_i]["lower"]
            u_bound_frac = distribution_dict[key_i]["upper"]
            distribution_is_relative = distribution_dict[key_i]["fraction"]
            """
            distibution_is_relative, when upper and lower bounds are a fraction of the value
            obtained in the input file 
            e.g. lower = 0.9, upper = 1.1, param value is 100, distribution is U(90, 110).
            """
            param_value_orig = param_dict[key_i]
            if distribution_is_relative:
                l_bound = param_value_orig + (l_bound_frac-1)*abs(param_value_orig)
                u_bound = param_value_orig + (u_bound_frac-1)*abs(param_value_orig)
                if type(l_bound) == float:
                    l_bound = [l_bound]
                    u_bound = [u_bound]

                lower_bound.extend(list(l_bound))
                upper_bound.extend(list(u_bound))
            else:
                # raise NotImplementedError
                lower_bound.append(distribution_dict[key_i]["lower"])
                upper_bound.append(distribution_dict[key_i]["upper"])
                assert type(param_value_orig) == float
            # store number of features, for setting up LHC sampler
            if type(param_value_orig) == float:
                num_features += 1
            else:
                num_features += param_value_orig.size

    # setup and draw samples from LHC
    sampler = qmc.LatinHypercube(num_features)
    sample = sampler.random(n_samples)
    sample_scaled = qmc.scale(sample, lower_bound, upper_bound)

    # now we must map the sampled data compute above to a class
    col_ind = 0
    sample_dict = dict.fromkeys(main_distribution_dict.keys())
    for app_i in main_distribution_dict.keys():
        distribution_dict = main_distribution_dict[app_i]
        param_dict = main_param_dict[app_i]
        sample_dict[app_i] = dict.fromkeys(distribution_dict.keys())
        for key_i in distribution_dict.keys():
            # find which column in sample_scaled corresponds to right dict entry
            var_size = 1
            if type(param_dict[key_i]) is np.ndarray:
                var_size = param_dict[key_i].size
            sample_scaled_column = sample_scaled[:, col_ind:col_ind+var_size]
            # next iteration, we sample the next column(s)
            col_ind += var_size

            # setup obj for sampling
            sample_dict[app_i][key_i] = DatasetSampler(data=sample_scaled_column)
    return sample_dict

def setup_uqpy_sampler(main_param_dict, main_distribution_dict, n_samples, sampler_string="latinhypercube"):

    upper_bound = []
    lower_bound = []
    num_features = 0
    distribution_list = []
    param_list = []
    for app_i in main_distribution_dict.keys():
        distribution_dict = main_distribution_dict[app_i]
        param_dict = main_param_dict[app_i]
        # get distribution for uncertain param in each app. 
        # Assumes uniform distribution. Get lower and upper bounds for scaling LHS output
        for key_i in distribution_dict.keys():
            distribution_is_relative = distribution_dict[key_i]["fraction"]
            # see scipy.stats docs for description of loc and scale
            # in scipy.stats.uniform, we have U[loc, loc+scale]
            loc = distribution_dict[key_i]["loc"] 
            scale = distribution_dict[key_i]["scale"]
            """
            distibution_is_relative, when upper and lower bounds are a fraction of the value
            obtained in the input file 
            e.g. lower = 0.9, upper = 1.1, param value is 100, distribution is U(90, 110).
            """
            param_value_orig = param_dict[key_i]
            if distribution_is_relative:
                loc = param_value_orig + (loc-1)*abs(param_value_orig)
                scale = scale*abs(param_value_orig)

            if hasattr(param_value_orig, '__iter__'):
                assert np.all(scale>0), (
                    f"{scale}, {distribution_dict[key_i]['scale']}, orig val = {param_value_orig}\n"
                    f"{loc}, {distribution_dict[key_i]['loc']}"
                )
                for loc_i, scale_i in zip(loc, scale):
                    distribution_list.append(distributions.Uniform(loc_i, scale_i))
            else:
                distribution_list.append(distributions.Uniform(loc, scale))
            # store number of features, for setting up LHC sampler
            if type(param_value_orig) == float:
                num_features += 1
            else:
                num_features += param_value_orig.size

    # setup and draw samples from LHC
    if sampler_string.lower() == "latinhypercube":
        sampler = sampling.LatinHypercubeSampling
    elif sampler_string.lower() == "montecarlo":
        sampler = sampling.MonteCarloSampling
    else:
        raise NotImplementedError(
            f"sampler_string = {sampler_string} not supported, options are 'latinhypercube' and 'montecarlo'"
        )
    sample_scaled = sampler(distribution_list, n_samples).samples

    # now we must map the sampled data compute above to a class
    col_ind = 0
    sample_dict = dict.fromkeys(main_distribution_dict.keys())
    for app_i in main_distribution_dict.keys():
        distribution_dict = main_distribution_dict[app_i]
        param_dict = main_param_dict[app_i]
        sample_dict[app_i] = dict.fromkeys(distribution_dict.keys())
        for key_i in distribution_dict.keys():
            # find which column in sample_scaled corresponds to right dict entry
            var_size = 1
            if type(param_dict[key_i]) is np.ndarray:
                var_size = param_dict[key_i].size
            sample_scaled_column = sample_scaled[:, col_ind:col_ind+var_size]
            # next iteration, we sample the next column(s)
            col_ind += var_size

            # setup obj for sampling
            sample_dict[app_i][key_i] = DatasetSampler(data=sample_scaled_column)
    return sample_dict
