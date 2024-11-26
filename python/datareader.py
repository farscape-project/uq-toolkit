"""
Classes for reading various data types e.g. exodus, vtk
"""
from fnmatch import fnmatch
import meshio
import numpy as np
from warnings import warn

class GlobDict(dict):
    """
    Taken from https://stackoverflow.com/a/3601346
    """
    def glob_to_list(self, match, nozero=True):
        """@match should be a glob style pattern match (e.g. '*.txt')"""
        # return dict([(k,v) for k,v  in self.items() if fnmatch(k, match)])
        if nozero:
            return [v for k,v  in self.items() if (fnmatch(k, match) and not fnmatch(k, "*_time0"))]
        else:
            return [v for k,v  in self.items() if fnmatch(k, match)]

    def glob_to_dict(self, match, nozero=True):
        """@match should be a glob style pattern match (e.g. '*.txt')"""
        # return dict([(k,v) for k,v  in self.items() if fnmatch(k, match)])
        if nozero:
            return dict([(k,v) for k,v  in self.items() if (fnmatch(k, match) and not fnmatch(k, "*_time0"))])
        else:
            return dict([(k,v) for k,v  in self.items() if fnmatch(k, match)])

class ExodusReader:
    def __init__(self, fieldname, nozero=True, to_array=True, to_dict=False):
        self.fieldname = fieldname
        self.nozero = nozero

        self.num_samples = 0
        self.num_mesh_points = -1
        self.num_steps = {}

        self.to_array = to_array
        self.to_dict = to_dict

        self.out_array = None
        self.out_dict = dict()
    
    def sort_steps(self, timestep_dict):
        """
        Assumes dict keys are of the form "fieldname_time{TIME_ID}"
        """
        time_keys = [step_name for step_name in timestep_dict.keys()]
        time_keys.sort(key=lambda x: int(x.split("_time")[1]))
        out_data = [timestep_dict[k] for k in time_keys]
        return out_data

    def read_fname(self, fname, return_mesh=False):
        # check for wildcard
        if "*" in fname:
            fname_glob = glob(fname)
            if len(fname_glob) > 1:
                warn(f"multiple files found {fname_glob}")
            elif len(fname_glob) == 0:
                raise IndexError(f"{fname_template} not matching")
            else:
                fname = fname_glob[0]
        mesh = meshio.read(fname)
        # Dict may be un-ordered
        glob_dict = GlobDict(mesh.point_data)
        field_data_all_t = glob_dict.glob_to_dict(f"{self.fieldname}_time*", self.nozero)
        # now we sort the list order
        field_data_all_t_sorted = self.sort_steps(field_data_all_t)
        self.num_mesh_points = len(field_data_all_t_sorted[-1])
        if return_mesh:
            return field_data_all_t_sorted, mesh
        else:
            return field_data_all_t_sorted

    def read_all_steps(self, fname):
        field_data_all_t = self.read_fname(fname)
        self.num_samples += 1
        self.num_steps[fname] = len(field_data_all_t)
        return field_data_all_t

    def read_final_step(self, fname):
        field_data = self.read_fname(fname)[-1]
        self.num_samples += 1
        self.num_steps[fname] = len(field_data)
        return field_data

    def read_all_samples(self, fname_list, dict_keys=None):
        """

        Returns
        -------
        out_data : array_like or list
            if possible, returns array of shape [n_sample, n_step, n_meshpoint]
            otherwise returns nested list of same ordering
        """
        out_data = []
        for i, fname in enumerate(fname_list):
            out_data.append(self.read_all_steps(fname))
            # add data from this loop iter to new key in dict
            if self.to_dict:
                self.out_dict[dict_keys[i]] = out_data[-1]

        # check if all samples have same number of timesteps
        all_num_steps = np.array(list(self.num_steps.values()))
        same_num_steps = np.all(all_num_steps == all_num_steps[0])
        
        if self.to_array:
            if same_num_steps:
                self.out_array = np.array(out_data)
                return self.out_array
            else:
                warn(
                    f"wanted to output to array, but shape is not consistent: {self.num_steps}"
                )
                return out_data


def write_timeseries(mesh, times, field_val_list, field_name_list, fname):
    with meshio.xdmf.TimeSeriesWriter(fname) as writer:
        writer.write_points_cells(mesh.points, mesh.cells)
        # for t, vals_i_t in enumerate(field_val_i):
        for time_index, time_val in enumerate(times):
            point_data_t = {}
            for (field_val_i, field_name_i) in zip(field_val_list, field_name_list):
                point_data_t[field_name_i] = field_val_i[time_index]
            writer.write_data(time_val, point_data=point_data_t)
