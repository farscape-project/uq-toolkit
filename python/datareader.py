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
    def __init__(self, fieldname, nozero=True, to_array=True, to_dict=False, block_name="target"):
        self.fieldname = fieldname
        self.nozero = nozero
        self.block_name = block_name

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

    def _read_cell_data(self, mesh):
        glob_dict = GlobDict(mesh.cell_data)
        # Dict may be un-ordered
        field_data_all_t = glob_dict.glob_to_dict(f"{self.fieldname}_time*", self.nozero)
        # filter to chosen block
        for b, block in enumerate(mesh.cells):
            # assumes tag is list containing only block name
            if block.tags[0] == self.block_name:
                block_ind = b
        self.points = np.mean(mesh.points[mesh.cells[block_ind].data], axis=1)
        for key_i, data_i in field_data_all_t.items():
            field_data_all_t[key_i] = data_i[block_ind]
        assert len(field_data_all_t) > 0
        return field_data_all_t

    def _read_point_data(self, mesh):
        glob_dict = GlobDict(mesh.point_data)
        # Dict may be un-ordered
        field_data_all_t = glob_dict.glob_to_dict(f"{self.fieldname}_time*", self.nozero)
        # filter to chosen block
        for b, block in enumerate(mesh.cells):
            # assumes tag is list containing only block name
            if block.tags[0] == self.block_name:
                block_ind = b

        # self.cells = [(mesh.cells[block_ind].type, mesh.cells[block_ind].data - mesh.cells[block_ind].data.min())]
        points_to_show = np.unique(mesh.cells[block_ind].data)
        self.points = mesh.points[points_to_show]
        for key_i, data_i in field_data_all_t.items():
            field_data_all_t[key_i] = data_i[points_to_show]
        assert len(field_data_all_t) > 0
        return field_data_all_t

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
        # try except for point-data or cell-data
        try:
            field_data_all_t = self._read_cell_data(mesh)
        except:
            field_data_all_t = self._read_point_data(mesh)
        # now we sort the list order
        field_data_all_t_sorted = self.sort_steps(field_data_all_t)
        self.num_mesh_points = len(field_data_all_t_sorted[-1])
        if return_mesh:
            return field_data_all_t_sorted, mesh
        else:
            return field_data_all_t_sorted
        
    def add_block_id_field(self, mesh):
        """
        When writing to VTK, we add a scalar for block ID to allow 
        filtering to a specific block in paraview.

        Parameters
        ----------
        mesh (meshio.Mesh): Multiblock exodus mesh data

        Returns
        -------
        mesh (meshio.Mesh): Multiblock exodus mesh with additional point_data field "block_id"
        """
        field_to_write = np.zeros(mesh.points.shape[0])
        for b, block in enumerate(mesh.cells):
            # assumes tag is list containing only block name
            points_to_show = np.unique(block.data)
            field_to_write[points_to_show] = b
        
        mesh.point_data["block_id"] = field_to_write
        return mesh

    def write_pointdata(self, mesh, field, name):
        """
        Create a new mesh.point_data field with an input numpy array. 
        The field is zero on all other blocks.

        Parameters
        ----------
        mesh (meshio.Mesh): Multiblock exodus mesh data
        field (np.array): 1D array of scalar data to be represented at corresponding grid points.
        name (str): Name for field to be looked-up in exodus.

        Returns
        -------
        mesh (meshio.Mesh): mesh with new data appended as additional point_data field
        """
        field_to_write = np.zeros(mesh.points.shape[0])
        for b, block in enumerate(mesh.cells):
            # assumes tag is list containing only block name
            if block.tags[0] == self.block_name:
                block_ind = b
        points_to_show = np.unique(mesh.cells[block_ind].data)
        field_to_write[points_to_show] = field
        mesh.point_data[name] = field_to_write
        return mesh

    def write_celldata(self, mesh, field, name):
        """
        Create a new mesh.cell_data field with an input numpy array. 
        The field is zero on all other blocks.

        Parameters
        ----------
        mesh (meshio.Mesh): Multiblock exodus mesh data
        field (np.array): 1D array of scalar data to be represented at corresponding cells.
        name (str): Name for field to be looked-up in exodus.

        Returns
        -------
        mesh (meshio.Mesh): mesh with new data appended as additional cell_data field
        """
        field_to_write = np.zeros(mesh.points.shape[0])
        for b, block in enumerate(mesh.cells):
            # assumes tag is list containing only block name
            if block.tags[0] == self.block_name:
                block_ind = b
        points_to_show = np.unique(mesh.cells[block_ind].data)
        
        field_to_write[points_to_show] = field
        mesh.point_data[name] = field_to_write
        return mesh

    def clean_exodus_data(self, mesh):
        """
        Empty all existing point and cell data from mesh. Useful when many timesteps are saved but not needed (due to excessive memory).

        Parameters
        ----------
        mesh (meshio.Mesh): Multiblock exodus mesh data

        Returns
        -------
        mesh (meshio.Mesh): mesh with point and cell data emptied.
        """
        mesh.point_data = {}
        mesh.cell_data = {}
        return mesh

    def read_all_steps(self, fname):
        field_data_all_t = self.read_fname(fname)
        self.num_samples += 1
        self.num_steps[fname] = len(field_data_all_t)
        return field_data_all_t

    def read_final_step(self, fname):
        # take final value from list, and make a list again (of size 1)
        field_data = [self.read_fname(fname)[-1]]
        self.num_samples += 1
        self.num_steps[fname] = len(field_data)
        return field_data

    def read_all_samples(self, fname_list, dict_keys=None, all_steps=True):
        """

        Returns
        -------
        out_data : array_like or list
            if possible, returns array of shape [n_sample, n_step, n_meshpoint]
            otherwise returns nested list of same ordering
        """
        out_data = []
        for i, fname in enumerate(fname_list):
            if all_steps:
                out_data.append(self.read_all_steps(fname))
            else:
                out_data.append(self.read_final_step(fname))
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
