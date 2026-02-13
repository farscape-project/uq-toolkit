"""
Classes for reading various data types e.g. exodus, vtk
"""
from fnmatch import fnmatch
import meshio
import numpy as np
from warnings import warn

# Source - https://stackoverflow.com/a/56944220
# Posted by Shadow
# Retrieved 2026-02-13, License - CC BY-SA 4.0

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator

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
        """
        Helper class for reading datasets of Exodus files output from MOOSE, using meshio.

        Parameters
        ----------
        fieldname : str
            Name of field in Exodus file to read e.g. "T". Usually corresponds a [Variable] or [AuxVariable]
            in the MOOSE input files.
        nozero : bool
            Skip the first timestep, which is usually a uniform initial condition.
        to_array : bool
            In the `ExodusReader.read_all_samples` API, output all fields as a stacked array.
        to_dict : bool
            In the `ExodusReader.read_all_samples` API, output all fields as a dictionary where the key names 
            are e.g. "sample0", "sample1",...
        block_name: str
            Name of block in Exodus file to search for, and return pointdata for that specific block.
        """
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
    
    def _sort_steps(self, timestep_dict):
        """
        Our meshio branch will convert time-varying exodus data into a dictionary of separate fields
        named as e.g. "{fieldname}_time{TIME_ID}"
        Here we order the keys in that dictionary and save them into a list which can be looped over.
        """
        time_keys = [step_name for step_name in timestep_dict.keys()]
        time_keys.sort(key=lambda x: int(x.split("_time")[1]))
        out_data = [timestep_dict[k] for k in time_keys]
        return out_data

    def _read_cell_data(self, mesh):
        """
        Access cell_data attribute from the meshio mesh and filters the time-varying output
        data to the specific block of interest.
        Also writes the cell-centroids to `self.points` (taken as the mean of all cell vertices).

        Parameters
        ----------
        mesh : meshio.Mesh
            Time-varying exodus field data.
        
        Returns
        -------
        field_data_all_t : dict
            field data for each timestep, sorted into a dictionary.
        """
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
        """
        Access point_data attribute from the meshio mesh and filters the time-varying output
        data to the specific block of interest.
        Also writes the points to `self.points`, and `self.points_in_block` which filters the 
        read data to the selected block, but is also used elsewhere for writing data to only a specific block.

        Parameters
        ----------
        mesh : meshio.Mesh
            Time-varying exodus field data.
        
        Returns
        -------
        field_data_all_t : dict
            field data for each timestep, sorted into a dictionary.
        """
        glob_dict = GlobDict(mesh.point_data)
        # Dict may be un-ordered
        field_data_all_t = glob_dict.glob_to_dict(f"{self.fieldname}_time*", self.nozero)
        # filter to chosen block
        for b, block in enumerate(mesh.cells):
            # assumes tag is list containing only block name
            if block.tags[0] == self.block_name:
                block_ind = b

        # self.cells = [(mesh.cells[block_ind].type, mesh.cells[block_ind].data - mesh.cells[block_ind].data.min())]
        self.points_in_block = np.unique(mesh.cells[block_ind].data)
        self.points = mesh.points[self.points_in_block]
        for key_i, data_i in field_data_all_t.items():
            field_data_all_t[key_i] = data_i[self.points_in_block]
        assert len(field_data_all_t) > 0
        return field_data_all_t

    def read_fname(self, fname, return_mesh=False, all_steps=False, reorder_fname=None):
        """
        Read an exodus file containing field data.

        Parameters
        ----------
        fname : str
            Path to the exodus file (may contain a wildcard)
        return_mesh : bool
            Also return the meshio.Mesh object to the user (useful for getting connectivity etc for viz).
        all_steps : bool
            Return all timesteps in the output file (or only the final step).
        reorder_fname : str | None
            Path to a file name which contains point ordering which can be used to map the points in `fname`
            to match a reference point order (i.e. for a baseline mesh). Needed for dealing with pipelines
            that include remeshing.
        
        Returns
        -------
        field_data : list
            List of numpy arrays for point or cell data for time-varying field, sorted in order of time.
        mesh : meshio.Mesh
            Optional access to the actual meshio object used to extract fields.
        """
        # check for wildcard
        if "*" in fname:
            fname_glob = glob(fname)
            if len(fname_glob) > 1:
                warn(f"multiple files found {fname_glob}")
            elif len(fname_glob) == 0:
                raise IndexError(f"{fname_template} not matching")
            fname = fname_glob[0]
        mesh = meshio.read(fname)
        # try except for point-data or cell-data
        try:
            field_data_all_t = self._read_cell_data(mesh)
        except:
            field_data_all_t = self._read_point_data(mesh)
        # now we sort the list order
        field_data = self._sort_steps(field_data_all_t)
        self.num_mesh_points = len(field_data[-1])

        if all_steps:
            pass
        else:
            field_data = [field_data[-1]]
        self.num_samples += 1
        self.num_steps[fname] = len(field_data)
        
        # user may wish to reorder points to a consistent ordering
        if reorder_fname is not None:
            ordering_arr = np.loadtxt(reorder_fname)[:,1].astype(int)
            field_data = [f[ordering_arr] for f in field_data]

        if return_mesh:
            return field_data, mesh
        else:
            return field_data
        
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

        # assumes self.points_to_show set in _read_pointdata
        field_to_write[self.points_in_block] = field
        mesh.point_data[name] = field_to_write
        return mesh

    def write_celldata(self, mesh, field, name):
        raise NotImplementedError

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

    def read_all_samples(self, fname_list, dict_keys=None, all_steps=True):
        """

        Returns
        -------
        out_data : array_like or list
            if possible, returns array of shape [n_sample, n_step, n_meshpoint]
            otherwise returns nested list of same ordering
        """
        out_data = []
        for i, fname in tqdm(enumerate(fname_list), desc="ExodusReader.read_all_samples"):
            out_data.append(self.read_fname(fname, return_mesh=False, all_steps=all_steps))
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
    """
    Writing to Exodus data is difficult, so instead we can write timeseries point data
    to an xdmf file as recommended in meshio's README.

    Parameters
    ----------
    mesh : meshio.Mesh
        Object containing connectivity, cells, points etc.
    times : iterable
        List or numpy array containing time values to loop over.
    field_val_list : iterable
        List of field values as arrays.
    field_name_list : iterable
        Strings for each field written e.g. ["T", "B"]
    fname: str
        Output name for xdmf file
    """
    with meshio.xdmf.TimeSeriesWriter(fname) as writer:
        writer.write_points_cells(mesh.points, mesh.cells)
        # for t, vals_i_t in enumerate(field_val_i):
        for time_index, time_val in enumerate(times):
            point_data_t = {}
            for (field_val_i, field_name_i) in zip(field_val_list, field_name_list):
                point_data_t[field_name_i] = field_val_i[time_index]
            writer.write_data(time_val, point_data=point_data_t)
