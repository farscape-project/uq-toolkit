import argparse
from glob import glob
import numpy as np
import vedo as v
from copy import copy
import pyssam
import matplotlib.pyplot as plt
from warnings import warn
from os import makedirs
from time import time


def get_inputs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path-to-samples",
        default=".",
        type=str,
        help="/path/to/samples",
    )
    parser.add_argument(
        "--fieldname",
        default="temperature",
        type=str,
        help="string for field name to lookup in vtk file",
    )
    parser.add_argument(
        "--csvname",
        default="*_out.csv",
        type=str,
        help="string for csv file to get timesteps",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=100000000,
        type=int,
        help="int containing num samples to read",
    )
    parser.add_argument(
        "--num-modes",
        "-m",
        default=4,
        type=int,
        help="int containing num samples to read",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="show debug checks",
    )
    parser.add_argument(
        "--nozero",
        default=False,
        action="store_true",
        help="ignore 0th file",
    )
    return parser.parse_args()

def name_template(basedir, sample, time, block):
    """
    Returns string needed to read VTK data used for training
    """
    return f"{basedir}/{sample}/vtk_data/{sample}/*/*_{time}_{block}_0.vtu"

def read_moose_csv(fname, nozero=True):
    if "*" in fname:
        fname = glob(fname)[0]
    csv_arr = np.loadtxt(
        fname, delimiter=",", usecols=[0], skiprows=1
    )
    warn("hard-coding skipping first row")
    csv_arr = csv_arr[1:]
    if nozero:
        csv_arr = csv_arr[1:]
    return csv_arr

def read_data(
    sample_names, 
    FIELDNAME="temperature", 
    csvname="*_out.csv", 
    basedir=".", 
    nozero=True
):
    """
    Loop over all sample names and read vtk files from each timestep
    and each block.
    Save all data to a np.ndarray, as well as a separate dict entry
    for time-evolution of some field (e.g. temperature)

    Parameters
    ----------
    sample_names : list
        list of strings containing sample names (corresponding to folders)
        containing data
    FIELDNAME : string
        string to read values from in vedo/vtk object

    Returns
    -------
    dataset : array_like
        2D array of data to model, where each row is one sample/timestep,
        and each column is a value from the field(s) sampled
    num_pts_list : list
        list of ints representing size of each block, used for indexing
        when all blocks combined to one array
    mesh_base : vedo.Mesh
        vedo mesh object containing fields to model in pointdata["FIELDNAME"]
        also contains connectivity etc.
    field_snapshot_dict : dict
        contains a dict entry for each sample. Each dict entry is a np.ndarray
        with time vs pointdata values
    block_id_list : list
        list of ints with IDs for each block read
    """
    field_snapshot_dict = dict.fromkeys(sample_names)
    dataset = []
    time_dict = dict.fromkeys(sample_names)

    num_steps = 0 # all sims should have same number of steps at this point
    for s, sample in enumerate(sample_names):
        fname_list = glob(name_template(basedir, sample, "*", "*"))
        time_id_list = np.unique(
            [int(f.split("_")[-3]) for f in fname_list]
        ).tolist()
        block_id_list = np.unique(
            [int(f.split("_")[-2]) for f in fname_list]
        ).tolist()
        time_id_list.sort()
        if nozero:
            time_id_list.pop(0)
        if s == 0:
            num_steps = len(time_id_list)
        else:
            assert len(time_id_list) == num_steps, f"expected {num_steps}, found {len(time_id_list)} for sample {sample}"
        block_id_list.sort()
        fname_list = sorted(
            fname_list, key=lambda x: int(x.split("/")[-2].split("_")[-1])
        )
        # remove first timestep (initial cond), since variance = 0
        fname_list = fname_list[1:]
        field_snapshot_dict[sample] = []
        time_name = f"{basedir}/{sample}/{csvname}"
        time_dict[sample] = read_moose_csv(time_name, nozero)
        print("loading", sample)

        for i, time_t in enumerate(time_id_list):
            t0 = time()
            field_data_t = []
            mesh_list = []
            for block in block_id_list:
                fname_template = name_template(basedir, sample, time_t, block)
                fname_list_block = glob(fname_template)
                if len(fname_list_block) > 1:
                    warn(f"multiple files found {fname_list_block}")
                elif len(fname_list_block) == 0:
                    raise IndexError(f"{fname_template} not matching")
                fname = fname_list_block[0]
                mesh = v.load(fname)
                field_data = mesh.pointdata[FIELDNAME]
                field_data_t.append(field_data)
                del field_data, mesh
                if DEBUG:
                    print("loading", fname, f"shape = {temp.shape}")
            # reshape so that all points in all blocks are stacked
            field_data_t = np.array(field_data_t).reshape(-1)
            field_snapshot_dict[sample].append(field_data_t)
            dataset.append(field_data_t)
        field_snapshot_dict[sample] = np.array(field_snapshot_dict[sample])
    # just take the last instance of `mesh`
    # get mesh base
    mesh_base = []
    for block in block_id_list:
        time_t = time_id_list[0]
        fname_template = name_template(basedir, sample, time_t, block)
        fname_list_block = glob(fname_template)
        if len(fname_list_block) > 1:
            warn(f"multiple files found {fname_list_block}")
        elif len(fname_list_block) == 0:
            raise IndexError(f"{fname_template} not matching")
        fname = fname_list_block[0]
        mesh_base.append(v.load(fname))

    dataset = np.array(dataset)
    return (
        dataset,
        mesh_base,
        field_snapshot_dict,
        block_id_list,
        time_dict,
    )


def setup_pod_model(dataset):
    """
    Use pyssam to setup statistical/POD model

    Parameters
    ----------
    dataset : array_like
        2D array of data to model, where each row is one sample/timestep,
        and each column is a value from the field(s) sampled

    Returns
    -------
    mean_dataset_columnvector : array_like
        mean shape of the training data in a 1D array.
    pca_model_components : array_like
        eigenvectors of covariance matrix, obtain by PCA.
    sam_obj : pyssam.SAM object
        contains additional object information on the dataset
    """
    sam_obj = pyssam.SAM(dataset)
    sam_obj.create_pca_model(dataset)
    mean_dataset_columnvector = dataset.mean(axis=0)
    pca_model_components = sam_obj.pca_model_components

    print("components shape:", pca_model_components.shape)

    print("cumsum:", np.cumsum(sam_obj.pca_object.explained_variance_ratio_))
    return (
        mean_dataset_columnvector,
        pca_model_components,
        sam_obj,
    )


def get_dataset_coefs(
    sample_names,
    field_snapshot_dict,
    sam_obj,
    mean_dataset_columnvector,
    pca_model_components,
    NUM_MODES,
):
    """
    Find model parameters which best match each input sample with the POD model.
    Effectively rearranges and solves the following for coef (also known as 'b'):
        input_sample = dataset_mean + model_components * model_std * coef

    Parameters
    ----------
    sample_names : list
        list of strings containing sample names (corresponding to folders)
        containing data
    field_snapshot_dict : dict
        contains a dict entry for each sample. Each dict entry is a np.ndarray
        with time vs pointdata values
    sam_obj : pyssam.SAM object
        contains additional object information on the dataset
    mean_dataset_columnvector : array_like
        mean shape of the training data in a 1D array.
    pca_model_components : array_like
        eigenvectors of covariance matrix, obtain by PCA.
    NUM_MODES : int
        number of modes to use in POD expansion.

    Returns
    -------
    dataset_coefs_pertime : dict
        dict for each sample. Each entry contains coefficients used to scale
        each mode
    """
    t1 = time()
    dataset_coefs_pertime = dict.fromkeys(sample_names)
    dataset_mean_pertime = dict.fromkeys(sample_names)
    dataset_scale_pertime = dict.fromkeys(sample_names)
    for sample in sample_names:
        dataset_mean_pertime[sample] = []
        dataset_scale_pertime[sample] = []
        dataset_coefs_pertime[sample] = []
        for i, snapshot_i in enumerate(field_snapshot_dict[sample]):
            dataset_mean_pertime[sample].append(snapshot_i.mean())
            dataset_scale_pertime[sample].append(snapshot_i.std())

            # compute model parameters that correspond to the current snapshot
            params_i = sam_obj.fit_model_parameters(
                snapshot_i - mean_dataset_columnvector,
                pca_model_components,
                num_modes=NUM_MODES,
            )
            dataset_coefs_pertime[sample].append(params_i)
        # convert list to numpy array (faster to append to list than np array)
        dataset_coefs_pertime[sample] = np.array(
            dataset_coefs_pertime[sample]
        )
    t2 = time()
    print(f"fitting parameters, time taken {t2-t1}")

    return dataset_coefs_pertime

def get_list_of_time_data(data_dict, col=0):
    """
    Helps rearrange data for post-processing

    Parameters
    ----------
    data_dict : dict
        dictionary with keys for each sample. Contained is an array or list for each timestep

    Returns
    -------
        data_list : list of lists
    """
    data_list = []
    for sample_i in data_dict.keys():
        sample_data = data_dict[sample_i]
        if sample_data.ndim == 2:
            sample_data = sample_data[:, col]
        n_steps = len(sample_data)
        break

    # create empty list of lists. Ordering is: list[time][samples]
    data_list = [None] * n_steps
    for t in range(n_steps):
        data_list[t] = [None] * len(data_dict.keys())
    for i, sample_i in enumerate(data_dict.keys()):
        sample_data = data_dict[sample_i]
        if sample_data.ndim == 2:
            sample_data = sample_data[:, col]
        for t in range(n_steps):
            data_list[t][i] = copy(sample_data[t])

    return data_list


if __name__ == "__main__":
    args = get_inputs()

    DEBUG = False
    RESULTS_DIR = "results/"
    makedirs(RESULTS_DIR, exist_ok=True)

    POD_DIR = "pod_data/"
    makedirs(POD_DIR, exist_ok=True)

    sample_names = glob(f"{args.path_to_samples}/sample*")
    sample_names = [s.split("/")[-1] for s in sample_names[: args.num_samples]]
    NUM_MODES = args.num_modes

    # read all vtk files. We have several samples, each sample has several timesteps
    # and each timestep may have several blocks (each with its own vtk file)
    # dataset is a np.ndarray of all data
    # mesh_base is a vedo/vtk object containing mesh data
    # field_snapshot_dict has a dict entry for each sample. Each entry contains np.ndarray of time vs field data
    # block_id_list is a list of ints, with the IDs for each VTK block in the mesh
    (
        dataset,
        mesh_base,
        field_snapshot_dict,
        block_id_list,
        time_dict,
    ) = read_data(
        sample_names, 
        FIELDNAME=args.fieldname, 
        csvname=args.csvname, 
        basedir=args.path_to_samples,
        nozero=args.nozero
    )

    # compute POD weights
    (
        mean_dataset_columnvector,
        pca_model_components,
        sam_obj,
    ) = setup_pod_model(dataset)
    # save POD weights
    np.savez(
        f"{POD_DIR}/pod_weights.npz",
        mean=mean_dataset_columnvector,
        pca_components=pca_model_components,
        pca_std=sam_obj.std,
    )
    # compute and save POD coefficients
    # dataset_coefs_pertime is list of arrays for each sample. 
    #   Array shape is (n_time, n_coefs)
    dataset_coefs_pertime = get_dataset_coefs(
        sample_names,
        field_snapshot_dict,
        sam_obj,
        mean_dataset_columnvector,
        pca_model_components,
        args.num_modes,
    )
    # write dataset_coefs_pertime. These are used later as target for regression model
    for sample_i in sample_names:
        # format header for columns in txt file
        pod_coef_str = [f"c{c}" for c in range(args.num_modes)]
        pod_coef_str = " ".join(pod_coef_str)
        np.savetxt(
            f"{POD_DIR}/pod_coefs_{sample_i}.txt", 
            np.c_[
                time_dict[sample_i],
                dataset_coefs_pertime[sample_i]
            ],
            header=f"t {pod_coef_str}"
        )
