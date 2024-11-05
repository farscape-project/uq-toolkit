import argparse
from glob import glob
import numpy as np
import meshio
from copy import copy
import pyssam
import matplotlib.pyplot as plt
from warnings import warn
from os import makedirs
from time import time
from datareader import ExodusReader

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
        "--exodus-name",
        "-ename",
        default="SS316LSTCThermal_out.e",
        type=str,
        help="string for exodus file",
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

def name_template_vtk(basedir, sample, time, block):
    """
    Returns string needed to read VTK data used for training
    """
    return f"{basedir}/{sample}/vtk_data/{sample}/*/*_{time}_{block}_0.vtu"

def name_template_exodus(basedir, sample, exodus_name):
    """
    Returns string needed to read VTK data used for training
    """
    return f"{basedir}/{sample}/{exodus_name}"

def read_moose_csv(fname, nozero=True):
    if "*" in fname:
        fname = glob(fname)[0]
    csv_arr = np.loadtxt(
        fname, delimiter=",", usecols=[0], skiprows=1
    )
    if nozero:
        csv_arr = csv_arr[1:]
    return csv_arr

def load_all_blocks(fname_args, fieldname, exodus_name, keepmesh=False, nozero=True):
    """
    Loop over all blocks in geometry and stack field data as an array

    Parameters
    ----------
    fname_args : list
        list containing args used to find vtk file name.
        Contents are [basedir, sample, time]

    Returns
    -------
    field_data_t : array_like
        1D array containing field point values from all blocks
    """
    field_data_t = []
    
    fname_template = name_template_exodus(*fname_args, exodus_name)
    fname_list_block = glob(fname_template)
    if len(fname_list_block) > 1:
        warn(f"multiple files found {fname_list_block}")
    elif len(fname_list_block) == 0:
        raise IndexError(f"{fname_template} not matching")
    fname = fname_list_block[0]
    mesh = meshio.read(fname)
    # Dict may be un-ordered, but timestep ordering not important for now(?)
    glob_dict = GlobDict(mesh.point_data)
    field_data = glob_dict.glob(f"{fieldname}*", nozero)
    if keepmesh:
        pass
    else:
        # delete mesh object, and create empty variable
        # means we can keep the same loop structure
        del mesh
        mesh = None
    return field_data, mesh

def read_data(
    sample_names, 
    exodus_name,
    fieldname="temperature", 
    csvname="*_out.csv", 
    basedir=".", 
    nozero=True
):
    """
    Loop over all sample names and read exodus files that contain 
    all timesteps (in seperate fields) and all blocks in stacked array.
    Save all data to a np.ndarray, as well as a separate dict entry
    for time-evolution of some field (e.g. temperature)

    Parameters
    ----------
    sample_names : list
        list of strings containing sample names (corresponding to folders)
        containing data
    fieldname : string
        string to read values from in vedo/vtk object

    Returns
    -------
    dataset : array_like
        2D array of data to model, where each row is one sample/timestep,
        and each column is a value from the field(s) sampled
    num_pts_list : list
        list of ints representing size of each block, used for indexing
        when all blocks combined to one array
    field_snapshot_dict : dict
        contains a dict entry for each sample. Each dict entry is a np.ndarray
        with time vs point_data values
    """
    field_snapshot_dict = dict.fromkeys(sample_names)
    dataset = []
    time_dict = dict.fromkeys(sample_names)

    num_steps = 0 # all sims should have same number of steps at this point
    fname_per_sample = []
    for s, sample in enumerate(sample_names):
        print("loading", sample)
        field_snapshot_dict[sample] = []
        time_name = f"{basedir}/{sample}/{csvname}"
        time_dict[sample] = read_moose_csv(time_name, nozero)
        fname_per_sample.append(name_template_exodus(basedir, sample, exodus_name))

    ex_reader = ExodusReader(fieldname, nozero=nozero, to_array=True, to_dict=True)
    dataset = ex_reader.read_all_samples(fname_per_sample, sample_names)
    print("dataset shape is:", dataset.shape)
    field_snapshot_dict = ex_reader.out_dict

    dataset = np.array(dataset).reshape(-1, ex_reader.num_mesh_points)
    return (
        dataset,
        field_snapshot_dict,
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

if __name__ == "__main__":
    args = get_inputs()

    DEBUG = False
    RESULTS_DIR = "results/"
    makedirs(RESULTS_DIR, exist_ok=True)

    POD_DIR = "pod_data/"
    makedirs(POD_DIR, exist_ok=True)

    path_to_search = f"{args.path_to_samples}/sample*"
    sample_names = glob(path_to_search)
    assert len(sample_names) > 0, f"no samples in path {path_to_search}"
    sample_names = [s.split("/")[-1] for s in sample_names[: args.num_samples]]
    NUM_MODES = args.num_modes

    # read all exodus files.
    # dataset is a np.ndarray of all data
    # field_snapshot_dict has a dict entry for each sample. Each entry contains np.ndarray of time vs field data
    (
        dataset,
        field_snapshot_dict,
        time_dict,
    ) = read_data(
        sample_names, 
        exodus_name=args.exodus_name,
        fieldname=args.fieldname, 
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
