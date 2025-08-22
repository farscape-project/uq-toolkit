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
from tqdm import tqdm
from uqtoolkit import ExodusReader, Reconstructor, SurrogateCLI

def get_inputs():
    parser = SurrogateCLI(description=__doc__)
    parser.add_argument(
        "--reorder",
        default=False,
        action="store_true",
        help="For meshes with same topology but inconsistent ordering, apply reorder data with previously computed hashmaps",
    )
    parser.add_argument(
        "--val",
        default=False,
        action="store_true",
        help="Run validation instead of training",
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
        "--nozero",
        default=False,
        action="store_true",
        help="ignore 0th file",
    )
    parser.add_argument(
        "--steady-state",
        default=False,
        action="store_true",
        help="consider final timestep only",
    )
    return parser.parse_args()

def name_template_exodus(basedir, sample, exodus_name):
    """
    Returns string needed to read VTK data used for training
    """
    return f"{basedir}/{sample}/{exodus_name}"

def read_moose_csv(fname, nozero=True):
    if "*" in fname:
        try:
            fname = glob(fname)[0]
        except IndexError:
            raise IndexError(f"could not read {fname}")
            
    csv_arr = np.loadtxt(
        fname, delimiter=",", usecols=[0], skiprows=1
    )
    if nozero:
        csv_arr = csv_arr[1:]
    return csv_arr

def read_data(
    sample_names, 
    exodus_name,
    fieldname="temperature", 
    csvname="*_out.csv", 
    basedir=".", 
    nozero=True,
    steady_state=True,
    reorder=False
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
    exodus_name : string
        string with name of exodus file to read from each directory.
    csv_name : string
        string with name of csv file to read from each directory (needed for time-dependent stuff).
    fieldname : string
        string to read values from in vedo/vtk object
    basedir : string
        string with /path/to/samples
    nozero : bool
        skip initial timestep
    steady_state : bool
        Only consider last timestep, and ignore time_dict
    reorder : bool
        Apply reordering to field data.

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
    ordering_per_sample = []
    for s, sample in tqdm(enumerate(sample_names)):
        field_snapshot_dict[sample] = []
        time_name = f"{basedir}/{sample}/{csvname}"
        if not steady_state:
            time_dict[sample] = read_moose_csv(time_name, nozero)
        fname_per_sample.append(name_template_exodus(basedir, sample, exodus_name))
        if reorder:
            ordering_per_sample.append(np.loadtxt(f"reorder_mapping/pointsmapping_{sample}.txt")[:,1].astype(int))

    ex_reader = ExodusReader(fieldname, nozero=nozero, to_array=True, to_dict=True)
    dataset = ex_reader.read_all_samples(fname_per_sample, sample_names, all_steps=(not steady_state))
    field_snapshot_dict = ex_reader.out_dict

    if type(dataset) is list:
        dataset = np.concatenate(dataset, axis=0)
    else:
        dataset = np.array(dataset).reshape(-1, ex_reader.num_mesh_points)
    print("dataset shape is:", dataset.shape)

    if reorder:
        ordering_per_sample = np.array(ordering_per_sample)
        dataset, field_snapshot_dict = reorder_all_data(dataset, field_snapshot_dict, ordering_per_sample)
    return (
        dataset,
        field_snapshot_dict,
        time_dict,
    )

def reorder_all_data(dataset_arr, dataset_dict, ordering_per_sample):
    dataset_arr = np.array(list(map(lambda x, y: y[x], ordering_per_sample, dataset_arr)))
    for i, key_i in enumerate(dataset_dict.keys()):
        dataset_dict[key_i] = [a[ordering_per_sample[i]] for a in dataset_dict[key_i]]
        
    return dataset_arr, dataset_dict

def setup_pod_model(dataset, num_modes):
    """
    Use pyssam to setup statistical/POD model

    Parameters
    ----------
    dataset : array_like
        2D array of data to model, where each row is one sample/timestep,
        and each column is a value from the field(s) sampled
    num_modes : int
        How many modes to print reduced cumulative variance.

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

    print("cumsum [full]:", np.cumsum(sam_obj.pca_object.explained_variance_ratio_))
    print("cumsum [reduced]:", np.cumsum(sam_obj.pca_object.explained_variance_ratio_)[:num_modes])
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
    num_modes,
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
    num_modes : int
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
        print("computing coeffs", sample)
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
                num_modes=num_modes,
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

    if args.val:
        with open(f"{args.path_to_samples}/complete_samples_val.txt", "r") as f:
            file_lines = f.read()
            sample_names = file_lines.strip().split("\n")

        sample_names = sample_names[:args.num_samples]
    else:
        with open(f"{args.path_to_samples}/complete_samples.txt", "r") as f:
            file_lines = f.read()
            sample_names = file_lines.strip().split("\n")

        sample_names = sample_names[:args.num_samples]

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
        nozero=args.nozero,
        steady_state=args.steady_state,
        reorder=args.reorder,
    )

    # compute POD weights
    if not args.val:
        (
            mean_dataset_columnvector,
            pca_model_components,
            sam_obj,
        ) = setup_pod_model(dataset, args.num_modes)
        # save POD weights
        np.savez(
            f"{args.pod_dir}/pod_weights_truncated.npz",
            mean=mean_dataset_columnvector.astype("float32"),
            pca_components=pca_model_components.astype("float32")[:args.num_modes],
            pca_std=sam_obj.std.astype("float32")[:args.num_modes],
            cumsum=sam_obj.pca_object.explained_variance_ratio_.astype("float32")[:args.num_modes]
        )
        np.savez(
            f"{args.pod_dir}/pod_weights_full.npz",
            mean=mean_dataset_columnvector.astype("float32"),
            pca_components=pca_model_components.astype("float32"),
            pca_std=sam_obj.std.astype("float32"),
            cumsum=sam_obj.pca_object.explained_variance_ratio_.astype("float32")
        )
    else:
        recon = Reconstructor(model_type=None, pod_coefs_fname=f"{args.pod_dir}/pod_weights_truncated.npz")
        mean_dataset_columnvector = recon.mean_dataset_columnvector
        pca_model_components = recon.pca_model_components
        sam_obj = recon.sam_obj
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
        header = f"{pod_coef_str}"
        data_to_write = dataset_coefs_pertime[sample_i]
        if not args.steady_state:
            data_to_write = np.c_[
                time_dict[sample_i],
                data_to_write
            ]
            header = "t " + header

        np.savetxt(f"{args.pod_dir}/pod_coefs_{sample_i}.txt", data_to_write, header=header)
