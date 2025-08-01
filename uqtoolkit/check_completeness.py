"""
Loop over --exodus-name and --csvname in sample dirs to check csv and exodus files exist.
"""
import argparse
from glob import glob
import numpy as np
from warnings import warn
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
        default="*_out.e",
        type=str,
        help="string for exodus file",
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
    """
    field_snapshot_dict = dict.fromkeys(sample_names)
    time_dict = dict.fromkeys(sample_names)

    missing_sample_index_list = []
    for s, sample in enumerate(sample_names):
        print("loading csv", sample)
        field_snapshot_dict[sample] = []
        time_name = f"{basedir}/{sample}/{csvname}"
        try:
            time_dict[sample] = read_moose_csv(time_name, nozero)
        except FileNotFoundError:
            warn(f"{sample} has missing file {time_name}")
            missing_sample_index_list.append(s)
            continue
    
    # # remove missing data from list
    for del_ind in missing_sample_index_list[::-1]:
        sample_names.pop(del_ind)

    ex_reader = ExodusReader(fieldname, nozero=nozero, to_array=True, to_dict=True)
    for s, sample in enumerate(sample_names):
        print("loading exodus", sample)
        exodus_fname = name_template_exodus(basedir, sample, exodus_name)
        ex_reader.read_fname(exodus_fname, return_mesh=False)

    # write list of complete samples
    with open(f"{basedir}/complete_samples.txt", "w") as f:
        [f.write(f"{s}\n") for s in sample_names]


if __name__ == "__main__":
    args = get_inputs()

    path_to_search = f"{args.path_to_samples}/sample*"
    sample_names = glob(path_to_search)
    assert len(sample_names) > 0, f"no samples in path {path_to_search}"
    sample_names = [s.split("/")[-1] for s in sample_names]
    sample_names.sort(key=lambda x: int(x.replace("sample", "")))

    # read all exodus files.
    # dataset is a np.ndarray of all data
    # field_snapshot_dict has a dict entry for each sample. Each entry contains np.ndarray of time vs field data
    read_data(
        sample_names, 
        exodus_name=args.exodus_name,
        fieldname=args.fieldname, 
        csvname=args.csvname, 
        basedir=args.path_to_samples,
    )
