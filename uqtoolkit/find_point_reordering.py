"""
Script used in workflows with inconsistent mesh node ordering, which finds mapping 
between a set of 'template' coordinates and the new 'target' coordinates.
The template is the first sample in complete_samples.txt, target is all other samples in the dataset.

Uses multi-processing to accelerate what would otherwise be a very slow loop. This is compute-intensive.
"""
import argparse
from functools import partial
from multiprocessing import Pool
from os import makedirs

import numpy as np
from tqdm import tqdm

from uqtoolkit.datareader import ExodusReader


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
        "--exodus-name",
        "-ename",
        default="SS316LSTCThermal_out.e",
        type=str,
        help="string for exodus file",
    )
    return parser.parse_args()


def read_data(
    sample_names,
    exodus_name,
    fieldname="temperature",
    basedir=".",
):
    """
    Read mesh points from all exodus files.

    Parameters
    ----------
    sample_names : list
        list of strings containing sample names (corresponding to folders) containing data
    fieldname : string
        string to read values from in vedo/vtk object

    Returns
    -------
    mesh_list : array_like
        3D array of XYZ point coordinates for all meshes in dataset.
    """
    ex_reader = ExodusReader(
        fieldname, nozero=True, to_array=True, to_dict=True, block_name="target"
    )
    mesh_list = []
    for s, sample in enumerate(sample_names):
        print("loading", sample)
        _, mesh = ex_reader.read_fname(
            f"{basedir}/{sample}/{exodus_name}", sample_names
        )
        mesh_list.append(ex_reader.points)
        del mesh
    mesh_list = np.array(mesh_list)
    return mesh_list

def reorder_all_data(sample_names, mesh_list, out_dir):
    """
    Use multiprocessing to find nearest neighbour for all points in mesh based on a template mesh
    (which is the first sample in complete_samples.txt).
    Write the mapping for each sample to `out_dir/pointsmapping_sampleX.txt`.
    
    Parameters
    ----------
    sample_names : list
        list of strings containing sample names (corresponding to folders) containing data
    mesh_list : array_like or list
        XYZ coordinates corresponding to vertices on each mesh. mesh_list[0] is considered the template.
    out_dir : str
        Path to write mappings to.
    """
    pts_template = mesh_list[0]
    for s, sample_i in enumerate(sample_names):
        print("reordering", sample_i)
        reordering_kernel_partial = partial(
            reordering_kernel, target_pt_list=mesh_list[s], pts_template=pts_template
        )
        with Pool() as pool:
            nearest_to_template_mapping = pool.map(
                reordering_kernel_partial, tqdm(range(0, len(pts_template)))
            )
        np.savetxt(f"{out_dir}/pointsmapping_{sample_i}.txt", nearest_to_template_mapping)
    return

def reordering_kernel(template_pt_index, target_pt_list, pts_template):
    """
    Kernel to be iterated over to find closest target mesh point for a given point on the template mesh.

    Parameters
    ----------
    template_pt_index : int
        Index of template point to find nearest target point for
    template_pt_index : array_like
        All points on the target mesh
    pts_template : array_like
        All points on the template mesh

    Returns
    -------
    template_pt_index : int
        Same as input
    nearest : int
        Index of nearest node on the target mesh
    dist : float
        Euclidean distance between each point (can be used for checking that it is 0).
    """
    nearest = np.sqrt(
        np.sum((target_pt_list - pts_template[template_pt_index]) ** 2, axis=-1)
    ).argmin()
    dist = np.sqrt(
        np.sum((target_pt_list[nearest] - pts_template[template_pt_index]) ** 2)
    )
    return [template_pt_index, nearest, dist]


if __name__ == "__main__":
    args = get_inputs()

    RESULTS_DIR = f"{args.path_to_samples}/"
    OUT_DIR = f"{RESULTS_DIR}/reorder_mapping"
    with open(f"{RESULTS_DIR}/complete_samples.txt", "r") as f:
        file_lines = f.read()
        sample_names = file_lines.strip().split("\n")
    makedirs(OUT_DIR, exist_ok=True)

    # read all exodus files.
    # dataset is a np.ndarray of all data
    # field_snapshot_dict has a dict entry for each sample. Each entry contains np.ndarray of time vs field data
    mesh_list = read_data(
        sample_names,
        exodus_name=args.exodus_name,
        fieldname=args.fieldname,
        basedir=args.path_to_samples,
    )

    reorder_all_data(sample_names, mesh_list, OUT_DIR)
