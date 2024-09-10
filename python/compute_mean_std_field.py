""" Utility for outputting mean and standard deviation of field from vtk
files in "sample*/vtk_data" as a vtk file for each timestep
"""
import argparse
from copy import copy
from glob import glob
from os import makedirs
from time import time
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pyssam
import vedo as v


def get_inputs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field",
        required=True,
        type=str,
        help="name of field",
    )
    parser.add_argument(
        "-o",
        "--outname",
        required=True,
        type=str,
        help="name of file to output (timestep ID will be included automatically)",
    )
    return parser.parse_args()


def write_fields(mesh_base, field_val_list, field_name_list, fname="mesh.vtk"):
    """
    write vtk object with new field.
    """
    mesh_new = mesh_base.clone()
    for field_vals_i, field_name_i in zip(field_val_list, field_name_list):
        mesh_new.pointdata[field_name_i] = field_vals_i
    v.write(mesh_new, fname)


def name_template(sample, time, block):
    """
    Returns string needed to read VTK data used for training
    """
    return f"{sample}/vtk_data/{sample}/*/*_{time}_{block}_0.vtu"


def read_data(sample_names, FIELDNAME="temperature"):
    """
    Loop over all sample names and read vtk files from each timestep
    and each block.
    Save all data to a np.ndarray

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
    """
    dataset = []

    for sample in sample_names:
        fname_list = glob(name_template(sample, "*", "*"))
        time_id_list = np.unique(
            [int(f.split("_")[-3]) for f in fname_list]
        ).tolist()
        block_id_list = np.unique(
            [int(f.split("_")[-2]) for f in fname_list]
        ).tolist()
        time_id_list.sort()
        block_id_list.sort()
        print("loading", sample)

        dataset_t = []
        for i, time_t in enumerate(time_id_list):
            temp_t = []
            mesh_list = []
            for block in block_id_list:
                fname_template = name_template(sample, time_t, block)
                fname_list_block = glob(fname_template)
                if len(fname_list_block) > 1:
                    warn(f"multiple files found {fname_list_block}")
                elif len(fname_list_block) == 0:
                    raise IndexError(f"{fname_template} not matching")
                fname = fname_list_block[0]
                mesh = v.load(fname)
            dataset_t.append(mesh.pointdata[FIELDNAME])
        dataset.append(dataset_t)

    dataset = np.array(dataset)
    return dataset


def get_mesh_properties(sample, FIELDNAME):
    """
    Extract vtk mesh object, block ids, and number of points in each block

    Parameters
    ----------
    sample : string
        sample name to read mesh from (assume all meshes are same)
    FIELDNAME : string
        string to read values from in vedo/vtk object

    Returns
    -------
    mesh_base : vedo.Mesh
        vedo mesh object containing fields to model in pointdata["FIELDNAME"]
        also contains connectivity etc.
    num_pts_list : list
        list of ints representing size of each block, used for indexing
        when all blocks combined to one array
    block_id_list : list
        list of ints with IDs for each block read
    """
    fname_list = glob(name_template(sample, "*", "*"))
    time_id_list = np.unique(
        [int(f.split("_")[-3]) for f in fname_list]
    ).tolist()
    block_id_list = np.unique(
        [int(f.split("_")[-2]) for f in fname_list]
    ).tolist()
    time_id_list.sort()
    block_id_list.sort()

    # get mesh base
    num_pts_list = []
    for block in block_id_list:
        time_t = time_id_list[0]
        # fname_template = f"{sample}/*/*_{time_t}_{block}_0.vtu"
        fname_template = name_template(sample, time_t, block)
        fname_list_block = glob(fname_template)
        if len(fname_list_block) > 1:
            warn(f"multiple files found {fname_list_block}")
        elif len(fname_list_block) == 0:
            raise IndexError(f"{fname_template} not matching")
        fname = fname_list_block[0]
        mesh_base = v.load(fname)
        num_pts_list.append(mesh_base.pointdata[FIELDNAME].size)

    return mesh_base, block_id_list, num_pts_list


if __name__ == "__main__":
    args = get_inputs()

    sample_names = glob("sample*")
    sample_names.sort()

    # read all vtk files. We have several samples, each sample has several timesteps
    # and each timestep may have several blocks (each with its own vtk file)
    # dataset is a np.ndarray of all data, shape is (n_samples, n_steps, n_points)
    dataset = read_data(sample_names, args.field)
    print(dataset.shape)

    # mesh_base is a vedo/vtk object containing mesh data
    # num_pts_list stores to size (int) of each block for reconstructing fields (not needed if only 1 block)
    # block_id_list is a list of ints, with the IDs for each VTK block in the mesh
    (
        mesh_base,
        num_pts_list,
        block_id_list,
    ) = get_mesh_properties(sample_names[0], args.field)

    # find mean and std-dev of all samples
    mean_field = dataset.mean(axis=0)
    stddev_field = dataset.std(axis=0)

    for t, (mean_t, stddev_t) in enumerate(zip(mean_field, stddev_field)):
        print("writing field for time", t)
        if len(block_id_list) > 1:
            raise NotImplementedError
        write_fields(
            mesh_base,
            [mean_t, stddev_t],
            ["mean", "stddev"],
            f"{args.outname}_t{t}.vtk",
        )
