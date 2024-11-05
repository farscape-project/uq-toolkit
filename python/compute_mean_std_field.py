""" Utility for outputting mean and standard deviation of field from exodus
files in "sample*" as a xdmf3 file containing all timesteps
"""
import argparse
from copy import copy
from glob import glob
from os import makedirs
from time import time
from warnings import warn

import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyssam
import vedo as v

from datareader import ExodusReader, write_timeseries

def get_inputs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path-to-samples",
        default=".",
        type=str,
        help="/path/to/samples",
    )
    parser.add_argument(
        "--field",
        default="temperature",
        type=str,
        help="name of field",
    )
    parser.add_argument(
        "--exodus-name",
        "-ename",
        default="SS316LSTCThermal_out.e",
        type=str,
        help="string for exodus file",
    )
    parser.add_argument(
        "-o",
        "--outname",
        required=True,
        type=str,
        help="name of file to output (timestep ID will be included automatically)",
    )
    return parser.parse_args()

def name_template_vtk(sample, time, block):
    """
    Returns string needed to read VTK data
    """
    return f"{sample}/vtk_data/{sample}/*/*_{time}_{block}_0.vtu"

def name_template_exodus(basedir, sample, exodus_name):
    """
    Returns string needed to read exodus data
    """
    return f"{basedir}/{sample}/{exodus_name}"

def read_data(sample_names, exodus_name, fieldname="temperature", basedir="./"):
    """
    Loop over all sample names and read data files.
    Save all data to a np.ndarray

    Parameters
    ----------
    sample_names : list
        list of strings containing sample names (corresponding to folders)
        containing data
    fieldname : string
        string to read values of pointdata object to read

    Returns
    -------
    dataset : array_like
        2D array of data to model, where each row is one sample/timestep,
        and each column is a value from the field(s) sampled
    """
    dataset = []

    fname_list = []
    for sample in sample_names:
        fname = name_template_exodus(basedir, sample, exodus_name)
        fname_list.append(fname)
    ex_reader = ExodusReader(fieldname, nozero=True, to_array=True, to_dict=False)
    dataset = ex_reader.read_all_samples(fname_list)
    _, mesh = ex_reader.read_fname(fname_list[0], return_mesh=True)
    return dataset, mesh

if __name__ == "__main__":
    args = get_inputs()

    sample_names = glob(f"{args.path_to_samples}/sample*")
    sample_names.sort()

    # read all data files. 
    # We have several samples, each sample has several timesteps
    # and each timestep may have several blocks (each with its own vtk file)
    # or all contained in a single exodus file (per sample)
    # dataset is a np.ndarray of all data, shape is (n_samples, n_steps, n_points)
    # mesh_base is a meshio object containing mesh data
    dataset, mesh_base = read_data(
        sample_names, 
        args.exodus_name, 
        args.field, 
        basedir=args.path_to_samples
    )

    # find mean and std-dev of all samples
    mean_field = dataset.mean(axis=0)
    stddev_field = dataset.std(axis=0)

    write_timeseries(
        mesh_base,
        np.arange(len(mean_field)),
        [mean_field, stddev_field],
        ["mean", "stddev"],
        f"{args.outname}.xdmf3",
    )
