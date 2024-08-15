# UQ-toolkit

This repo is envisaged as being an interface mainly to MOOSE, but also must be able to interface with other tools such as meshers for pre-processing. Potentially this could also work with other simulation software such as OpenFOAM (using PyFOAM for parsing).


## TODO list
- Post-processing of runs
    - get PDFs, variance, conf intervals
- surrogate modelling
    - Can use [emukit](https://github.com/EmuKit/emukit), [GPy](https://github.com/SheffieldML/GPy)

- generate initial sampling, then use GP surrogate to sample high spaces with high uncertainty

## Usage
### Installation

Currently we do not have anything installable. The library is used by `python /path/to/source/python/setup_uq_run.py`. In future, we will probably look to make this installable with `pip`, and can be used via the command line (or imported to an existing python tool?)

### Examples overview

We have provided a several examples to show how to set up a UQ run, for the kind of cases we are working on. The scope of these examples is as follows:

- `run_case1_thermomechanicalcube`: This is a cube geometry with an applied heat-flux BC. The material properties are given as csv files which are read by MOOSE as `PiecewiseLinear` functions. The `config_thermomech.jsonc` file describes the uncertain parameters. In this example, we show how to put uncertainty on the heat-flux (a scalar value), and material properties. We fit a polynomial to each material properties and the coefficients of the polynomial are treated as uncertain. The `basedir` contains info on the baseline simulation, with a script to setup a geometry and run the executable. `config_thermomech.jsonc` points to this run script. The basedir is copied N times, new input files with perturbed parameters are created, and the `run.sh` or `runpar.sh` script is launched. We have not yet set up any post-processing.
- `run_case2_thermomechanicalcube_with_coolant` builds on the previous example by showing how to impose uncertainties in a MOOSE MultiApp. The MultiApp couples the cube to a coolant pipe flowing through the center. Heat-flux is sent from the solid to the coolant. Coolant temperature is sent back to the solid. The same uncertainties as above are included, but with additional uncertainty on the friction factor of the coolant pipe in the sub-app.
- `run_case3_chimera_coolant_only` shows how to put uncertainties on some general simulator which is set up using a json file. In this case, we have a custom python script which makes a MOOSE input file based on the geometry of the CHIMERA coolant network (`digraph.gml`) and other simulation parameters taken from `chimera_params.jsonc`. The config file for the UQ run (`config.jsonc`) overwrites the `chimera_params.jsonc` variable `straight_pipe_friction_factor` for each new sample. When the job is submitted, the new `chimera_params.jsonc` is used to create a MOOSE input file which is then run.
