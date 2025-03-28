# UQ-toolkit

This repo is envisaged as being an interface mainly to MOOSE, but also must be able to interface with other tools such as meshers for pre-processing. Potentially this could also work with other simulation software such as OpenFOAM (using PyFOAM for parsing).

## Getting started
### Installation
Dependencies can be installed as:
```bash
pip install hjson UQpy numpy scipy
```

To use for modifying MOOSE input files, we require `pyhit` is available, which is installed with MOOSE. This can be located by following:
```bash
export PYTHONPATH=$MOOSE_DIR/python:$PYTHONPATH
```

### Usage 

To set-up UQ jobs, we require a directory containing (1) a folder named `basedir` containing input files to be copied and modified, and (2) a config file (json format) with information on uncertain parameters. Several examples are provided, and explained in further detail below. In general, the config file has the following structure:

```json5
{
    "apps":
    {
        "cube_thermal_mechanical.i":
        {
            "type":"moose",
            "uncertain-params":
            {
                // items to modify in the file `cube_thermal_mechanical.i`
            },
            "uq_log_name": "uq_log", // name of file used to archive the uncertain params
        }
    },
    "paths":
    {
        // generally unchanged
        "workdir":"./", 
        "baseline_dir":"basedir",
    },
    "sampler" : "montecarlo", // currently supports "montecarlo" or "latinhypercube"
    "num_samples" : 10, // can be overriden at command line
    
    // if not launched through the nextflow pipeline, a bash script can be used to launch all jobs on e.g. a SLURM cluster
    "launcher_name" : "launcher.sh",
    "launcher" : "bash", // supports "bash", "slurm" or "lsf"
    "template_launcher_script":"runpar.sh", // refers to a script in the `paths/baseline_dir` that will launch the job
}
```

To create a new number of cases which sample the uncertain PDFs, we run the following script:

```bash
python python/setup_uq_run.py \
    -c run_case1_thermomechanicalcube/config_thermomech.jsonc \
    -b run_case1_thermomechanicalcube/basedir \
    -n 10
```

### Examples overview

We have provided a several examples to show how to set up a UQ run, for the kind of cases we are working on. The scope of these examples is as follows:

- `run_case1_thermomechanicalcube`: This is a cube geometry with an applied heat-flux BC. The material properties are given as csv files which are read by MOOSE as `PiecewiseLinear` functions. The `config_thermomech.jsonc` file describes the uncertain parameters. In this example, we show how to put uncertainty on the heat-flux (a scalar value), and material properties. We fit a polynomial to each material properties and the coefficients of the polynomial are treated as uncertain. The `basedir` contains info on the baseline simulation, with a script to setup a geometry and run the executable. `config_thermomech.jsonc` points to this run script. The basedir is copied N times, new input files with perturbed parameters are created, and the `run.sh` or `runpar.sh` script is launched. We have not yet set up any post-processing.
- `run_case2_thermomechanicalcube_with_coolant` builds on the previous example by showing how to impose uncertainties in a MOOSE MultiApp. The MultiApp couples the cube to a coolant pipe flowing through the center. Heat-flux is sent from the solid to the coolant. Coolant temperature is sent back to the solid. The same uncertainties as above are included, but with additional uncertainty on the friction factor of the coolant pipe in the sub-app.
- `run_case3_chimera_coolant_only` shows how to put uncertainties on some general simulator which is set up using a json file. In this case, we have a custom python script which makes a MOOSE input file based on the geometry of the CHIMERA coolant network (`digraph.gml`) and other simulation parameters taken from `chimera_params.jsonc`. The config file for the UQ run (`config.jsonc`) overwrites the `chimera_params.jsonc` variable `straight_pipe_friction_factor` for each new sample. When the job is submitted, the new `chimera_params.jsonc` is used to create a MOOSE input file which is then run.
