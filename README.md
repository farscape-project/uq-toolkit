# UQ-toolkit

This repo is envisaged as being an interface mainly to MOOSE, but also must be able to interface with other tools such as meshers for pre-processing. Potentially this could also work with other simulation software such as OpenFOAM (using PyFOAM for parsing).


## TODO list
- Post-processing of runs
    - get PDFs, variance, conf intervals
- surrogate modelling
    - Can use [emukit](https://github.com/EmuKit/emukit), [GPy](https://github.com/SheffieldML/GPy)

- generate initial sampling, then use GP surrogate to sample high spaces with high uncertainty
    - requires handling sampling **globally** rather than parameter-by-parameter as done currently (how would this work with different distributions?)
        - Would allow other sampling methods e.g. latin hypercube

