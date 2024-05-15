#!/bin/bash

rm *.e
mpiexec -n 32 $MOOSE_DIR/modules/combined/combined-opt -i cube_thermal_mechanical.i

