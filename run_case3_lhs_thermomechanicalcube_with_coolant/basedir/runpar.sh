#!/bin/bash

rm *.e
mpiexec -n 4 $MOOSE_DIR/modules/combined/combined-opt -w -i cube_thermal_mechanical.i

