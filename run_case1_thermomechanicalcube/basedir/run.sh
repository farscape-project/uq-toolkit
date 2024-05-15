#!/bin/bash

rm *.e
$MOOSE_DIR/modules/combined/combined-opt -i cube_thermal_mechanical.i

