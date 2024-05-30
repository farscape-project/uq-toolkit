#!/bin/bash

python parse_digraph_to_moose.py

rm *.e
$MOOSE_DIR/modules/combined/combined-opt -i coolant.i

