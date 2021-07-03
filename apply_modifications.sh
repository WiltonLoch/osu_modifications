#! /bin/sh

# Script destined for applying the Allgatherv modifications to the OSU Micro Benchmarks
# (modifications written for version 5.7).
#
# The modifications add size distributions for the aforementioned call in order to create unbalanced
# block sizes throughout the processes. 

# Utilization: ./apply_modifications.sh <path to OSU folder>


cp collective/* $1/mpi/collective/
cp util/* $1/util/
