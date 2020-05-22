#!/bin/bash

# Stop when ANY error occurs
set -e

# Make sure the number of threads is given
n_threads=16
if [ $# -eq 2 ];
then
    # Override the default number
    n_threads=$1
    echo "OVERRIDE: Using ${n_threads} threads"
    echo ""
elif [ $# -ne 1 ]; 
then
    echo "Usage: $0 [<n_threads>]"
    exit 64
fi

# Do sequential first
echo "Compiling sequential..."
make clean > /dev/null; make digits > /dev/null
echo "Benchmarking sequential..."
bin/digits.out ./digits.csv | grep "Time taken:\|Network accuracy:"

# Loop through all OpenMP variations
for VARIANT in 1 2 3 4 5 6 7 8
do
    echo ""
    echo "Compiling OpenMP variation ${VARIANT}..."
    make clean > /dev/null; make digits OPENMP=${VARIANT} > /dev/null
    echo "Benchmarking OpenMP variation ${VARIANT}..."
    bin/digits.out ./digits.csv n_threads | grep "Time taken:\|Network accuracy:"
done
