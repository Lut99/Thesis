#!/bin/bash

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
    bin/digits.out ./digits.csv | grep "Time taken:\|Network accuracy:"
done
