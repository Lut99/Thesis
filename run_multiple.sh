# Stop when ANY command fails
set -e

# Script that runs one of the optimised codes multiple times
if [ $# -eq 2 ];
then
    # Set the number of times
    iters=$2
elif [ $# -eq 1 ];
then
    # Set it to the default number
    iters=10
elif [ $# -ne 1 ]; 
then
    echo "Usage: $0 <sequential | {OpenMP variation}>"
    exit 64
fi

# Compile that variation first
echo "Compiling OpenMP variation $1..."
make clean > /dev/null; make digits OPENMP=$1 > /dev/null

for ((i=1;i<=$iters;i++));
do
    echo ""
    echo "(${i}/${iters}) Benchmarking OpenMP variation $1..."
    bin/digits.out ./digits.csv | grep "Time taken:\|Network accuracy:"
done
