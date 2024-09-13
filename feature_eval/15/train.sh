#!/bin/bash

if [ $# -ne 3 ]; then
    echo "No sufficient arguments provided. Usage: ./train.sh j_value platformtype_value threadnum4each_value"
    exit 1
fi

j=$1
pt=$2
tn=$3

ORIGINAL_DIR=$(pwd)

export OMP_NUM_THREADS=$tn

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}/train_scripts"

for script in *.sh; do
    echo "Running script $script with pt=$pt"
    ./$script $j $pt $tn || echo "Script $script failed to execute properly."
done

echo "All training scripts have been processed!"

cd "${ORIGINAL_DIR}"
