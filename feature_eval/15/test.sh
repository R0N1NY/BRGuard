#!/bin/bash

if [ $# -ne 2 ]; then
    echo "No sufficient arguments provided. Usage: ./test.sh j_value platformtype_value"
    exit 1
fi

j=$1
pt=$2
ORIGINAL_DIR=$(pwd)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}/test_scripts"

for script in *.sh; do
    echo "Running script $script with pt=$pt"
    ./$script $j $pt || echo "Script $script failed to execute properly."
done

echo "All testing scripts have been processed!"

cd "${ORIGINAL_DIR}"
