#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <n_max> <pt>"
    exit 1
fi

n_max=$1
pt=$2

for ((n=1; n<=n_max; n++))
do
    echo "Running python BattleScan_test.py $n $pt"
    python BattleScan_test.py $n $pt
    
    if [ $? -ne 0 ]; then
        echo "Error running BattleScan_test.py with n=$n and pt=$pt"
        exit 1
    fi
done

echo "All $n_max tests completed successfully."
