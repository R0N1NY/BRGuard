#!/bin/bash

j=$1
pt=$2
tn=$3

if [ -z "$j" ] || [ -z "$pt" ] || [ -z "$tn" ]; then
    echo "No values provided. Usage: ./training_script j_value platformtype_value threadnum4each_value"
    exit 1
fi

export OMP_NUM_THREADS=$tn

for i in $(seq 1 $j)
do
   echo "Running train_model.py for dataset $i, pt $pt"
   python SVM.py $i $pt&
done

# Wait for all background jobs to complete
wait

echo "All datasets have been processed!"
