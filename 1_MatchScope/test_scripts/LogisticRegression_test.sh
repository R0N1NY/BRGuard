#!/bin/bash
j=$1
pt=$2

if [ -z "$j" ] || [ -z "$pt" ]; then
    echo "No values provided. Usage: ./training_script j_value platformtype_value"
    exit 1
fi

for i in $(seq 1 $j)
do
   echo "Running LogisticRegression_test.py for dataset $i, pt $pt"
   python LogisticRegression_test.py $i $pt&
done

# Wait for all background jobs to complete
wait

echo "All datasets have been processed!"
