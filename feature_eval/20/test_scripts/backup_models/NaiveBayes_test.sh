#!/bin/bash

for i in {1..2}
do
   echo "Running _test.py for dataset $i"
   python NaiveBayes_test.py $i &
done

# Wait for all background jobs to complete
wait

echo "All datasets have been processed!"
