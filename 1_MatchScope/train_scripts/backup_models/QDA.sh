#!/bin/bash

for i in {1..2}
do
   echo "Running train_model.py for dataset $i"
   python QDA.py $i &
done

# Wait for all background jobs to complete
wait

echo "All datasets have been processed!"
