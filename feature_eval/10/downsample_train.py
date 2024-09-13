import csv
import math
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python downsample_train.py trainingdata_path")
    sys.exit(1)

trainingdata_path = sys.argv[1]

def read_final_train(input_file):
    cheaters = []
    non_cheaters = []
    
    with open(input_file, mode='r', encoding='utf_8_sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['isCheater'] == '1':
                cheaters.append(row)
            else:
                non_cheaters.append(row)

    return cheaters, non_cheaters

def write_to_file(filename, headers, data):
    with open(filename, mode='w', newline='', encoding='utf_8_sig') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

def split_data(input_file):
    cheaters, non_cheaters = read_final_train(input_file)
    n = len(cheaters)

    partitions = len(non_cheaters) // n
    last_partition_size = len(non_cheaters) % n

    if last_partition_size > n/2:
        partitions += 1

    input_dir = os.path.dirname(input_file)
    headers = cheaters[0].keys()

    for i in range(partitions):
        if i == partitions - 1 and 0 < last_partition_size <= n/2:
            start_idx = (i - 1) * n
            end_idx = start_idx + n + last_partition_size
        else:
            start_idx = i * n
            end_idx = start_idx + n

        data = cheaters + non_cheaters[start_idx:end_idx]
        filename = os.path.join(input_dir, f'final_train_{i + 1}.csv')
        write_to_file(filename, headers, data)

if __name__ == '__main__':
    split_data(trainingdata_path)
    print("Data downsampling complete.")
