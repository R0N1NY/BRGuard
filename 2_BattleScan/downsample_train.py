import csv
from collections import defaultdict
from itertools import cycle
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python downsample_train.py trainingdata_path")
    sys.exit(1)

trainingdata_path = sys.argv[1]

def read_and_group_data_by_id(input_file):
    grouped_data_by_id = defaultdict(list)

    with open(input_file, mode='r', encoding='utf_8_sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            grouped_data_by_id[row['ID']].append(row)

    cheater_groups = []
    non_cheater_groups = []

    for id, rows in grouped_data_by_id.items():
        if any(row['isCheater'] == '1' for row in rows):
            cheater_groups.append(rows)
        else:
            non_cheater_groups.append(rows)

    return cheater_groups, non_cheater_groups

def split_data(input_file):
    input_dir = os.path.dirname(input_file)
    cheater_groups, non_cheater_groups = read_and_group_data_by_id(input_file)
    n_cheaters = len(cheater_groups)
    n_non_cheaters = len(non_cheater_groups)

    print(f"Total unique cheater IDs: {n_cheaters}")
    print(f"Total unique non-cheater IDs: {n_non_cheaters}")

    non_cheaters_cycle = cycle(non_cheater_groups)

    total_files = (n_non_cheaters + n_cheaters - 1) // n_cheaters

    for i in range(total_files):
        current_non_cheaters = [next(non_cheaters_cycle) for _ in range(n_cheaters)]
        data_for_partition = [row for group in cheater_groups for row in group] + \
                             [row for group in current_non_cheaters for row in group]

        headers = data_for_partition[0].keys()
        filename = os.path.join(input_dir, f'final_train_{i + 1}.csv')
        print(f"Writing to {filename}: {n_cheaters} cheater(s) and {len(current_non_cheaters)} non-cheater(s)")
        write_to_file(filename, headers, data_for_partition)

def write_to_file(filename, headers, data):
    with open(filename, mode='w', newline='', encoding='utf_8_sig') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

if __name__ == '__main__':
    split_data(trainingdata_path)
    print("Data downsampling complete.")
