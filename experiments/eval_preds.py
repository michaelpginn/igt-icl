import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from igt_icl import evaluate_igt

def read_tsv(file_path):
    col1 = []
    col2 = []

    with open(file_path, mode='r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
            if len(row) >= 2 and len(row[0].strip()) > 0:
                col1.append(row[0])
                col2.append(row[1])
            else:
                print("skipping ", row[1])

    return col1, col2

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_tsv>")
        sys.exit(1)

    file_path = sys.argv[1]
    col1, col2 = read_tsv(file_path)
    print(evaluate_igt(col1, col2))