import json

def split_jsonl_file(input_file, output_prefix, lines_per_file=200):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    total_lines = len(lines)
    num_files = (total_lines + lines_per_file - 1) // lines_per_file

    for i in range(num_files):
        start_idx = i * lines_per_file
        end_idx = min((i + 1) * lines_per_file, total_lines)
        output_file = f"{output_prefix}_{i+1}.jsonl"
        
        with open(output_file, 'w') as outfile:
            for line in lines[start_idx:end_idx]:
                outfile.write(line)

    print(f"Split {total_lines} lines into {num_files} files with up to {lines_per_file} lines each.")

# Usage
input_file = '/Users/milesper/Documents/Research/IGT-Glossing/ICL/experiments/testing/gpt4-chrf/gpt-4o-2024-05-13.uspa1245.chrf.requests.jsonl'
output_prefix = 'output'
split_jsonl_file(input_file, output_prefix, lines_per_file=317)