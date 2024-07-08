import os
import glob

def combine_jsonl_files(input_prefix, output_file):
    # Use glob to find all files matching the input prefix pattern
    input_files = sorted(glob.glob(f"{input_prefix}*.jsonl"))
    
    with open(output_file, 'w') as outfile:
        for file_name in input_files:
            with open(file_name, 'r') as infile:
                for line in infile:
                    outfile.write(line)
    
    print(f"Combined {len(input_files)} files into {output_file}.")

# Usage
input_prefix = 'testing/gpt4-chrf/uspa1245.response.'
output_file = 'uspa1245.jsonl'
combine_jsonl_files(input_prefix, output_file)