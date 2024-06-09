#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

glottocode=$1

for seed in 0 1 2
do  
  python3 "$SCRIPT_DIR/../../../run_experiment.py" --glottocode $glottocode \
                                                --segmented False \
                                                --system_prompt_key base \
                                                --prompt_key zeroshot \
                                                --llm_type cohere \
                                                --model command-r-plus \
                                                --output_dir "$( dirname -- "$( readlink -f -- "$0"; )"; )" \
                                                --seed $seed
done