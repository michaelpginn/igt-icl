#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

glottocode=$1

for seed in 1
do  
  python3 "$SCRIPT_DIR/../../../run_experiment.py" --glottocode $glottocode \
                                                --segmented False \
                                                --system_prompt_key base-glosslist \
                                                --prompt_key zeroshot \
                                                --use_gloss_list split_morphemes \
                                                --llm_type cohere \
                                                --model command-r-plus \
                                                --output_dir "$( dirname -- "$( readlink -f -- "$0"; )"; )" \
                                                --seed $seed
done