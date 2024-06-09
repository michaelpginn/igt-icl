#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

for seed in 0 1 2
do  
  python3 "$SCRIPT_DIR/../../../run_experiment.py" --glottocode $1 \
                                                  --segmented False \
                                                  --system_prompt_key base \
                                                  --prompt_key fewshot \
                                                  --retriever_key random \
                                                  --num_fewshot_examples 10 \
                                                  --llm_type cohere \
                                                  --model command-r-plus \
                                                  --output_dir "$( dirname -- "$( readlink -f -- "$0"; )"; )" \
                                                  --seed $seed
done