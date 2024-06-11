#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

python3 "$SCRIPT_DIR/../../../run_experiment.py" --glottocode $1 \
                                                --segmented False \
                                                --system_prompt_key base \
                                                --prompt_key fewshot \
                                                --use_gloss_list split_morphemes \
                                                --retriever_key chrf \
                                                --num_fewshot_examples 2 \
                                                --llm_type cohere \
                                                --model command-r-plus \
                                                --output_dir "$( dirname -- "$( readlink -f -- "$0"; )"; )" \
                                                --seed $2
