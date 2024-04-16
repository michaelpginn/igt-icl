#!/bin/bash

python3 ./run_experiment.py --glottocode gitx1241 \
                            --system_prompt_key base-glosslist \
                            --prompt_key fewshot \
                            --use_gloss_list split_morphemes \
                            --retriever_key random \
                            --num_fewshot_examples 3 \
                            --output_dir "$( dirname -- "$( readlink -f -- "$0"; )"; )"