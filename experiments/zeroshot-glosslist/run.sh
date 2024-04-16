#!/bin/bash

python3 ./run_experiment.py --glottocode gitx1241 \
                            --system_prompt_key base-glosslist \
                            --prompt_key zeroshot \
                            --use_gloss_list split_morphemes \
                            --output_dir "$( dirname -- "$( readlink -f -- "$0"; )"; )"