#!/bin/bash

# Runs with a gloss list, where morphemes are split 
#   and fusional morphemes (with periods) are split into individual glosses
python3 ./run_experiment.py --glottocode gitx1241 \
                            --system_prompt_key base-glosslist \
                            --prompt_key zeroshot \
                            --use_gloss_list split_morphemes_and_fusional \
                            --output_dir "$( dirname -- "$( readlink -f -- "$0"; )"; )"