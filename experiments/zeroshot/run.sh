#!/bin/bash

python3 ./run_experiment.py --glottocode gitx1241 \
                            --system_prompt_key base \
                            --prompt_key zeroshot \
                            --output_dir "$( dirname -- "$( readlink -f -- "$0"; )"; )"