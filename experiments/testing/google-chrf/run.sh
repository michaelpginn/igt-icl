#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

glottocode=$1


python3 "$SCRIPT_DIR/../../run_experiment.py" --glottocode $1 \
                                                  --segmented False \
                                                  --system_prompt_key base-no_transl \
                                                  --prompt_key fewshot-no_transl \
                                                  --use_gloss_list split_morphemes \
                                                  --retriever_key chrf \
                                                  --num_fewshot_examples 200 \
                                                  --omit_translations True \
                                                  --llm_type google \
                                                  --model gemini-1.5-pro \
                                                  --split test \
                                                  --temperature 0.2 \
                                                  --output_dir "$( dirname -- "$( readlink -f -- "$0"; )"; )" \
                                            
