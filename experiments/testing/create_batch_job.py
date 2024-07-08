"""Script for running experiments evaluating the various LLM strategies"""
import openai
import morfessor
from tqdm.autonotebook import tqdm
from dotenv import load_dotenv
import fire
import datasets
from typing import Callable, Dict, List, Tuple, Optional
from pathlib import Path
import re
import json
import os
import sys
import random
import time
import string
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from igt_icl import (IGT, Prompt, PromptType, Retriever, evaluate_igt,
                     gloss_with_llm, retrieval)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("CO_API_KEY")


def create_batch_job(glottocode: str,
                     segmented: bool,
                     system_prompt_key: str,
                     prompt_key: str,
                     retriever_key: Optional[str] = None,
                     num_fewshot_examples: Optional[int] = None,
                     llm_type: str = 'openai',
                     model="gpt-3.5-turbo-0125",
                     temperature=0,
                     seed=0):

    assert (num_fewshot_examples is None and retriever_key is None) or (
        num_fewshot_examples is not None and retriever_key is not None)

    random.seed(seed)

    # Load the appropriate eval dataset
    print("Loading dataset...")
    glosslm_corpus = datasets.load_dataset("lecslab/glosslm-corpus-split")
    glottocodes_ID = set(glosslm_corpus['test_ID']['glottocode'])
    glottocodes_OOD = set(glosslm_corpus['test_OOD']['glottocode'])

    if glottocode in glottocodes_ID:
        id_or_ood = "ID"
    elif glottocode in glottocodes_OOD:
        id_or_ood = "OOD"
    else:
        raise Exception(
            f"Glottocode should be one of: {list(glottocodes_ID) + list(glottocodes_OOD)}")

    # Filter by segmentation and glottocode
    glosslm_corpus = glosslm_corpus.filter(
        lambda row: row["is_segmented"] == "yes" if segmented else row["is_segmented"] == "no")
    train_dataset = glosslm_corpus[f"train_{id_or_ood}"].filter(
        lambda row: row['glottocode'] == glottocode)
    eval_dataset = glosslm_corpus[f"test_{id_or_ood}"].filter(
        lambda row: row['glottocode'] == glottocode)
    language = eval_dataset['language'][0]
    print(
        f"Lang: {language}\n# Train examples: {len(train_dataset)}\n# Eval examples: {len(eval_dataset)}")

    # If needed, create silver segmented data
    if retriever_key in ['morpheme_recall']:
        io = morfessor.MorfessorIO()
        morfessor_model_path = os.path.join(
            project_root, f"experiments/segmentation/{glottocode}.model")
        morfessor_model = io.read_binary_model_file(morfessor_model_path)

        punctuation_to_remove = string.punctuation.replace("'", "")

        def _segment(row):
            transcription = row['transcription'].translate(
                str.maketrans("", "", punctuation_to_remove))
            row['segmentation'] = ' '.join([' '.join(morfessor_model.viterbi_segment(
                word.lower())[0]) for word in transcription.split()])
            return row

        train_dataset = train_dataset.map(_segment)
        eval_dataset = eval_dataset.map(_segment)

    additional_data = {}

    # Create a Retriever, if applicable
    retriever = None
    if retriever_key is not None:
        if retriever_key == 'morpheme_recall':
            retriever = retrieval.WordRecallRetriever(n_examples=num_fewshot_examples,
                                                      dataset=train_dataset,
                                                      transcription_key='segmentation')
        else:
            retriever = Retriever.stock(retriever_key,
                                        n_examples=num_fewshot_examples,
                                        dataset=train_dataset)

    api_key = OPENAI_API_KEY if llm_type == 'openai' else COHERE_API_KEY
    client = openai.OpenAI(api_key=api_key)

    system_prompt = Prompt.stock(system_prompt_key, PromptType.SYSTEM)
    prompt = Prompt.stock(prompt_key, PromptType.USER)

    requests = []

    for example in tqdm(eval_dataset):
        igt = IGT.from_dict(example)

        # If appropriate, run retrieval and add examples to the prompt data payload
        fewshot_examples = []
        if retriever is not None:
            if retriever_key in ['morpheme_recall']:
                igt.transcription = example['segmentation']
            fewshot_examples = retriever.retrieve(igt)
            for ex in fewshot_examples:
                ex.translation = None

        fewshot_examples_dict = {'fewshot_examples':
                                 '\n\n'.join(map(str, fewshot_examples))}
        hydrated_system_prompt = system_prompt.hydrate(igt.__dict__,
                                                       additional_data,
                                                       fewshot_examples_dict)
        hydrated_prompt = prompt.hydrate(igt.__dict__,
                                         additional_data,
                                         fewshot_examples_dict)

        request = {
            "custom_id": example['id'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": hydrated_system_prompt},
                    {"role": "user", "content": hydrated_prompt}
                ],
                "max_tokens": 256,
                "temperature": 0.2,
            }
        }
        requests.append(request)

    filename = f'./{model}.{glottocode}.{retriever_key}.requests.jsonl'
    with open(filename, 'w') as outfile:
        for entry in requests:
            json.dump(entry, outfile)
            outfile.write('\n')

    batch_input_file = client.files.create(file=open(filename, "rb"),
                                           purpose="batch")
    info = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"{glottocode} test"
        }
    )
    print(info)


if __name__ == "__main__":
    fire.Fire(create_batch_job)
