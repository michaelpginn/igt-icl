"""Script for running experiments evaluating the various LLM strategies"""
import os
import sys
import json 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from pathlib import Path

import datasets
import fire
from dotenv import load_dotenv
from tqdm.autonotebook import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from igt_icl import evaluate_igt
from igt_icl.llm import _parse_response

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("CO_API_KEY")



def evaluate_response(path,
                      glottocode: str,
                      segmented: bool):

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
    glosslm_corpus = glosslm_corpus.filter(lambda row: row["is_segmented"] == "yes" if segmented else row["is_segmented"] == "no")
    eval_dataset = glosslm_corpus[f"test_{id_or_ood}"].filter(
        lambda row: row['glottocode'] == glottocode)

    # Read in response
    responses = {}
    with open(path, 'r') as file:
        for line in file:
            json_line = json.loads(line.strip())
            custom_id = json_line['custom_id']
            content = json_line['response']['body']['choices'][0]['message']['content']
            content = _parse_response(content)
            responses[custom_id] = content
    
    predictions = []
    references = []

    for example in eval_dataset:
        references.append(example['glosses'])
        predictions.append(responses[example['id']])
    metrics = evaluate_igt(predictions=predictions, references=references)
    print(metrics)

    predictions_data = datasets.Dataset.from_dict(
        {'predicted_glosses': predictions, 'glosses': references})
    predictions_data.to_csv(f"{path}.preds.tsv", sep='\t')

if __name__ == "__main__":
    fire.Fire(evaluate_response)
