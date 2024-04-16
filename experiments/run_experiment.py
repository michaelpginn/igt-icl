"""Script for running experiments evaluating the various LLM strategies"""

from typing import Callable, Dict, List, Tuple
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
import re

import datasets
from tqdm.autonotebook import tqdm
import fire

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from igt_icl import gloss_with_llm, Prompt, PromptType, IGT, Retriever, evaluate_igt

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _create_gloss_list(include_gloss_list: str, dataset: datasets.Dataset) -> List[str]:
    """Creates a sorted list of unique (functional) glosses given a dataset and a strategy for splitting. Does not include stem translations.

    Args:
        include_gloss_list (str): 'split_morphemes_and_fusional' | 'split_morphemes'
        dataset (datasets.Dataset): The source dataset

    Returns:
        List[str]: The sorted list of glosses
    """
    assert include_gloss_list in ['split_morphemes_and_fusional', 'split_morphemes'], \
        "`include_gloss_list` must be one of None, 'split_morphemes', 'split_morphemes_and_fusional'"

    dataset = dataset.map(lambda row:
                          {'glosses_split': re.split(r'[ -]', row['glosses'])})
    gloss_list = set()
    for example in dataset['glosses_split']:
        gloss_list.update([gloss for gloss in example if gloss.isupper()])

    if include_gloss_list == 'split_morphemes_and_fusional':
        # Split current glosses into additional glosses for fusional morhemes
        updated_gloss_list = set()
        for gloss in gloss_list:
            updated_gloss_list.update([subgloss for subgloss in
                                       gloss.split('.') if subgloss.isupper()])
        gloss_list = updated_gloss_list
    return sorted(gloss_list)


def _eval_dataset(dataset: datasets.Dataset, inference_function: Callable[[Dict], Tuple[str, int]]):
    """Runs inference over a dataset and computes summary metrics. May iterate over every example, so `inference_function` may be called many times

    Args:
        dataset (datasets.Dataset): The dataset of rows to eval
        inference_function (Callable): Function to run that takes the eval row and returns a predicted gloss line and tokens used.
    """
    predictions = []
    references = []
    total_tokens = 0
    for example in tqdm(dataset, "Running inference"):
        response = inference_function(example)

        references.append(example['glosses'])
        predictions.append(response['response'])
        total_tokens += response['total_tokens']

    metrics = evaluate_igt(predictions=predictions, references=references)
    metrics["total_tokens"] = total_tokens
    metrics["avg_tokens"] = total_tokens / len(dataset)
    print(metrics)
    return metrics, predictions, references


def run_experiment(glottocode: str,
                   system_prompt_key: str,
                   prompt_key: str,
                   output_dir: str,
                   use_gloss_list: str = None,
                   retriever_key: str = None,
                   llm_type: str = 'openai',
                   model="gpt-3.5-turbo-0125",
                   temperature=1,
                   seed=0):
    """Runs an experiment with a given language and prompt combination. Writes results to the specified `output_dir`.

    Args:
        glottocode (str): Glottocode for an eval language. Options: 'dido1241', 'uspa1245', 'arap1274', 'gitx1241', 'lezg1247', 'natu1246', 'nyan1302'
        system_prompt_key (str): The name of a prompt in `prompts/system`, without the file extension.
        prompt_key (str): The name of a prompt in `prompts/user`, without the file extension.
        output_dir (str): The directory to write results to.
        use_gloss_list (boolean): None | 'split_morphemes_and_fusional' | 'split_morphemes'
        retriever_key (Callable[[Dict, datasets.DatasetDict], List[Dict]]): If provided, a function for retrieving similar examples for few-shot prompts. Defaults to None.
        llm_type (str): 'openai' | 'local'
        model (str, optional): The API model to use. Defaults to "gpt-3.5-turbo-0125".
        temperature (int, optional): Defaults to 1.
        seed (int, optional): Defaults to 0.
    """
    assert os.path.isdir(output_dir)

    # Load the appropriate eval dataset
    print("Loading dataset...")
    glosslm_corpus = datasets.load_dataset("lecslab/glosslm-corpus-split")
    glottocodes_ID = set(glosslm_corpus['eval_ID']['glottocode'])
    glottocodes_OOD = set(glosslm_corpus['eval_OOD']['glottocode'])

    if glottocode in glottocodes_ID:
        id_or_ood = "ID"
    elif glottocode in glottocodes_OOD:
        id_or_ood = "OOD"
    else:
        raise Exception(
            f"Glottocode should be one of: {list(glottocodes_ID) + list(glottocodes_OOD)}")

    train_dataset = glosslm_corpus[f"train_{id_or_ood}"].filter(
        lambda row: row['glottocode'] == glottocode)
    eval_dataset = glosslm_corpus[f"eval_{id_or_ood}"].filter(
        lambda row: row['glottocode'] == glottocode)
    print(
        f"Lang: {eval_dataset['language'][0]}\n# Train examples: {len(train_dataset)}\n# Eval examples: {len(eval_dataset)}")

    additional_data = {}

    # Compile the list of glosses observed in the training data, if applicable
    if use_gloss_list is not None:
        additional_data['gloss_list'] = _create_gloss_list(use_gloss_list,
                                                           train_dataset)

    # Create a Retriever, if applicable
    retriever = None
    if retriever_key is not None:
        retriever = Retriever.stock(
            'random', n_examples=3, dataset=glosslm_corpus['train'])

    # Create the inference function for the evaluation loop
    def _inference(example: Dict):
        igt = IGT.from_dict(example)

        # If appropriate, run retrieval and add examples to the prompt data payload
        fewshot_examples = None
        if retriever is not None:
            fewshot_examples = retriever.retrieve(example)

        return gloss_with_llm(igt,
                              system_prompt=Prompt.stock(
                                  system_prompt_key, PromptType.SYSTEM),
                              prompt=Prompt.stock(prompt_key, PromptType.USER),
                              additional_data=additional_data,
                              fewshot_examples=fewshot_examples,
                              llm_type=llm_type,
                              model=model,
                              api_key=OPENAI_API_KEY,
                              temperature=temperature,
                              seed=seed)

    # Run evaluation and write to files
    metrics, predictions, references = _eval_dataset(eval_dataset, _inference)

    with open(os.path.join(output_dir, f"{glottocode}-metrics.json"), 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=4)

    predictions_data = datasets.Dataset.from_dict(
        {'predicted_glosses': predictions, 'glosses': references})
    predictions_data.to_csv(os.path.join(
        output_dir, f"{glottocode}-preds.tsv"), sep='\t')


if __name__ == "__main__":
    fire.Fire(run_experiment)
