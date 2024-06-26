from typing import Dict, Union, List, Tuple, Optional
from openai import OpenAI
import cohere
import re
import os
from io import TextIOWrapper
from httpx import ReadTimeout

from . import prompts
from .igt import IGT

dirname = os.path.dirname(__file__)


def _run_openai_prompt(hydrated_system_prompt: str,
                       hydrated_prompt: str,
                       api_key: str,
                       model="gpt-3.5-turbo-0125",
                       temperature=0,
                       seed=0,
                       verbose=False) -> Tuple[str, int]:
    """Runs a specified system prompt and user prompt using the OpenAI API.

    Args:
        system_prompt (str): A (hydrated) system prompt
        prompt (str): A (hydrated) user prompt.
        api_key (str): OpenAI API key
        model (str): The model name. Check [the documentation](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
        temperature (int, optional): Defaults to 1.
        seed (int, optional): Defaults to 0.
        verbose (bool, optional): If true, log information about the LLM response. Defaults to True.

    Returns:
        Tuple[str, int]: The LLM chat completion and number of tokens used.
    """
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": hydrated_system_prompt},
            {"role": "user", "content": hydrated_prompt}
        ],
        temperature=temperature,
        top_p=1,
        seed=seed
    )

    if verbose:
        print(completion.usage)
        print(completion.model)

    tokens = completion.usage.total_tokens if completion.usage is not None else -1

    return completion.choices[0].message.content or "Error", tokens


def _run_cohere_prompt(hydrated_system_prompt: str,
                       hydrated_prompt: str,
                       api_key: str,
                       model="command-r-plus",
                       temperature=0,
                       seed=0,
                       verbose=False) -> Tuple[str, int]:
    """Runs a specified system prompt and user prompt using the OpenAI API.

    Args:
        system_prompt (str): A (hydrated) system prompt
        prompt (str): A (hydrated) user prompt.
        api_key (str): OpenAI API key
        model (str): The model name. Check [the documentation](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
        temperature (int, optional): Defaults to 1.
        seed (int, optional): Defaults to 0.
        verbose (bool, optional): If true, log information about the LLM response. Defaults to True.

    Returns:
        Tuple[str, int]: The LLM chat completion and number of tokens used.
    """
    client = cohere.Client(api_key=api_key)

    try:
        completion = client.chat(
            model=model,
            message=hydrated_prompt,
            preamble=hydrated_system_prompt,
            temperature=temperature,
            p=1,
            seed=seed
        )
    except ReadTimeout:
        # Try one more time
        completion = client.chat(
            model=model,
            message=hydrated_prompt,
            preamble=hydrated_system_prompt,
            temperature=temperature,
            p=1,
            seed=seed
        )

    if verbose:
        print(completion.meta)

    total_tokens = -1
    if completion.meta is not None and completion.meta.tokens is not None:
        input_tokens = completion.meta.tokens.input_tokens or 0
        output_tokens = completion.meta.tokens.output_tokens or 0
        total_tokens = int(input_tokens + output_tokens)


    return completion.text, total_tokens


class LLMResponseError(Exception):
    pass


def _parse_response(response: str) -> Union[str, List[str]]:
    """Parses an LLM response, hopefully including 'Glosses: '.

    Args:
        response (str): The response, which should include 'Glosses: ___'

    Raises:
        LLMResponseError: If the response is not in the correct format.

    Returns:
        Union[str, List[str]]: Either the gloss line (if a single example) or a list of gloss lines (if multiple examples)
    """
    response_pattern = r"Glosses: (.*)(\n|$)"
    matches = re.findall(response_pattern, response, re.M)
    if len(matches) == 0:
        raise LLMResponseError("Response was not in correct format")
    if len(matches) > 1:
        return [match[0] for match in matches]
    else:
        return matches[0][0]


def gloss_with_llm(example: IGT,
                   system_prompt: prompts.Prompt,
                   prompt: prompts.Prompt,
                   additional_data: Dict = {},
                   fewshot_examples: List[IGT] = [],
                   llm_type: str = 'openai',
                   model: str = 'gpt-3.5-turbo-0125',
                   api_key: Optional[str] = None,
                   temperature=0,
                   log_file: Optional[TextIOWrapper] = None,
                   seed=0,
                   verbose: bool = False) -> Dict:
    """Actually runs LLM inference on a single example.

    Args:
        example (IGT): The example to run inference on.
        system_prompt (prompts.Prompt): A system prompt to run
        prompt (prompts.Prompt): A prompt to run
        additional_data (Dict): A dictionary containing additional fields to fill into the prompt
        fewshot_examples (List[igt.IGT]): A list of examples to include as few-shots, for relevant prompts
        llm_type (str): 'openai' | 'cohere' | 'local'. Defaults to 'openai'.
        model (str, optional): Name of the model to use. Defaults to 'gpt-3.5-turbo-0125'.
        temperature (int, optional): Defaults to 0.
        log_file: (TextIOWrapper, optional): If provided, a file to write logs to. 
        seed (int, optional): Defaults to 0.
        verbose (bool): Defaults to False

    Returns:
        Dict: A dictionary containing the following:
                `response`: The gloss line(s) predicted by the model
                `total_tokens`: The total number of tokens used
                `system_prompt`: The (hydrated) system prompt
                `prompt`: The (hydrated) prompt
    """
    fewshot_examples_dict = {'fewshot_examples':
                        '\n\n'.join(map(str, fewshot_examples))}
    hydrated_system_prompt = system_prompt.hydrate(example.__dict__, 
                                                   additional_data, 
                                                   fewshot_examples_dict)
    hydrated_prompt = prompt.hydrate(example.__dict__, 
                                     additional_data, 
                                     fewshot_examples_dict)

    if log_file is not None:
        log_file.write(
            f"===SYSTEM PROMPT===\n{hydrated_system_prompt}\n\n===PROMPT===\n{hydrated_prompt}\n")

    if llm_type == 'openai':
        assert api_key is not None
        response, num_tokens_used = _run_openai_prompt(hydrated_system_prompt=hydrated_system_prompt,
                                                       hydrated_prompt=hydrated_prompt,
                                                       api_key=api_key,
                                                       model=model,
                                                       temperature=temperature,
                                                       seed=seed,
                                                       verbose=verbose)
    elif llm_type == 'cohere':
        assert api_key is not None
        response, num_tokens_used = _run_cohere_prompt(hydrated_system_prompt=hydrated_system_prompt,
                                                       hydrated_prompt=hydrated_prompt,
                                                       api_key=api_key,
                                                       model=model,
                                                       temperature=temperature,
                                                       seed=seed,
                                                       verbose=verbose)
    elif llm_type == 'local':
        raise NotImplementedError
    else:
        raise Exception('Invalid `llm_type` passed')

    if log_file is not None:
        log_file.write(f"===RESPONSE===\n{response}\n\n===END====\n\n\n\n")

    response = _parse_response(response)
    if isinstance(response, list):
        response = response[0]

    return {
        'response': response,
        'total_tokens': num_tokens_used,
        'system_prompt': hydrated_system_prompt,
        'prompt': hydrated_prompt
    }
