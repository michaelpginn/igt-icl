from typing import Dict, Union, List, Tuple
from openai import OpenAI
import re
import os

from . import prompts

dirname = os.path.dirname(__file__)


def _run_openai_prompt(hydrated_system_prompt: str,
                       hydrated_prompt: str,
                       api_key: str,
                       model="gpt-3.5-turbo-0125",
                       temperature=1,
                       seed=0,
                       verbose=True) -> Tuple[str, int]:
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

    return completion.choices[0].message.content, completion.usage.total_tokens


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


def gloss_with_llm(data: Dict,
                   system_prompt: prompts.Prompt,
                   prompt: prompts.Prompt,
                   llm_type: str,
                   model: str,
                   api_key: str = None,
                   temperature=1,
                   seed=0) -> Tuple[Union[str, List[str]], int]:
    """Actually runs LLM inference on a single example.

    Args:
        data (Dict): The example to run inference on. Must contain 'transcription', 'translation', 'language', and 'metalang', and any other fields required by the prompts.
        system_prompt (prompts.Prompt): A system prompt to run
        prompt (prompts.Prompt): A prompt to run
        llm_type (str): 'openai' | 'local'
        model (str, optional): Name of the model to use.
        temperature (int, optional): Defaults to 1.
        seed (int, optional): Defaults to 0.

    Returns:
        Tuple[Union[str, List[str]], int]: The gloss line or list of gloss lines predicted by the LLM, and number of tokens used.
    """
    hydrated_system_prompt = system_prompt.hydrate(data)
    hydrated_prompt = prompt.hydrate(data)

    if llm_type == 'openai':
        response, num_tokens_used = _run_openai_prompt(hydrated_system_prompt=hydrated_system_prompt,
                                                       hydrated_prompt=hydrated_prompt,
                                                       api_key=api_key,
                                                       model=model,
                                                       temperature=temperature,
                                                       seed=seed,
                                                       verbose=True)
    elif llm_type == 'local':
        raise NotImplementedError
    else:
        raise Exception('Invalid `llm_type` passed')

    return _parse_response(response), num_tokens_used
