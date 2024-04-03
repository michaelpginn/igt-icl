from typing import Dict, Union, List, Tuple
from openai import OpenAI
import re
import os

from . import prompts

dirname = os.path.dirname(__file__)


def _run_openai_prompt(system_prompt: str, prompt: str, api_key: str, model="gpt-3.5-turbo-0125", temperature=1, seed=0, verbose=True) -> Tuple[str, int]:
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
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


def gloss_with_llm(data: Dict, system_prompt_key: str, prompt_key: str, llm_type: str, model: str, api_key: str = None, temperature=1, seed=0) -> Tuple[Union[str, List[str]], int]:
    """Actually runs LLM inference on a single example.

    Args:
        data (Dict): The example to run inference on. Must contain 'transcription', 'translation', 'language', and 'metalang', and any other fields required by the prompts.
        system_prompt_key (str): The name of a prompt in `prompts/system`, without the file extension.
        prompt_key (str): The name of a prompt in `prompts/user`, without the file extension.
        llm_type (str): 'openai' | 'local'
        model (str, optional): Name of the model to use.
        temperature (int, optional): Defaults to 1.
        seed (int, optional): Defaults to 0.

    Returns:
        Tuple[Union[str, List[str]], int]: The gloss line or list of gloss lines predicted by the LLM, and number of tokens used.
    """
    system_prompt = prompts.hydrate_prompt(os.path.join(dirname, f"prompts/system/{system_prompt_key}.prompt"), data)
    prompt = prompts.hydrate_prompt(os.path.join(dirname, f"prompts/user/{prompt_key}.prompt"), data)

    if llm_type == 'openai':
        response, num_tokens_used = _run_openai_prompt(system_prompt=system_prompt,
                                                       prompt=prompt,
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
