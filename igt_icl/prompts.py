"""
Defines dynamic prompts (system and user) for generating IGT.
"""
from typing import Dict
from string import Template


def hydrate_prompt(path: str, data: Dict):
    """Loads the prompt from the given file path and hydrates it with information from the row.

    Args:
        path (str): Path to a `.prompt` file.
        data (Dict): A dict containing required fields: 'transcription', 'translation', 'language', and 'metalang'
    """
    assert path.endswith('.prompt'), "Please provide a .prompt file!"

    with open(path, 'r') as prompt_file:
        prompt_string = prompt_file.read()
        prompt_template = Template(prompt_string)
        return prompt_template.substitute(data)


def list_prompt_keys():
    """Lists all of the available prompt keys"""
    raise NotImplementedError
