"""
Defines dynamic prompts (system and user) for generating IGT.
"""
from typing import Dict, List, Optional
from string import Template
from pathlib import Path
import os
from enum import Enum
import re


class PromptType(Enum):
    SYSTEM = "system"
    USER = "user"


def default_prompt(key: str, type: PromptType):
    """Initializes a prompt from the defaults included in the package.

    Args:
        key (str): The key of the prompt. See all keys by running `list_prompt_keys`.
        type (PromptType): Whether the prompt is a PromptType.SYSTEM or PromptType.USER prompt.
    """
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, f"prompts/{type.value}/{key}.prompt")
    return Prompt(path=path, type=type)


class Prompt:
    key: str
    path: str
    type: PromptType
    template: Template
    required_fields: List[str]

    def __init__(self, path: str, type: PromptType) -> None:
        """Initializes a prompt from a given path

        Args:
            path (str): Path to a `.prompt` file
            type (PromptType): Whether the prompt is a PromptType.SYSTEM or PromptType.USER prompt.
        """
        assert path.endswith('.prompt'), "Please provide a .prompt file!"

        self.type = type
        self.key = Path(path).stem

        with open(path, 'r') as prompt_file:
            prompt_string = prompt_file.read()
            self.template = Template(prompt_string)

            # Read all of the required fields
            pattern = r'\$\{?(\w+)\}?'
            self.required_fields = sorted(set(re.findall(pattern, prompt_string)))

    def hydrate(self, data: Dict):
        """Hydrates the prompt with the given data"""
        return self.template.substitute(data)


def list_prompt_keys():
    """Lists all of the available prompt keys"""
    all_prompts = {}

    dirname = os.path.dirname(__file__)
    base_path = Path(os.path.join(dirname, "prompts/"))
    # Iterate over each subdirectory in the base directory
    for subdirectory in base_path.iterdir():
        if subdirectory.is_dir():  # Ensure it's a directory
            # Search for all files ending with `.prompt` within each subdirectory
            files = list(subdirectory.glob("**/*.prompt"))
            # Store the file names (without the full path) in the dictionary, keyed by subdirectory name
            all_prompts[subdirectory.name] = [f.stem for f in files]

    print("System prompts:", all_prompts['system'])
    print("User prompts:", all_prompts['user'])
    return all_prompts
