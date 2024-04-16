import unittest

from igt_icl import prompts
import os

dirname = os.path.dirname(__file__)


class TestPrompts(unittest.TestCase):
    example_row = {
        "transcription": "Hola",
        "translation": "Hello",
        "language": "Spanish",
        "metalang": "English"
    }
    prompt_path = os.path.join(dirname, "test_prompt_1.prompt")
    prompt_reference_path = os.path.join(
        dirname, "test_prompt_1_hydrated.prompt")
    default_prompt_reference_path = os.path.join(
        dirname, "test_prompt_base_hydrated.prompt")

    def test_hydrate_prompt(self):
        """Test that hydrating prompts works as expected."""
        prompt = prompts.Prompt(self.prompt_path, prompts.PromptType.SYSTEM)
        hydrated_prompt = prompt.hydrate(self.example_row)
        with open(self.prompt_reference_path, 'r') as reference:
            self.assertEqual(hydrated_prompt, reference.read())

    def test_default_prompt(self):
        """Test that we can fetch and hydrate a built-in prompt"""
        prompt = prompts.Prompt.stock('base', prompts.PromptType.SYSTEM)
        assert prompt.required_fields == ['language', 'metalang']
        hydrated_prompt = prompt.hydrate(self.example_row)
        with open(self.default_prompt_reference_path, 'r') as reference:
            self.assertEqual(hydrated_prompt, reference.read())

    def test_list_stock_prompts(self):
        """Test that we can get a list of the built-in prompt keys"""
        all_prompts = prompts.Prompt.list_stock_prompts()
        assert 'system' in all_prompts
        assert 'user' in all_prompts
        assert len(all_prompts['system']) > 0
        assert len(all_prompts['user']) > 0


if __name__ == '__main__':
    unittest.main()
