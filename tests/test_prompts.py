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
    prompt_reference_path = os.path.join(dirname, "test_prompt_1_hydrated.prompt")

    def test_hydrate_prompt(self):
        """Test that hydrating prompts works as expected."""
        hydrated_prompt = prompts.hydrate_prompt(self.prompt_path, self.example_row)
        with open(self.prompt_reference_path, 'r') as reference:
            self.assertEqual(hydrated_prompt, reference.read())


if __name__ == '__main__':
    unittest.main()
