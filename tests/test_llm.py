import os
from igt_icl import llm
import unittest


class TestLLM(unittest.TestCase):
    def test_single_response(self):
        """Test that parsing a single valid response works"""
        response_1 = "Glosses: DET dog-PL walk-SI"
        self.assertEqual(llm._parse_response(response_1), "DET dog-PL walk-SI")

        response_2 = "Certainly! Here are the glosses you asked for.\nGlosses: DET dog-PL walk-SI"
        self.assertEqual(llm._parse_response(response_2), "DET dog-PL walk-SI")

    def test_invalid_response(self):
        response = "I'm sorry, as a large language model I am not able to provide interlinear glosses, as this may be unsafe for children."
        self.assertRaises(llm.LLMResponseError, lambda: llm._parse_response(response))

    def test_multiple_responses(self):
        response_1 = "Glosses: DET dog-PL walk-SI\nGlosses: NOUN cat-SG sleep-SI"
        self.assertEqual(llm._parse_response(response_1), ["DET dog-PL walk-SI", "NOUN cat-SG sleep-SI"])

        response_2 = "Certainly! I'd love to provide glosses. Glosses: DET dog-PL walk-SI\nHere's another one!\nGlosses: NOUN cat-SG sleep-SI\nHave a good day!"
        self.assertEqual(llm._parse_response(response_2), ["DET dog-PL walk-SI", "NOUN cat-SG sleep-SI"])


if __name__ == '__main__':
    unittest.main()
