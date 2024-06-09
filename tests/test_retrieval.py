import os
import random
import unittest

import datasets

from igt_icl import Retriever, IGT

dirname = os.path.dirname(__file__)


class TestPrompts(unittest.TestCase):
    example_row = IGT(transcription="Hola señor",
                      translation="Hello sir",
                      language="Spanish",
                      metalang="English",
                      glosses=None,)

    example_dataset = datasets.Dataset.from_dict({
        "transcription": ["Hola señora", "estoy bien", "Hola señor"],
        "translation": ["Hello ma'am", "I'm fine", "Hello sir"],
        "glosses": ["hello lady-F.SG", "be.1SG well", "hello man-M.SG"],
        "language": ["Spanish", "Spanish", "Spanish"],
        "metalang": ["English", "English", "English"]
    })

    def test_random_retriever(self):
        """Test that retrieving a random example works as expected."""
        random.seed(0)
        retriever = Retriever.stock(method="random", 
                                    n_examples=1, 
                                    dataset=self.example_dataset,
                                    seed=0)
        retrieved_examples = retriever.retrieve(self.example_row)
        self.assertListEqual(retrieved_examples, [IGT.from_dict(self.example_dataset[0])])

    def test_word_recall_retriever(self):
        retriever = Retriever.stock(method="word_recall", 
                                    n_examples=1, 
                                    dataset=self.example_dataset)
        retrieved_examples = retriever.retrieve(self.example_row)
        self.assertListEqual(retrieved_examples, [IGT.from_dict(self.example_dataset[0])])

    def test_max_word_coverage_retriever(self):
        # Although example 0 also has a word in common with the target (Hola), we should not select it since it is covered by the first selection (example 2)
        example_row = IGT(transcription="Hola señor bien",
                      translation="Hello good sir",
                      language="Spanish",
                      metalang="English",
                      glosses=None,)
        
        retriever = Retriever.stock(method="max_word_coverage", 
                                    n_examples=2, 
                                    dataset=self.example_dataset)
        retrieved_examples = retriever.retrieve(example_row)
        self.assertListEqual(retrieved_examples, [IGT.from_dict(self.example_dataset[i]) for i in [2, 1]])

if __name__ == '__main__':
    unittest.main()
