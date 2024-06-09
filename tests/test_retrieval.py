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
        "transcription": ["Hola señora", "estoy bien"],
        "translation": ["Hello ma'am", "I'm fine"],
        "glosses": ["hello lady-F.SG", "be.1SG well"],
        "language": ["Spanish", "Spanish"],
        "metalang": ["English", "Spanish"]
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

if __name__ == '__main__':
    unittest.main()
