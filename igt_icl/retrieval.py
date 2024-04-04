from typing import Dict, List
from abc import ABC, abstractmethod
import datasets


class Retriever(ABC):
    n_examples: int
    dataset: datasets.Dataset

    def __init__(self, n_examples: int, dataset: datasets.Dataset):
        """Initializes a `Retriever`

        Args:
            n_examples (int): How many examples to retrieve for a given call
            dataset (datasets.Dataset): The source dataset to use
        """
        super().__init__()
        self.n_examples = n_examples
        self.dataset = dataset

    @classmethod
    def glosslm(cls, n_examples: int):
        """Initializes a `Retriever` that retrieves from the [glosslm corpus](https://huggingface.co/lecslab/glosslm).

        Args:
            n_examples (int): How many examples to retrieve for a given call
        """
        glosslm_corpus = datasets.load_dataset("lecslab/glosslm-corpus")
        return Retriever(n_examples=n_examples, dataset=glosslm_corpus['train'])

    @abstractmethod
    def retrieve_related(example: Dict) -> List[Dict]:
        """Retrieves `self.n_examples` examples from `self.dataset`. The retrieval strategy depends on the subclass implementation.

        Args:
            example (Dict): _description_
        """
        pass
