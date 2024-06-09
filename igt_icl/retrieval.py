from typing import Dict, List, Callable, Type, Union, Optional
from abc import ABC, abstractmethod
import datasets
from .igt import IGT

datasets.utils.logging.disable_progress_bar()

class Retriever(ABC):
    _stock_subclasses: Dict[str, Type['Retriever']] = {}

    n_examples: int
    dataset: datasets.Dataset

    def __init__(self,
                 n_examples: int,
                 dataset: Union[datasets.Dataset, str],
                 seed: Optional[int] = None):
        """Initializes a `Retriever`

        Args:
            n_examples (int): How many examples to retrieve for a given call
            dataset (Union[datasets.Dataset, str]): The source dataset to use, or 'glosslm' to use the [glosslm corpus](https://huggingface.co/lecslab/glosslm).
        """
        super().__init__()
        self.n_examples = n_examples
        self.seed = seed 

        if isinstance(dataset, str):
            if dataset == 'glosslm':
                self.dataset = datasets.load_dataset("lecslab/glosslm-corpus") # type: ignore
            else:
                raise Exception(
                    "`dataset` should either be 'glosslm' or a `Dataset` object")
        else:
            self.dataset = dataset

    @classmethod
    def register_class(cls, key: str, concrete_class: Type['Retriever']):
        cls._stock_subclasses[key] = concrete_class

    @classmethod
    def stock_retriever_methods(cls):
        """Returns the keys for the valid stock retrievers"""
        return cls._stock_subclasses

    @classmethod
    def stock(cls,
              method: str,
              n_examples: int,
              dataset: Union[datasets.Dataset, str],
              seed: Optional[int] = None) -> 'Retriever':
        """Initializes a `Retriever` using a stock (built-in) subclass.

        Args:
            method (str): A key for a type of retrieval methods. See all valid options by running `Retriever.stock_retriever_methods()`
            n_examples (int): How many examples to retrieve for a given call
            dataset (Union[datasets.Dataset, str]): The source dataset to use, or 'glosslm' to use the [glosslm corpus](https://huggingface.co/lecslab/glosslm).
        """
        # Look at the _stock_subclasses, which is dynamically updated after subclasses are written
        if method in cls._stock_subclasses:
            return cls._stock_subclasses[method](n_examples=n_examples, dataset=dataset, seed=seed)
        else:
            raise ValueError("fNo stock class with key {key}")

    @abstractmethod
    def retrieve(self, example: IGT) -> List[IGT]:
        """Retrieves `self.n_examples` examples from `self.dataset`. Implementation depends on the subclass.    

        Args:
            example (igt.IGT): The target example
        """
        pass


class RandomRetriever(Retriever):
    def retrieve(self, example: IGT) -> List[IGT]:
        examples = self.dataset \
                       .shuffle(seed=self.seed) \
                       .select(range(self.n_examples))
        return [IGT.from_dict(ex) for ex in examples] # type: ignore
    

class WordRecallRetriever(Retriever):
    """Selects examples that include the max number of words from the target string, divided by the # of words in the other example."""

    def retrieve(self, example: IGT) -> List[IGT]:
        target_words = set(example.transcription.split())

        def _compute_recall(row):
            row_words = set(row['transcription'].split())
            matches = sum([1 for word in row_words if word in target_words])
            return {"recall": matches / len(target_words)}

        examples = self.dataset \
                        .shuffle(seed=self.seed) \
                        .map(_compute_recall, batched=False) \
                        .sort("recall", reverse=True) \
                        .select(range(self.n_examples))
        return [IGT.from_dict(ex) for ex in examples] # type: ignore
        

class WordPrecisionRetriever(Retriever):
    """Selects examples that include the max number of words from the target string, divided by the # of words in the other example."""

    def retrieve(self, example: IGT) -> List[IGT]:
        target_words = set(example.transcription.split())

        def _compute_precision(row):
            row_words = row['transcription'].split()
            matches = sum([1 for word in row_words if word in target_words])
            return {"precision": matches / len(row_words)}

        examples = self.dataset \
                        .shuffle(seed=self.seed) \
                        .map(_compute_precision, batched=False) \
                        .sort("precision", reverse=True) \
                        .select(range(self.n_examples))
        return [IGT.from_dict(ex) for ex in examples] # type: ignore


Retriever.register_class('random', RandomRetriever)
Retriever.register_class('word_recall', WordRecallRetriever)
Retriever.register_class('word_precision', WordRecallRetriever)