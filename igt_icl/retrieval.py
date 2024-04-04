from typing import Dict, List, Callable, Type, Union
from abc import ABC, abstractmethod
import datasets
from .igt import IGT


class Retriever(ABC):
    _stock_subclasses: Dict[str, Type['Retriever']] = {}

    n_examples: int
    dataset: datasets.Dataset

    def __init__(self,
                 n_examples: int,
                 dataset: Union[datasets.Dataset, str]):
        """Initializes a `Retriever`

        Args:
            n_examples (int): How many examples to retrieve for a given call
            dataset (Union[datasets.Dataset, str]): The source dataset to use, or 'glosslm' to use the [glosslm corpus](https://huggingface.co/lecslab/glosslm).
        """
        super().__init__()
        self.n_examples = n_examples

        if isinstance(dataset, str):
            if dataset == 'glosslm':
                self.dataset = datasets.load_dataset("lecslab/glosslm-corpus")
            else:
                raise Exception("`dataset` should either be 'glosslm' or a `Dataset` object")
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
              dataset: Union[datasets.Dataset, str]) -> 'Retriever':
        """Initializes a `Retriever` using a stock (built-in) subclass.

        Args:
            method (str): A key for a type of retrieval methods. See all valid options by running `Retriever.stock_retriever_methods()`
            n_examples (int): How many examples to retrieve for a given call
            dataset (Union[datasets.Dataset, str]): The source dataset to use, or 'glosslm' to use the [glosslm corpus](https://huggingface.co/lecslab/glosslm).
        """
        if method in cls._stock_subclasses:
            return cls._stock_subclasses[method].__init__(n_examples=n_examples, dataset=dataset)
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
        examples = self.dataset.shuffle().select(range(self.n_examples))
        return [IGT(*ex) for ex in examples]


Retriever.register_class('random', RandomRetriever)
