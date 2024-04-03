# igt-icl

LLM-based Automated Interlinear Glossing

`igt-icl` is a package that allows for automated interlinear glossing using the in-context abilities of large language models (LLMs) to produce context-sensitive gloss lines.

## Basic Usage
`igt-icl` packages a number of prompts for easy IGT glossing.

```python
from igt_icl import gloss_with_llm

example = {
    'transcription': 'los gatos corren',
    'translation': 'the cats run',
    'language': 'Spanish',
    'metalang': 'English'
}

glosses, number_of_tokens = gloss_with_llm(example,
                                           system_prompt_key='base',
                                           prompt_key='zeroshot',
                                           llm_type='openai',
                                           model='gpt-3.5-turbo-0125',
                                           api_key='your_key_here',
                                           temperature=1,
                                           seed=0)

print(glosses) # "DET.PL cat.PL run.3PL"
```

You can see the entire list of prompts by running
```python
igt_icl.list_prompt_keys()
```

## Retreival-Augmented Generation
`igt-icl` supports RAG for more controllable IGT glossing. There are a number of retrieval strategies implemented, which can be used as follows:

```python
from igt_icl import retriever

my_dataset = datasets.load_dataset("...")
retriever = retriever(method='longest_common_substring', n=4, dataset=my_dataset)
retrieved_examples = retriever(example)

glosses, number_of_tokens = gloss_with_llm(example,
                                           fewshot_examples=retrieved_examples,
                                           system_prompt_key='base',
                                           prompt_key='zeroshot',
                                           llm_type='openai',
                                           model='gpt-3.5-turbo-0125',
                                           api_key='your_key_here',
                                           temperature=1,
                                           seed=0)
```

### GlossLM Retrieval
There is also support for using the [GlossLM Corpus](https://huggingface.co/datasets/lecslab/glosslm-corpus) as a retrieval source. The GlossLM Corpus is a massive multilingual corpus of nearly half a million IGT examples in over a thousand languages.

```python
from igt_icl import glosslm_retriever
retriever = glosslm_retriever(method='longest_common_substring', n=4)
```

