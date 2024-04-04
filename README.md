# igt-icl

LLM-based Automated Interlinear Glossing

`igt-icl` is a package that allows for automated interlinear glossing using the in-context abilities of large language models (LLMs) to produce context-sensitive gloss lines.

## Basic Usage
`igt-icl` packages a number of stock prompts for easy IGT glossing.

```python
from igt_icl import IGT, gloss_with_llm, Prompt, PromptType

example = IGT(transcription='los gatos corren',
              translation='the cats run',
              language='Spanish',
              metalang='English')

glosses, number_of_tokens = gloss_with_llm(example,
                                           system_prompt=Prompt.stock('base', PromptType.SYSTEM),
                                           prompt=Prompt.stock('zeroshot', PromptType.USER),
                                           llm_type='openai',
                                           model='gpt-3.5-turbo-0125',
                                           api_key='your_key_here')

print(glosses) # "DET.PL cat.PL run.3PL"
```

### Stock Prompts

You can see the entire list of stock prompts by running
```python
Prompt.list_prompt_keys()
```

### Prompt Fields
Some prompts, such as `base_glosslist`, require additional data to be passed in via the `additional_data={}` argument of the `gloss_with_llm` function. You can view the requirements of a given prompt using:

```python
prompt = Prompt.stock('base_glosslist', PromptType.SYSTEM),
prompt.required_fields
# ['language', 'metalang', 'gloss_list']
```

### Custom Prompts
You can easily add custom prompts as well. First, you will need to create a file with the `.prompt` extension, as in `myprompt.prompt`. This prompt can be either **system** prompt or **user** prompt. System prompts are called at the start of the conversation and generally provide broad guidance to the LLM. User prompts provide the actual request, i.e., to create the IGT.

```python
custom_prompt = Prompt('/path/to/myprompt.prompt`, PromptType.SYSTEM)
```

Your prompt may contain whatever text you want. You may use **placeholders** to input values from the IGT example and context. For example, the following block uses the `$language` and `$metalang` placeholders:

```text
You are an expert documentary linguist, specializing in $language. You are working on a documentation project for $language text, where you are creating annotated text corpora using the interlinear glossed text (IGT) and following the Leipzig glossing conventions.

Specifically, you will be provided with a line of text in $language as well as a translation of the text into $metalang, in the following format.
```

Currently, we support the following standard placeholders:
- `$language`
- `$metalang`
- `$transcription`
- `$translation`
- `$fewshot_examples`, for methods with retrieval/few-shot learning

You may also add additional placeholders, which should be specified in the `additional_data={}` argument of the `gloss_with_llm` function.

**Crucially**, your prompt must cause the LLM to output something containing the gloss line in the format `Glosses: <gloss line here>`, or the parsing method will fail. 

## Retrieval-Augmented Generation
`igt-icl` supports RAG for more controllable IGT glossing. There are a number of retrieval strategies implemented, accessed through `Retriever.stock()` as follows:

```python
from igt_icl import Retriever

my_dataset = datasets.load_dataset("...")
retriever = Retriever.stock(method='longest_common_substring', n=4, dataset=my_dataset)
retrieved_examples = retriever.retrieve(example)

glosses, number_of_tokens = gloss_with_llm(example,
                                           system_prompt_key='base',
                                           prompt_key='zeroshot',
                                           fewshot_examples=retrieved_examples,
                                           llm_type='openai',
                                           model='gpt-3.5-turbo-0125',
                                           api_key='your_key_here',
                                           temperature=1,
                                           seed=0)
```

### GlossLM Retrieval
There is also support for using the [GlossLM Corpus](https://huggingface.co/datasets/lecslab/glosslm-corpus) as a retrieval source. The GlossLM Corpus is a massive multilingual corpus of nearly half a million IGT examples in over a thousand languages.

```python
from igt_icl import Retriever
retriever = Retriever.stock(method='longest_common_substring', n=4, dataset='glosslm')
```

