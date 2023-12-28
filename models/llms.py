from functools import partial
from models.gpt import gpt_completion_fn, gpt_nll_fn
from models.gpt import tokenize_fn as gpt_tokenize_fn
from models.llama import llama_completion_fn, llama_nll_fn
from models.llama import tokenize_fn as llama_tokenize_fn

completion_fns = {
    'text-davinci-003': partial(gpt_completion_fn, model='text-davinci-003'),
    'gpt-4': partial(gpt_completion_fn, model='gpt-4'),
    'gpt-3.5-turbo-instruct': partial(gpt_completion_fn, model='gpt-3.5-turbo-instruct'),
    'llama-7b': partial(llama_completion_fn, model='7b'),
    'llama-13b': partial(llama_completion_fn, model='13b'),
    'llama-70b': partial(llama_completion_fn, model='70b'),
    'llama-7b-chat': partial(llama_completion_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_completion_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_completion_fn, model='70b-chat'),
}

nll_fns = {
    'text-davinci-003': partial(gpt_nll_fn, model='text-davinci-003'),
    'llama-7b': partial(llama_nll_fn, model='7b'),
    'llama-13b': partial(llama_nll_fn, model='13b'),
    'llama-70b': partial(llama_nll_fn, model='70b'),
    'llama-7b-chat': partial(llama_nll_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_nll_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_nll_fn, model='70b-chat'),
}

tokenization_fns = {
    'text-davinci-003': partial(gpt_tokenize_fn, model='text-davinci-003'),
    'gpt-3.5-turbo-instruct': partial(gpt_tokenize_fn, model='gpt-3.5-turbo-instruct'),
    'llama-7b': partial(llama_tokenize_fn, model='7b'),
    'llama-13b': partial(llama_tokenize_fn, model='13b'),
    'llama-70b': partial(llama_tokenize_fn, model='70b'),
    'llama-7b-chat': partial(llama_tokenize_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_tokenize_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_tokenize_fn, model='70b-chat'),
}

context_lengths = {
    'text-davinci-003': 4097,
    'gpt-3.5-turbo-instruct': 4097,
    'llama-7b': 4096,
    'llama-13b': 4096,
    'llama-70b': 4096,
    'llama-7b-chat': 4096,
    'llama-13b-chat': 4096,
    'llama-70b-chat': 4096,
}

def __init__():
    """
    Initialize the LLMS module.

    This method is called when an instance of the LLMS class is created.
    It can be used to initialize any attributes or perform any setup tasks.
    """
    pass
