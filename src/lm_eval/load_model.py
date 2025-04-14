import tempfile
from typing import *

from transformers import AutoModelForCausalLM, AutoTokenizer
from minicons import scorer


# todo: take script from the actual MultiBLiMP file, not in this order
SCRIPTS = [
    "arab",
    "beng",
    "cyrl",
    "ethi",
    "tibt",
    "thaa",
    "grek",
    "hebr",
    "deva",
    "armn",
    "cans",
    "latn",
]
SIZES = [
    "5mb",
    "10mb",
    "100mb",
    "1000mb",
]


def load_hf_model(model_name: str, no_cache=False, **kwargs):
    model = None
    tokenizer = None

    try:
        if no_cache:
            with tempfile.TemporaryDirectory() as tmpdirname:
                model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=tmpdirname, **kwargs)
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=tmpdirname, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    except OSError:
        pass

    if model is None:
        return None

    ilm_model = scorer.IncrementalLMScorer(
        model,
        'cuda',
        tokenizer=tokenizer,
    )

    return ilm_model


def load_goldfish_model(langcode: str, size: str, script: Optional[str] = None, no_cache=False, **kwargs):
    if script is not None:
        return load_goldfish(langcode, script, size, no_cache=no_cache)
    else:
        model = None

        for script in SCRIPTS:
            model_name = f"goldfish-models/{langcode}_{script}_{size}"
            model = load_hf_model(model_name, no_cache=no_cache, **kwargs)
            if model is not None:
                break

        return model
