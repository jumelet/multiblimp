# MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs
[![arXiv](https://img.shields.io/badge/arXiv-2504.02768-<COLOR>.svg)](https://arxiv.org/abs/2504.02768)

MultiBLiMP is a massively multilingual benchmark of linguistic minimal pairs, covering 101 languages, 6 linguistic phenomena and containing more than 125,000 minimal pairs.
This repository contains the code for creating the corpus and the scripts for LLM evaluation.

## Dataset
The full MultiBLiMP dataset is available on [HuggingFace](https://huggingface.co/datasets/jumelet/multiblimp).

A more detailed explanation of evaluating your own LM on MultiBLiMP is provided by Catherine Arnett (thanks!) in this repository: https://github.com/catherinearnett/multiblimp

## Results
We provide a `.csv` dataframe of all model results here (759MB): [Google Drive](https://drive.google.com/file/d/1meCW4AXKLXhOwMnFEV5QmbJnQhMqA0he/view?usp=sharing). Note that, to save disk space, this dataframe does not contain the original sentence pairs. In case you need those, you can download another `.csv` dataframe here (2.4GB): [Google Drive](https://drive.google.com/file/d/10zeCqAJj3tTi1LbokxmNuWtT8Z3652L5/view?usp=sharing).

The most important column in these dataframes is `delta` (log probability difference of the LM). Accuracy can be derived from this as well (`delta > 0`), or directly by taking the mean over the `pred` column. Specific tests should be easy to conduct using `pandas` `groupby` functionality.


## Citation
The paper has been published in TACL:

```
@article{jumelet-etal-2026-multiblimp,
    title = "{M}ulti{BL}i{MP} 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs",
    author = "Jumelet, Jaap  and
      Weissweiler, Leonie  and
      Nivre, Joakim  and
      Bisazza, Arianna",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "14",
    year = "2026",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2026.tacl-1.10/",
    doi = "10.1162/tacl.a.600",
    pages = "193--216"
}
```
