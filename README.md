# MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs
[![arXiv](https://img.shields.io/badge/arXiv-2504.02768-<COLOR>.svg)](https://arxiv.org/abs/2504.02768)

MultiBLiMP is a massively multilingual benchmark of linguistic minimal pairs, covering 101 languages, 6 linguistic phenomena and containing more than 125,000 minimal pairs.
This repository contains the code for creating the corpus and the scripts for LLM evaluation.

## Dataset
The full MultiBLiMP dataset is available on [HuggingFace](https://huggingface.co/datasets/jumelet/multiblimp).

A more detailed explanation of evaluating your own LM on MultiBLiMP is provided by Catherine Arnett (thanks!) in this repository: https://github.com/catherinearnett/multiblimp

We provide a `.csv` dataframe of all model results here (759MB): [Google Drive](https://drive.google.com/file/d/1meCW4AXKLXhOwMnFEV5QmbJnQhMqA0he/view?usp=sharing).


## Citation
The paper has been accepted into TACL and should be on MIT Press soon!

```
@misc{jumelet2025multiblimp10massivelymultilingual,
      title={MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs}, 
      author={Jaap Jumelet and Leonie Weissweiler and Arianna Bisazza},
      year={2025},
      eprint={2504.02768},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.02768}, 
}
```
