import os
import pickle
import sys
sys.path.append("../src")

from mblimp.pipeline import Pipeline
from mblimp.languages import get_ud_langs, lang2langcode
from mblimp.filters.utils import tokenlist2sen


if __name__ == "__main__":
    resource_dir = "../resources"

    lang_candidates = get_ud_langs(resource_dir)
    lang_candidates = ["Italian"]

    pipeline = Pipeline(
        None,
        None,
        resource_dir=resource_dir,
        load_from_pickle=False,
    )

    for lang in sorted(lang_candidates):
        langcode = lang2langcode(lang)

        treebank = pipeline.load_treebank(lang)

        corpus = [tokenlist2sen(tree) for tree in treebank]

        corpus_dir = os.path.join(resource_dir, "corpora")
        os.makedirs(corpus_dir, exist_ok=True)

        with open(os.path.join(corpus_dir, f"{langcode}.txt"), "w") as f:
            f.write('\n'.join(corpus))

        print(lang, len(corpus))
