import argparse
import os
from pathlib import Path
import pickle
import sys

sys.path.append("../../src")

from multiblimp.treebank import Treebank
from multiblimp.languages import (
    get_ud_langs,
    remove_diacritics_langs,
    gblang2udlang,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource_dir", default="../../resources")
    args = parser.parse_args()
    resource_dir = args.resource_dir
    ud_langs = get_ud_langs(resource_dir)

    for lang in ud_langs:
        for remove_typo in [True, False]:
            lang = gblang2udlang.get(lang, lang).replace(" ", "_")
            if remove_typo:
                pickle_path = os.path.join(resource_dir, "ud/ud_pickles")
            else:
                pickle_path = os.path.join(resource_dir, "ud/ud_typo_pickles")
            Path(pickle_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(pickle_path, f"{lang}.pickle")
            if os.path.exists(out_path):
                continue

            treebank = Treebank(
                lang,
                remove_diacritics=(lang in remove_diacritics_langs),
                resource_dir=resource_dir,
                remove_typo=remove_typo,
            )

            print(lang, len(treebank))
            if len(treebank)==0:
                continue

            with open(out_path, "wb") as f:
                pickle.dump(treebank, f)
