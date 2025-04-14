import os
from pathlib import Path
import pickle
import sys
sys.path.append("../../src")

from mblimp.treebank import Treebank
from mblimp.languages import (
    get_ud_langs,
    remove_diacritics_langs,
    gblang2udlang,
)


if __name__ == "__main__":
    resource_dir = "../resources"
    ud_langs = get_ud_langs(resource_dir)

    for lang in ud_langs:
        for remove_typo in [True, False]:
            lang = gblang2udlang.get(lang, lang).replace(" ", "_")
            treebank = Treebank(
                lang,
                remove_diacritics=(lang in remove_diacritics_langs),
                resource_dir=resource_dir,
                remove_typo=remove_typo
            )

            print(lang, len(treebank))

            if remove_typo:
                pickle_path = os.path.join(resource_dir, "ud/ud_pickles")
            else:
                pickle_path = os.path.join(resource_dir, "ud/ud_typo_pickles")
            Path(pickle_path).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(pickle_path, f"{lang}.pickle"), "wb") as f:
                pickle.dump(treebank, f)
