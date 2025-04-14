import os
from pathlib import Path
import pickle
import sys
sys.path.append("../src")

from mblimp.treebank import Treebank
from mblimp.languages import (
    get_ud_langs,
    remove_diacritics_langs,
    convert_arabic_to_latin_langs,
    lang2udlang,
)


if __name__ == "__main__":
    resource_dir = "../resources"

    for lang in get_ud_langs(resource_dir):
        lang = lang2udlang.get(lang, lang).replace(" ", "_")
        ud_path = f"ud/ud-treebanks-v2.15/UD_{lang}*/*.conllu"
        treebank = Treebank.init(
            os.path.join(resource_dir, ud_path),
            remove_diacritics=(lang in remove_diacritics_langs),
            convert_arabic_to_latin=(lang in convert_arabic_to_latin_langs),
        )

        print(lang, len(treebank))

        pickle_path = os.path.join(resource_dir, "ud/ud_pickles")
        Path(pickle_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(pickle_path, f"{lang}.pickle"), "wb") as f:
            pickle.dump(treebank, f)
