import argparse
import os
from pathlib import Path
import pickle
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from multiblimp.treebank import Treebank
from multiblimp.languages import (
    get_ud_langs,
    remove_diacritics_langs,
    gblang2udlang,
    ancient_langs,
)
from word_order.process_treebank import load_treebank, subsample_treebank
from word_order.utils import MAX_TREEBANK_LEN


def pickle_treebanks(resource_dir: str, langs: list[str] | None = None, overwrite: bool = False) -> None:
    """Pickle the UD treebanks of `langs` (default: all UD languages), with and without typos."""
    for lang in langs or get_ud_langs(resource_dir):
        lang = gblang2udlang.get(lang, lang).replace(" ", "_")
        for remove_typo in [True, False]:
            if remove_typo:
                pickle_path = os.path.join(resource_dir, "ud/ud_pickles")
            else:
                pickle_path = os.path.join(resource_dir, "ud/ud_typo_pickles")
            Path(pickle_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(pickle_path, f"{lang}.pickle")
            if os.path.exists(out_path) and not overwrite:
                continue

            treebank = Treebank(
                lang,
                remove_diacritics=(lang in remove_diacritics_langs),
                resource_dir=resource_dir,
                remove_typo=remove_typo,
                filter_style=(lang.replace("_", " ") not in ancient_langs),
            )

            print(lang, len(treebank))
            if len(treebank) == 0:
                continue

            with open(out_path, "wb") as f:
                pickle.dump(treebank, f)


def subsample_pickles(
    resource_dir: str,
    langs: list[str] | None = None,
    max_treebank_len: int = MAX_TREEBANK_LEN,
    overwrite: bool = False,
) -> None:
    """Store the pickled treebanks subsampled to max_treebank_len trees in ud/ud_pickles_max{N}."""
    full_dir = os.path.join(resource_dir, "ud/ud_pickles")
    out_dir = os.path.join(resource_dir, f"ud/ud_pickles_max{max_treebank_len}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for lang in langs or get_ud_langs(resource_dir):
        lang = gblang2udlang.get(lang, lang).replace(" ", "_")
        out_path = os.path.join(out_dir, f"{lang}.pickle")
        if not os.path.exists(os.path.join(full_dir, f"{lang}.pickle")):
            continue
        if os.path.exists(out_path) and not overwrite:
            continue
        treebank = subsample_treebank(load_treebank(lang, full_dir), max_treebank_len)
        with open(out_path, "wb") as f:
            pickle.dump(treebank, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource_dir", default="../../resources")
    parser.add_argument("--overwrite_existing", action="store_true")
    args = parser.parse_args()
    pickle_treebanks(args.resource_dir, overwrite=args.overwrite_existing)
    subsample_pickles(args.resource_dir, overwrite=args.overwrite_existing)
