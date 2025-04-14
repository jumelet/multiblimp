import os
import pickle
import re
from glob import glob
from typing import *

from arabic2latin import arabic_to_latin
from conllu import parse_incr, TokenList
from indic_transliteration.sanscript import IAST, DEVANAGARI, transliterate
from unidecode import unidecode

from .languages import udlang2treebanks, convert_arabic_to_latin_langs


def has_typo(item):
    is_reparandum = item['deprel'] == 'reparandum'

    feats = item.get("feats") or {}
    feats_has_typo = feats.get("Typo", "No") == "Yes"
    feats_has_style = "Style" in feats
    feats_has_foreign = "Foreign" in feats

    misc = item.get("misc") or {}
    misc_has_correction = any("Correct" in misc_key for misc_key in misc.keys())
    misc_has_lang = ("Lang" in misc) or ("OrigLang" in misc)

    if (
            is_reparandum
            or feats_has_typo
            or feats_has_style
            or feats_has_foreign
            or misc_has_correction
            or misc_has_lang
    ):
        return True

    return False


def tree_has_typo(tree):
    for item in tree:
        if has_typo(item):
            return True

    return False


class Treebank:
    def __new__(
        cls,
        lang: str,
        remove_diacritics: bool = False,
        verbose: bool = False,
        load_from_pickle: bool = False,
        resource_dir: str = ".",
        test_files_only: bool = False,
        use_selected_treebanks: bool = True,
        remove_typo: bool = True,
        pickle_path: str = "ud/ud_pickles",
    ):
        if load_from_pickle:
            pickle_path = os.path.join(resource_dir, pickle_path, f"{lang}.pickle")
            with open(pickle_path, "rb") as f:
                return pickle.load(f)

        if test_files_only:
            treebank_glob = f"ud/ud-treebanks-v2.15/UD_{lang}*/*test*.conllu"
        else:
            treebank_glob = f"ud/ud-treebanks-v2.15/UD_{lang}*/*.conllu"
        treebank_glob = os.path.join(resource_dir, treebank_glob)
        treebank_paths = glob(treebank_glob)

        selected_treebanks = udlang2treebanks.get(lang)
        if use_selected_treebanks and selected_treebanks is not None:
            selected_paths = []
            for path in treebank_paths:
                treebank_name = path.split("/")[-2].split("-")[-1]
                if treebank_name in selected_treebanks:
                    selected_paths.append(path)
            treebank_paths = selected_paths

        if verbose:
            print("Loading:\n", "\n".join(treebank_paths))

        treebank = []
        for filename in treebank_paths:
            with open(filename) as f:
                for tree in parse_incr(f):
                    tree.metadata["treebank"] = "/".join(filename.split("/")[-2:])
                    treebank.append(tree)

        if remove_typo:
            treebank = [
                tree
                for tree in treebank
                if not tree_has_typo(tree)
            ]

        for tree in treebank:
            remove_items = [tok for tok in tree if isinstance(tok["id"], tuple)]
            for item in remove_items:
                tree.remove(item)

        if lang in convert_arabic_to_latin_langs:
            for tree in treebank:
                for item in tree:
                    if item["form"] == "بورسۇق":
                        x=1
                    if item.get("misc", {}).get("Translit"):
                        item["form"] = item["misc"]["Translit"]
                    else:
                        item["form"] = arabic_to_latin(item["form"])

                    if item.get("misc", {}).get("LTranslit"):
                        item["lemma"] = item["misc"]["LTranslit"]
                    else:
                        item["lemma"] = arabic_to_latin(item["lemma"])

        if remove_diacritics:
            for tree in treebank:
                for item in tree:
                    item["form"] = unidecode(item["form"])
                    item["lemma"] = unidecode(item["lemma"])

        if lang == "Sanskrit":
            for tree in treebank:
                for item in tree:
                    item["form"] = transliterate(item["form"], IAST, DEVANAGARI)
                    item["lemma"] = transliterate(item["lemma"], IAST, DEVANAGARI)

        if lang == "Ancient_Hebrew":
            def remove_hebrew_cantillation(text):
                # https://stackoverflow.com/q/44479533/351197
                pattern = r"[\u0591-\u05AF\u05BE\u05C0\u05C3]"
                return re.sub(pattern, "", text)

            for tree in treebank:
                for item in tree:
                    item["form"] = remove_hebrew_cantillation(item["form"])
                    item["lemma"] = remove_hebrew_cantillation(item["lemma"])

        return treebank
