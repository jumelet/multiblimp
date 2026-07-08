import os
import pickle
import re
from glob import glob
import urllib.request

from arabic2latin import arabic_to_latin
from conllu import parse_incr
from indic_transliteration.sanscript import IAST, DEVANAGARI, transliterate
from unidecode import unidecode
from bs4 import BeautifulSoup

from .config import UD_PATH
from .languages import udlang2treebanks, convert_arabic_to_latin_langs


def has_typo(item):
    is_reparandum = item["deprel"] == "reparandum"

    feats = item.get("feats") or {}
    feats_has_typo = feats.get("Typo", "No") == "Yes"
    feats_has_style = "Style" in feats
    feats_has_foreign = "Foreign" in feats

    misc = item.get("misc") or {}
    misc_has_correction = any("Correct" in misc_key for misc_key in misc.keys())
    # misc_has_lang = ("Lang" in misc) or ("OrigLang" in misc)

    if (
        is_reparandum
        or feats_has_typo
        or feats_has_style
        or feats_has_foreign
        or misc_has_correction
        # or misc_has_lang
    ):
        return True

    return False


def tree_is_malformed(tree):
    for item in tree:
        if has_typo(item):
            return True

    if all(item["form"] == "_" for item in tree):
        return True

    return False


def flag_treebanks(flag_type: str) -> dict[str, list[str]]:
    """
    Scrape UD and flag treebanks matching a given category.

    Args:
        flag_type: Either "words removed" (treebanks where the underlying
            text has been removed, detected via a data-hint marker on the
            treebank header itself) or "sign language" (treebanks whose
            language is a sign language, detected via the "Sign Language"
            text on the parent language header, since individual sign
            language treebank rows carry no distinguishing marker of
            their own).

    Returns:
        Dict mapping language_name -> [treebank_names]
    """
    flag_type = flag_type.strip().lower()
    if flag_type not in ("words removed", "sign language"):
        raise ValueError(
            f'flag_type must be "words removed" or "sign language", got {flag_type!r}'
        )

    fp = urllib.request.urlopen("https://universaldependencies.org")
    html_str = fp.read().decode("utf8")
    fp.close()

    soup = BeautifulSoup(html_str, features="html.parser")
    results = {}

    def name_of(header) -> str:
        try:
            return header.select_one("span.doublewidespan").get_text(strip=True)
        except AttributeError:
            return "UNKNOWN"

    if flag_type == "words removed":
        marker = 'span[data-hint="Underlying text not included"]'

        for treebank_header in soup.select("div.ui-accordion-header"):
            if treebank_header.select_one("span.flagspan img") is not None:
                continue  # skip language-level headers
            if treebank_header.select_one(marker) is None:
                continue

            try:
                lang_header = (
                    treebank_header
                    .find_parent("div", class_="ui-accordion-content")
                    .find_previous_sibling("div", class_="ui-accordion-header")
                )
                language_name = name_of(lang_header)
            except AttributeError:
                language_name = None

            results.setdefault(language_name, []).append(name_of(treebank_header))

    elif flag_type=="sign language":  # "sign language"
        for lang_header in soup.select("div.ui-accordion-header"):
            if lang_header.select_one("span.flagspan img") is None:
                continue  # skip treebank-level headers

            is_sign_language = any(
                span.get_text(strip=True) == "Sign Language"
                for span in lang_header.select("span.triplewidespan")
            )
            if not is_sign_language:
                continue

            language_name = name_of(lang_header).replace(" Sign Language", "")

            try:
                treebank_headers = lang_header.find_next_sibling(
                    "div", class_="ui-accordion-content"
                ).select("div.ui-accordion-header")
            except AttributeError:
                treebank_headers = []

            results.setdefault(language_name, []).extend(
                name_of(tb) for tb in treebank_headers
            )

    return results


class Treebank:
    def __new__(
        cls,
        lang: str,
        remove_diacritics: bool = False,
        verbose: bool = False,
        load_from_pickle: bool = False,
        resource_dir: str | None = None,
        test_files_only: bool = False,
        use_selected_treebanks: bool = True,
        remove_typo: bool = True,
        pickle_path: str = "ud/ud_pickles",
    ):
        if resource_dir is None:
            resource_dir = "."

        if load_from_pickle:
            pickle_path = os.path.join(resource_dir, pickle_path, f"{lang}.pickle")
            with open(pickle_path, "rb") as f:
                return pickle.load(f)

        if test_files_only:
            treebank_glob = os.path.join(UD_PATH, f"UD_{lang}*/*test*.conllu")
        else:
            treebank_glob = os.path.join(UD_PATH, f"UD_{lang}*/*.conllu")
        treebank_glob = os.path.join(resource_dir, treebank_glob)
        treebank_paths = glob(treebank_glob)

        skip_flagged = flag_treebanks("sign language").get((lang if not "_" in lang else lang.split("_")[0]), [])
        treebank_paths = [p for p in treebank_paths if p.split("/")[-2].split("-")[-1].lower() not in skip_flagged]
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
            with open(filename, encoding="utf-8") as f:
                for tree in parse_incr(f):
                    tree.metadata["treebank"] = "/".join(filename.split("/")[-2:])
                    treebank.append(tree)

        if remove_typo:
            treebank = [tree for tree in treebank if not tree_is_malformed(tree)]

        for tree in treebank:
            remove_items = [tok for tok in tree if isinstance(tok["id"], tuple)]
            for item in remove_items:
                tree.remove(item)

        if lang in convert_arabic_to_latin_langs:
            for tree in treebank:
                for item in tree:
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
