import os
import pickle
import re
from glob import glob
import urllib.request
from pathlib import Path

from arabic2latin import arabic_to_latin
from conllu import parse_incr
from indic_transliteration.sanscript import IAST, DEVANAGARI, transliterate
from unidecode import unidecode
from bs4 import BeautifulSoup

from .config import UD_PATH
from .languages import udlang2excluded_treebanks, convert_arabic_to_latin_langs, excluded_sentences


def has_typo(item, filter_style: bool = True):
    is_reparandum = item["deprel"] == "reparandum"

    feats = item.get("feats") or {}
    feats_has_typo = feats.get("Typo", "No") == "Yes"
    feats_has_style = filter_style and "Style" in feats
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


def tree_is_malformed(tree, filter_style: bool = True):
    for item in tree:
        if has_typo(item, filter_style=filter_style):
            return True

    if all(item["form"] == "_" for item in tree):
        return True

    # Masked tokens: Czech-CAC writes numbers and symbols as "#" and "*", with a
    # placeholder lemma such as "&camount;", "&cyear;" or "&cwildcard;".
    if any(re.fullmatch(r"&c[a-z]+;", item["lemma"] or "") for item in tree):
        return True

    # Broken lines (e.g. a missing column) leave a syntactic word without a head.
    if any(isinstance(item["id"], int) and item["head"] is None for item in tree):
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


def remove_multiword_tokens(tree):
    """
    Remove multiword token lines (e.g. "zum" = "zu" + "dem") and empty nodes from
    a tree, so that only syntactic words remain. Each multiword token is stored on
    the misc of its words, so the surface string can still be rendered:
      MWT=<id of its first word> on every word of the token,
      MWTForm=<surface form> on its first word,
      SpaceAfter=No on its last word, if the multiword token had it.
    """
    for item in [tok for tok in tree if isinstance(tok["id"], tuple)]:
        start, sep, end = item["id"]
        if sep == "-":
            words = [
                tok
                for tok in tree
                if isinstance(tok["id"], int) and start <= tok["id"] <= end
            ]
            for word in words:
                word["misc"] = word["misc"] or {}
                word["misc"]["MWT"] = str(start)
            words[0]["misc"]["MWTForm"] = item["form"]
            if (item["misc"] or {}).get("SpaceAfter") == "No":
                words[-1]["misc"]["SpaceAfter"] = "No"
        tree.remove(item)


def remove_excluded_sentences(treebank) -> list:
    """Remove the sentences listed in languages.excluded_sentences."""

    def excluded(tree):
        pattern = excluded_sentences.get(tree.metadata.get("treebank", "").split("/")[0])
        return pattern is not None and re.search(pattern, tree.metadata.get("sent_id", "")) is not None

    return [tree for tree in treebank if not excluded(tree)]


class Treebank:
    def __new__(
        cls,
        lang: str,
        remove_diacritics: bool = False,
        verbose: bool = False,
        load_from_pickle: bool = False,
        resource_dir: str | None = None,
        test_files_only: bool = False,
        # Skip the treebanks listed in languages.udlang2excluded_treebanks.
        exclude_treebanks: bool = True,
        remove_typo: bool = True,
        filter_style: bool = True,
        pickle_path: str = "ud/ud_pickles",
        # Only needed to match UD forms to UniMorph in the original MultiBLiMP
        # pipeline; it also removes the maqaf (U+05BE), which joins words.
        remove_cantillation: bool = False,
        # Remove the sentences listed in languages.excluded_sentences.
        remove_excluded: bool = False,
    ):
        if resource_dir is None:
            resource_dir = "."

        if load_from_pickle:
            pickle_path = os.path.join(resource_dir, pickle_path, f"{lang}.pickle")
            with open(pickle_path, "rb") as f:
                return pickle.load(f)

        if test_files_only:
            treebank_glob = os.path.join(UD_PATH, f"UD_{lang}-*/*test*.conllu")
        else:
            treebank_glob = os.path.join(UD_PATH, f"UD_{lang}-*/*.conllu")
        treebank_glob = os.path.join(resource_dir, treebank_glob)
        treebank_paths = glob(treebank_glob)

        skip_flagged = flag_treebanks("sign language").get((lang if not "_" in lang else lang.split("_")[0]), [])
        treebank_paths = [p for p in treebank_paths if p.split("/")[-2].split("-")[-1].lower() not in skip_flagged]
        if exclude_treebanks:
            excluded_treebanks = udlang2excluded_treebanks.get(lang, [])
            treebank_paths = [
                p for p in treebank_paths if Path(p).parent.name.split("-")[-1] not in excluded_treebanks
            ]

        if verbose:
            print("Loading:\n", "\n".join(treebank_paths))

        treebank = []
        for filename in treebank_paths:
            with open(filename, encoding="utf-8") as f:
                for tree in parse_incr(f):
                    tree.metadata["treebank"] = "/".join(filename.split("/")[-2:])
                    treebank.append(tree)

        if remove_typo:
            treebank = [
                tree
                for tree in treebank
                if not tree_is_malformed(tree, filter_style=filter_style)
            ]

        if remove_excluded:
            treebank = remove_excluded_sentences(treebank)

        if lang in convert_arabic_to_latin_langs:
            for tree in treebank:
                for item in tree:
                    if (item["misc"] or {}).get("Translit"):
                        item["form"] = item["misc"]["Translit"]
                    else:
                        item["form"] = arabic_to_latin(item["form"])

                    if (item["misc"] or {}).get("LTranslit"):
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

        if lang == "Ancient_Hebrew" and remove_cantillation:

            def remove_hebrew_cantillation(text):
                # https://stackoverflow.com/q/44479533/351197
                pattern = r"[\u0591-\u05AF\u05BE\u05C0\u05C3]"
                return re.sub(pattern, "", text)

            for tree in treebank:
                for item in tree:
                    item["form"] = remove_hebrew_cantillation(item["form"])
                    item["lemma"] = remove_hebrew_cantillation(item["lemma"])

        # Done after the form conversions above, so they also apply to the stored
        # multiword token forms.
        for tree in treebank:
            remove_multiword_tokens(tree)

        return treebank
