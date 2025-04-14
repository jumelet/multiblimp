import os
import pandas as pd
from collections import defaultdict, Counter
from glob import glob
from typing import *

from .languages import lang2langcode, skip_langs
from .treebank import Treebank, has_typo

ud_upos2um_upos = {
    "VERB": "V",
    "NOUN": "N",
    "PRON": "PRON",
    "ADJ": "ADJ",
    "DET": "DET",
    "ADV": "ADV",
    "AUX": "V",  # NB: we map auxiliaries to verbs
}


def is_finite(features):
    return features.get("VerbForm") == "Fin"


def require_mood(features):
    return "Mood" in features


default_values = [
    ("Mood", is_finite, "Ind"),
    ("VerbForm", require_mood, "Fin"),
    ("Voice", is_finite, "Act"),
]


def load_ud_features(path: str):
    with open(path) as f:
        lines = f.read().split("\n")

    feat2val_UD = {
        "upos": [
            "N",
            "PROPN",
            "ADJ",
            "PRO",
            "CLF",
            "ART",
            "DET",
            "V",
            "ADV",
            "AUX",
            "ADP",
            "COMP",
            "CONJ",
            "NUM",
            "PART",
            "INTJ",
            "AJD",  # classical armenian
            "PRE",  # Livvi
            "ADJ.CVB",  # Korean
            "PRON",
        ],
    }
    val2feat_UD = {}
    cur_feat = None

    for idx, line in enumerate(lines):
        if line == lines[idx - 1]:
            cur_feat = line
            feat2val_UD[cur_feat] = []
        if ":" in line:
            val = line.split(":")[0]
            val = f"{cur_feat}_{val}"
            feat2val_UD[cur_feat].append(val)

    for feat, vals in feat2val_UD.items():
        for val in vals:
            val2feat_UD[val] = feat

    return feat2val_UD, val2feat_UD


def create_all_unimorph_from_ud(
    ud_langs: List[str],
    verbose=False,
    resource_dir=".",
    dup_form_threshold=100.0,
    dup_feat_threshold: float = 100.0,
    load_from_pickle=False,
):
    ud_feature_path = os.path.join(resource_dir, "ud/ud_feats.txt")
    _, val2feat_UD = load_ud_features(ud_feature_path)

    for lang in sorted(ud_langs):
        if lang in skip_langs:
            continue

        isolang = lang2langcode(lang)
        um_file = os.path.join(resource_dir, f"ud_unimorph/{isolang}")

        ud_lang = lang.replace(" ", "_")
        treebank = Treebank(
            ud_lang,
            remove_diacritics=False,
            load_from_pickle=load_from_pickle,
            resource_dir=resource_dir,
            remove_typo=False,
            pickle_path="ud/ud_typo_pickles",
        )

        df = create_unimorph_from_ud(
            treebank,
            val2feat_UD,
            file=um_file,
            skip_no_lemma=True,
            skip_prep_lemma=True,
            verbose=verbose,
            dup_form_threshold=dup_form_threshold,
            dup_feat_threshold=dup_feat_threshold,
        )
        print(lang, len(df.lemma.unique()), len(df.form.unique()), len(df), sep="\t")


def create_unimorph_from_ud(
    treebank: Treebank,
    val2feat_UD: Dict[str, str],
    file: Optional[str] = None,
    skip_no_lemma: bool = False,
    skip_prep_lemma: bool = False,
    verbose: bool = False,
    dup_form_threshold: float = 100.0,
    dup_feat_threshold: float = 100.0,
):
    rows = []

    iterator = tqdm_notebook(treebank) if verbose else treebank

    for tree in iterator:
        for token in tree:
            form = token["form"]
            upos = token["upos"]
            lemma = token["lemma"]

            if skip_no_lemma and lemma == "_":
                continue

            if skip_prep_lemma and "_" in lemma:
                continue

            if has_typo(token):
                continue

            if upos in ud_upos2um_upos:
                token_feats = token["feats"] or {}
                if len(token_feats) == 0:
                    continue

                upos = ud_upos2um_upos[upos]
                ufeats: str = upos

                for feat, condition, val in default_values:
                    if (feat not in token_feats) and condition(token_feats):
                        token_feats[feat] = val

                for feat, val in sorted(token_feats.items()):
                    if f"{feat}_{val}" in val2feat_UD or "," in val:
                        ufeats += ";" + f"{feat}_{val}"
                    else:
                        print("Feature value not found:", form, upos, feat, val)

                row = (lemma.lower(), form.lower(), ufeats)
                rows.append(row)

    row_freqs = [(*row, freq) for row, freq in Counter(rows).items()]

    row_freqs = remove_duplicate_features(row_freqs, dup_form_threshold, dup_feat_threshold)

    df = pd.DataFrame(
        sorted(row_freqs), columns=["lemma", "form", "ufeats", "frequency"]
    )

    if (file is not None) and (len(df) > 0):
        df.to_csv(file, sep="\t", index=False)

    return df


def remove_duplicate_features(
    row_freqs: List[Tuple[str, str, str, int]],
    dup_form_threshold: float,
    dup_feat_threshold: float,
) -> List[Tuple[str, str, str, int]]:
    """
    Form mismatch might indicate annotation error, since we can expect
    a lemma+feat combination to usually map to a single form
    (lemma1, form1, feat1)  <> (lemma1, form2, feat1)

    Feature mismatch might indicate feature annotation error, but here
    we only do this if the less frequent triple occurs once. Different
    lemma+form combinations can be instantiations of different feature
    sets, e.g. I saw_PRS / I saw_PST / you saw_PST
    (lemma1, form1, feat1)  <> (lemma1, form1, feat2)
    """
    lemma_feat2form = defaultdict(list)
    lemma_form2feat = defaultdict(list)

    for lemma, form, features, freq in row_freqs:
        lemma_feat2form[lemma, features].append((form, freq))
        lemma_form2feat[lemma, form].append((features, freq))

    dedup_forms = []
    for (lemma, features), forms in lemma_feat2form.items():
        if len(forms) == 1:
            form, freq = forms[0]
            dedup_forms.append((lemma, form, features, freq))
        else:
            sorted_forms: List[str, int] = sorted(forms, key=lambda x: x[1], reverse=True)
            most_common_form = sorted_forms[0]
            # If the most frequent form is _n_ times more frequent we remove that item
            dedup_forms.append(
                (lemma, most_common_form[0], features, most_common_form[1])
            )
            for next_common_form in sorted_forms[1:]:
                if (most_common_form[1] / next_common_form[1]) <= dup_form_threshold:
                    dedup_forms.append(
                        (lemma, next_common_form[0], features, next_common_form[1])
                    )

    dedup_feats = []
    for (lemma, form), all_features in lemma_form2feat.items():
        if len(all_features) == 1:
            features, freq = all_features[0]
            dedup_feats.append((lemma, form, features, freq))
        else:
            sorted_feats: List[str, int] = sorted(all_features, key=lambda x: x[1], reverse=True)
            most_common_feat = sorted_feats[0]
            # If the most frequent feature is _n_ times more frequent we remove that item
            dedup_feats.append(
                (lemma, form, most_common_feat[0], most_common_feat[1])
            )
            for next_common_feat in sorted_feats[1:]:
                most_common_feat_set = set(most_common_feat[0].split(";"))
                next_common_feat_set = set(next_common_feat[0].split(";"))
                mismatching_feats = set(next_common_feat_set).symmetric_difference(most_common_feat_set)

                if (
                    ((most_common_feat[1] / next_common_feat[1]) <= dup_feat_threshold)
                    or (len(mismatching_feats) > 2)
                    or (next_common_feat[1] > 1)
                    or (most_common_feat[0][0] != next_common_feat[0][0])
                ):
                    dedup_feats.append(
                        (lemma, form, next_common_feat[0], next_common_feat[1])
                    )

    dedup_rows = list(set(dedup_forms) & set(dedup_feats))

    return dedup_rows
