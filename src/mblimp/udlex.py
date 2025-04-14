import os
from collections import defaultdict
from glob import glob
from typing import *

import pandas as pd

UDLEX2UM = {
    "Plur": "PL",
    "Sing": "SG",
    "Pres": "PRS",
    "Past": "PST",
    "Sub": "SBJV",
    "Cnd": "COND",
    "Perf": "PRF",
    "Dual": "DU",
}
UM2UDLEX = {v: k for k, v in UDLEX2UM.items()}

UM2UDLEX_POS = {
    "V": ["VERB", "AUX"],
    "N": ["NOUN"],
    "ADJ": ["ADJ"],
}


def featStringToDict(featString):
    """Turns a UD feature string into a dictionary.

    If the featString is '_', returns an empty dictionary.
    If verbose, print if a k,v pair can't be parsed.

    :param featString: string of features, e.g. "Number=Sing|Case=Nom"
    :return: dictionary of features, e.g. {"Number":"Sing","Case":"Nom"}
    """
    if isinstance(featString, float):
        featString = str(float)
    if featString == "_":
        return dict()
    split = featString.split("|")
    if len(split) == 0:
        return dict()
    for s in split:
        if len(s.split("=")) != 2:
            return dict()
    result = {k.split("=")[0]: k.split("=")[1] for k in split}
    return result


class Lexicon:
    def __init__(
        self,
        lexfiles: List[str],
        remove_no_lemma=False,
        lex_pos: Optional[List[str]] = None,
    ):
        self.remove_no_lemma = remove_no_lemma
        self.lex_pos = lex_pos
        self.ufeats = set()

        self.lex = self.init_lexicon(lexfiles)

    def __len__(self):
        return len(self.lex)

    def init_lexicon(self, lexfiles: List[str]) -> pd.DataFrame:
        lex_dfs = []

        for lexfile in lexfiles:
            cols = ["i", "j", "form", "lemma", "upos", "cpos", "ufeat", "something"]
            lex = pd.read_csv(
                lexfile, delimiter="\t", names=cols, error_bad_lines=False
            )
            lex = lex.drop(lex[~lex.i.isin([0, "0"])].index)
            lex = lex.drop(["i", "j", "something"], axis=1)

            lex_dfs.append(lex)

        lex = pd.concat(lex_dfs, ignore_index=True).drop_duplicates()

        if self.lex_pos is not None:
            lex = lex[lex.upos.isin(self.lex_pos)]

        if self.remove_no_lemma:
            lex = lex[lex.lemma != "_"]

        lex["ufeatdict"] = lex.apply(lambda r: featStringToDict(r.ufeat), axis=1)

        self.ufeats = {ufeat for row_ufeats in lex.ufeatdict for ufeat in row_ufeats}

        for ufeat in self.ufeats:
            lex[ufeat] = [row_ufeats.get(ufeat) for row_ufeats in lex.ufeatdict]

        for column in lex.columns:
            if isinstance(lex[column].iloc[0], str):
                lex[column] = lex[column].astype("category")

        return lex


class UDLexInflector:
    def __init__(
        self,
        lang: str,
        upos: str,
        swap_feature_map: Tuple[str, Dict[str, str]],
        ignore_ufeats: Set[str] = {},
        verbose: bool = False,
        resource_dir: str = None,
    ):
        resource_dir = resource_dir or ""
        lexfiles = glob(os.path.join(resource_dir, f"udlex/*{lang}*"))
        lex_pos = UM2UDLEX_POS.get(upos)
        lexicon = Lexicon(lexfiles, remove_no_lemma=True, lex_pos=lex_pos)

        self.lexicon = lexicon
        self.swap_feature_map = swap_feature_map
        self.verbose = verbose

        self.feature_dicts: Dict[str, defaultdict[str, Set[str]]] = {}

        self.ufeats: List[str] = self.lexicon.ufeats
        self.ufeats.add("lemma")
        self.ufeats.difference_update(ignore_ufeats)

        self.init_feature_dicts(self.lexicon.lex)

    def init_feature_dicts(self, lex: pd.DataFrame) -> None:
        self.feature_dicts = {ufeat: defaultdict(set) for ufeat in self.ufeats}

        for ufeat, feature_dict in self.feature_dicts.items():
            for form, feature in zip(lex.form, lex[ufeat]):
                if feature != "_" and isinstance(feature, str):
                    feature_dict[form].add(feature)

    def form2feature(self, form, ufeat) -> Union[str, Set[str]]:
        feature_dict = self.feature_dicts[ufeat]
        feature = feature_dict[form]

        # if form feature is not ambiguous return immediately
        if len(feature) == 1:
            return next(iter(feature))

        return feature

    def form2features(self, form, features) -> Dict[str, str]:
        return {
            ufeat: features.get(ufeat) or self.form2feature(form, ufeat)
            for ufeat in self.feature_dicts.keys()
        }

    def is_candidate(self, features: Dict[str, str], swap_ufeat: str):
        # We always require a single lemma to map onto
        # + We need a single feature that will be swapped
        return isinstance(features["lemma"], str) and isinstance(
            features[swap_ufeat], str
        )

    def lemma2form(
        self, features: Dict[str, str], strat: Optional[Dict[str, str]] = None
    ):
        strat = strat or {}

        mask = self.lexicon.lex.lemma == features["lemma"]
        features.update(strat)

        for ufeat, feature in features.items():
            if isinstance(feature, str) and ufeat.startswith("neg"):
                ufeat = ufeat.split("_")[1]
                mask &= self.lexicon.lex[ufeat] != feature
            elif isinstance(feature, str) and (f"neg_{ufeat}" not in features):
                mask &= self.lexicon.lex[ufeat] == feature

        forms = self.lexicon.lex.form[mask].unique()

        if self.verbose:
            print(strat)
            print(features, list(forms), "\n")

        return forms

    def inflect(self, form, features, strategies: List[Dict[str, str]] = []):
        swap_ufeat, feature_map = self.swap_feature_map
        strategies = strategies or [{}]
        features = features or {}

        features = self.form2features(form, features)
        is_candidate = self.is_candidate(features, swap_ufeat)

        if not is_candidate:
            if self.verbose:
                print(features["lemma"], features[swap_ufeat])
            return None

        current_feature = UDLEX2UM[features[swap_ufeat]]
        opp_feature = UM2UDLEX[feature_map[current_feature]]
        swap_features = features
        swap_features[swap_ufeat] = opp_feature

        for strat in strategies:
            swap_form = self.lemma2form(dict(swap_features), strat)

            if len(swap_form) == 1:
                return list(swap_form)

        return swap_form
