from typing import Any, Dict, List, Optional, Tuple, Union

from conllu import TokenList

from .base import UDFilter
from .utils import tokenlist2sen


class NsubjFilter(UDFilter):
    def __init__(
        self,
        *args,
        noun_upos: Union[str, List[str]] = ["NOUN"],
        verb_upos: Optional[List[str]] = None,
        set_copula: bool = True,
        transfer_head_child_features: bool = True,
        skip_conjunctions: bool = True,
        head_deprels: List[str] = ["expl", "csubj:outer", "nsubj:outer"],
        ufeat: Optional[str] = None,
        **kwargs
    ):
        super(NsubjFilter, self).__init__(*args, **kwargs)

        self.noun_upos = [noun_upos] if isinstance(noun_upos, str) else noun_upos
        self.verb_upos = verb_upos
        self.set_copula = set_copula
        self.transfer_head_child_features = transfer_head_child_features
        self.skip_conjunctions = skip_conjunctions
        self.head_deprels = head_deprels
        self.ufeat = ufeat

    def filter_item(self, tree: TokenList) -> Optional[Dict[str, Any]]:
        for child_idx, child in enumerate(tree):
            if child["deprel"] == "nsubj":
                sen = tokenlist2sen(tree)
                verb_idx = child["head"] - 1
                head = tree[verb_idx]
                verb = head["form"]
                verb_upos = head["upos"]
                verb_lemma = head["lemma"]
                noun_upos = child["upos"]

                if self.skip_conjunctions and self.has_conjunction(tree, child_idx+1):
                    continue

                if self.has_head_deprels(tree, child["head"]):
                    continue

                cop, cop_idx, cop_features, cop_lemma, cop_upos = self.fetch_copula(
                    tree, child["head"]
                )

                if (
                    (noun_upos not in self.noun_upos)
                    or (verb_upos not in ["VERB", "AUX"])
                    or (self.verb_upos is not None and verb_upos not in self.verb_upos)
                ):
                    continue

                if self.set_copula:
                    main_verb = cop or verb
                    main_verb_idx = cop_idx or verb_idx
                    head_upos = cop_upos or verb_upos
                    head_features = cop_features or head["feats"] or {}
                    head_features["lemma"] = (cop_lemma or verb_lemma).replace("_", "")
                else:
                    main_verb = verb
                    main_verb_idx = verb_idx
                    head_upos = verb_upos
                    head_features = head["feats"] or {}
                    head_features["lemma"] = verb_lemma.replace("_", "")
                noun = child["form"]

                child_features = child["feats"] or {}
                child_features["lemma"] = child["lemma"].replace("_", "")
                keep_item = True

                if self.transfer_head_child_features:
                    for feature in ["Number", "Person", "Gender"]:
                        if (
                            (feature not in head_features)
                            and (feature in child_features)
                        ):
                            head_features[feature] = child_features[feature]
                        elif (feature in head_features) and (feature in child_features):
                            # subj - verb number mismatch indicates mistag!
                            if head_features[feature] != child_features[feature]:
                                if self.verbose:
                                    error_msg = (
                                        f"{sen}\n{main_verb}\n{noun}\n{head_features[feature]}"
                                    )
                                    print(error_msg)
                                keep_item = False

                distance = main_verb_idx - child_idx

                if self.ufeat is not None:
                    if distance > 0:
                        start_idx = child_idx + 1
                        end_idx = main_verb_idx - 1
                    else:
                        start_idx = main_verb_idx + 1
                        end_idx = child_idx - 1

                    attractors = self.fetch_attractors(tree, start_idx, end_idx)
                else:
                    attractors = []

                filtered_item = {
                    "sen": sen,
                    "verb": verb,
                    "verb_idx": verb_idx,
                    "cop": cop,
                    "cop_idx": cop_idx,
                    "child": noun,
                    "child_idx": child_idx,
                    "child_features": child_features,
                    "child_upos": noun_upos,
                    "head": main_verb,
                    "head_idx": main_verb_idx,
                    "head_features": head_features,
                    "head_upos": head_upos,
                    "distance": distance,
                    "attractors": attractors,
                }
                if keep_item and self.assert_features(filtered_item):
                    yield filtered_item

    @staticmethod
    def fetch_copula(
        tree, child_head_idx
    ) -> Tuple[str, Optional[int], Optional[Dict[str, str]], Optional[str], Optional[str]]:
        for cop_idx, cop in enumerate(tree):
            if (cop["head"] == child_head_idx) and (
                cop["deprel"] in ["cop", "aux"] or cop["deprel"].startswith("aux")
            ):
                return cop["form"], cop_idx, cop["feats"], cop["lemma"], cop["upos"]

        return "", None, None, None, None

    @staticmethod
    def has_conjunction(tree, noun_idx: int) -> bool:
        for conj in tree:
            if (conj["deprel"] == "conj") and (conj["head"] == noun_idx):
                return True

        return False

    def has_head_deprels(self, tree, child_head_idx) -> bool:
        for cop_idx, cop in enumerate(tree):
            if cop["head"] == child_head_idx:
                for deprel in self.head_deprels:
                    if deprel in cop["deprel"]:
                        return True

        return False

    def fetch_attractors(self, tree, start_idx, end_idx):
        attractors = []

        for token in tree[start_idx:end_idx]:
            if self.ufeat in (token.get("feats") or {}):
                attractors.append(token)

        return attractors
