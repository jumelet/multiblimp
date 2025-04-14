from typing import Any, Dict, Optional

from conllu import TokenList

from .base import UDFilter
from .utils import tokenlist2sen


class DetNounFilter(UDFilter):
    def filter_item(self, tree: TokenList) -> Optional[Dict[str, Any]]:
        for det_idx, child in enumerate(tree):
            if child["deprel"] == "det":
                det = child["form"]
                det_pos = child["upos"]
                noun_idx = child["head"] - 1

                head = tree[noun_idx]
                noun = head["form"]
                noun_pos = head["upos"]

                child_features = child["feats"] or {}
                head_features = head["feats"] or {}
                child_features["lemma"] = child["lemma"].replace("_", "")
                head_features["lemma"] = head["lemma"].replace("_", "")

                if (det_pos == "DET") and (noun_pos == "NOUN"):
                    filtered_item = {
                        "sen": tokenlist2sen(tree),
                        "det": det,
                        "det_idx": det_idx,
                        "noun": noun,
                        "noun_idx": noun_idx,
                        "distance": child["head"] - child["id"],
                        "child_features": child_features,
                        "head_features": head_features,
                    }
                    if self.assert_features(filtered_item):
                        yield filtered_item
