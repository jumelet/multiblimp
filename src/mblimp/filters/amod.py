from typing import Any, Dict, Optional

from conllu import TokenList

from .base import UDFilter
from .utils import tokenlist2sen


class AdjNounFilter(UDFilter):
    def filter_item(self, tree: TokenList) -> Optional[Dict[str, Any]]:
        for child_idx, child in enumerate(tree):
            if child["deprel"] == "amod":
                adj = child["form"]
                adj_pos = child["upos"]

                noun_idx = child["head"] - 1
                head = tree[noun_idx]
                noun = head["form"]
                noun_pos = head["upos"]

                child_features = child["feats"] or {}
                head_features = head["feats"] or {}
                child_features["lemma"] = child["lemma"].replace("_", "")
                head_features["lemma"] = head["lemma"].replace("_", "")
                child_features["prev_token"] = tree[child_idx - 1]["form"]

                for feat in ["Number", "Gender"]:
                    if (feat not in child_features) and (feat in head_features):
                        child_features[feat] = head_features[feat]

                if (adj_pos == "ADJ") and (noun_pos == "NOUN"):
                    filtered_item = {
                        "sen": tokenlist2sen(tree),
                        "child": adj,
                        "child_idx": child_idx,
                        "child_features": child_features,
                        "head": noun,
                        "head_idx": noun_idx,
                        "head_features": head_features,
                        "distance": noun_idx - child_idx,
                    }
                    if self.assert_features(filtered_item):
                        yield filtered_item
