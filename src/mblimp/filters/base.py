from typing import *

from conllu import TokenList
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook

from ..treebank import Treebank


class UDFilter:
    def __init__(
        self,
        treebank: Treebank,
        head_features: Dict[str, str] = {},
        child_features: Dict[str, str] = {},
        max_sen_len: int = 64,
        verbose: bool = False,
        **kwargs,
    ):
        self.treebank = treebank
        self.head_features = head_features
        self.child_features = child_features
        self.max_sen_len = max_sen_len
        self.verbose = verbose

        self.filtered_items = []

    def filter_treebank(self, verbose=False):
        iterator = enumerate(self.treebank)
        iterator = tqdm_notebook(iterator) if verbose else iterator

        for tree_idx, tree in iterator:
            if len(tree) > self.max_sen_len:
                continue

            for filtered_item in self.filter_item(tree):
                filtered_item["tree"] = tree
                filtered_item["metadata"] = tree.metadata
                self.filtered_items.append((tree_idx, filtered_item))

        if verbose:
            print(
                f"{len(self.filtered_items)} out of {len(self.treebank)} have been filtered"
            )

    def filter_item(self, tree: TokenList) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def plot_distances(self, max_dist=10e8, min_dist=-10e8) -> None:
        distances = [
            item["distance"]
            for _, item in self.filtered_items
            if max_dist > item["distance"] > min_dist
        ]
        plt.hist(distances, bins=range(min(distances), max(distances) + 2))
        plt.show()

        return distances

    def assert_features(self, filtered_item: Dict[str, str]) -> bool:
        for feat, feat_val in self.child_features.items():
            if feat_val.startswith('-'):
                feat_val = feat_val[1:]
                child_feat = filtered_item["child_features"].get(feat)
                if child_feat == feat_val:
                    return False
            else:
                child_feat = filtered_item["child_features"].get(feat)
                if child_feat != feat_val:
                    return False

        for feat, feat_val in self.head_features.items():
            if feat_val.startswith('-'):
                feat_val = feat_val[1:]
                head_feat = filtered_item["head_features"].get(feat)
                if head_feat == feat_val:
                    return False
            else:
                head_feat = filtered_item["head_features"].get(feat)
                if head_feat != feat_val:
                    return False

        return True
