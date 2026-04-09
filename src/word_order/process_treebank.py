import sys
import ast
import os
import random

from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Literal

sys.path.append("src")

from tqdm import tqdm
import pandas as pd
import numpy as np

from multiblimp.treebank import Treebank
from multiblimp.languages import remove_diacritics_langs, gblang2udlang


@dataclass
class PredictionTarget:
    """
    Defines what word order pattern to predict.

    Examples:
        # Single child-head relationship
        PredictionTarget(mode="child_head", child_deprel="amod")

        # Multi-child relationship (e.g., SVO)
        PredictionTarget(
            mode="multi_child",
            child_deprels=["nsubj", "obj"],
            head_deprel="root"  # optional: only consider children of root verbs
        )
    """

    mode: Literal["child_head", "multi_child"]

    # For child_head mode: single deprel to predict
    child_deprel: str | None = None

    # For multi_child mode: list of deprels to order relative to shared head
    child_deprels: list[str] | None = None

    # Optional: filter by head's deprel (e.g., only root verbs for SVO)
    head_deprel: str | None = None

    # Optional: filter by head's POS
    head_pos: str | None = None

    def __post_init__(self):
        if self.mode == "child_head" and self.child_deprel is None:
            raise ValueError("child_head mode requires child_deprel")
        if self.mode == "multi_child" and (
            self.child_deprels is None or len(self.child_deprels) < 2
        ):
            raise ValueError("multi_child mode requires at least 2 child_deprels")


META_FEATURES = ["sen", "treebank", "sent_id", "tree_idx"]


def load_treebank(
    lang: str, resource_dir: str | None = None, max_treebank_len: int | None = None
) -> Treebank:
    lang = gblang2udlang.get(lang, lang).replace(" ", "_")

    treebank = Treebank(
        lang,
        remove_diacritics=(lang in remove_diacritics_langs),
        resource_dir=resource_dir,
        remove_typo=True,
        load_from_pickle=True,
    )
    if max_treebank_len is not None and len(treebank) > max_treebank_len:
        rng = random.Random(42)
        treebank = rng.sample(treebank, max_treebank_len)

    return treebank


def read_df(lang, word_order_dir=None, deprels=None) -> pd.DataFrame:
    lang = gblang2udlang.get(lang, lang).replace(" ", "_")

    deprel_suffix = "" if deprels is None else "_" + "_".join(deprels)

    df = pd.read_csv(
        os.path.join(word_order_dir or "", f"{lang}{deprel_suffix}.csv"),
        low_memory=False,
        converters={"sen": ast.literal_eval},
    )
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.replace([np.inf, -np.inf], "inf")

    return df


def tree2sen(tree):
    return [x["form"] for x in tree]


def get_all_feats(treebank, min_freq=0.005):
    all_feats = Counter()
    all_deprel = set()
    all_lemma_freqs = Counter()
    all_pos = set()
    num_tokens = 0

    for tree in treebank:
        for token in tree:
            all_lemma_freqs[token["lemma"]] += 1
            for feat in token["feats"] or {}:
                all_feats[feat] += 1

            all_deprel.add(token["deprel"])
            all_pos.add(token["upos"])
            num_tokens += 1

    all_feats = Counter(
        {x: c / num_tokens for x, c in all_feats.items() if (c / num_tokens) > min_freq}
    )

    return set(all_feats), all_lemma_freqs, all_deprel, all_pos


def extract_node_features(
    node,
    tree,
    child2root,
    head2child_deprels,
    head2child_lemmas,
    head2left_deprels,
    head2right_deprels,
    head2child_pos,
    head2child_feat,
    prefix: str,
    all_feats: set,
    all_deprel: set,
    all_pos: set,
    target: PredictionTarget,
):
    """
    Extract all features for a specific node (whether it's a child, head, or co-child).
    This ensures consistent feature extraction across all node types.
    """
    features = {}

    # Basic node features
    features[f"{prefix}_deprel"] = node["deprel"]
    features[f"{prefix}_pos"] = node["upos"]
    features[f"{prefix}_form"] = node["form"]
    features[f"{prefix}_lemma"] = node["lemma"]

    # Morphological features
    for feat in all_feats:
        features[f"{prefix}_{feat}"] = (node["feats"] or {}).get(feat)

    # Features about this node's relationship to its head
    if node["head"] != 0:
        head = tree[node["head"] - 1]
        features[f"{prefix}_head_pos"] = head["upos"]
        features[f"{prefix}_head_deprel"] = head["deprel"]
        features[f"{prefix}_dir"] = int(node["id"] > node["head"])

        # Grandhead features
        if head["head"] != 0:
            features[f"{prefix}_grandhead_deprel"] = tree[head["head"] - 1]["deprel"]

    # Path to root features
    for deprel in all_deprel:
        features[f"{prefix}_under_{deprel}"] = deprel in child2root[node["id"]]

    # Sibling features (if node has siblings)
    if node["head"] != 0:
        # Get siblings excluding this node
        def remove_first_occurrence(lst, x):
            result = []
            removed = False
            for item in lst:
                if not removed and item == x:
                    removed = True
                    continue
                result.append(item)
            return result

        sibling_deprels = remove_first_occurrence(
            head2child_deprels[node["head"]], node["deprel"]
        )
        sibling_pos = head2child_pos[node["head"]].copy()

        sibling_deprel_candidates = set(all_deprel) - set(target.child_deprels or [])

        # Sibling deprel presence
        for deprel in sibling_deprel_candidates:
            features[f"{prefix}_sibling-deprel_{deprel}"] = deprel in sibling_deprels

        # Directional sibling deprels
        for sibling_dir in ["L", "R"]:
            if sibling_dir == "L":
                dir_sibling_deprels = head2left_deprels[node["head"]]
                if node["id"] < node["head"]:
                    dir_sibling_deprels = remove_first_occurrence(
                        dir_sibling_deprels, node["deprel"]
                    )
            else:
                dir_sibling_deprels = head2right_deprels[node["head"]]
                if node["id"] > node["head"]:
                    dir_sibling_deprels = remove_first_occurrence(
                        dir_sibling_deprels, node["deprel"]
                    )

            for deprel in sibling_deprel_candidates:
                features[f"{prefix}_sibling-deprel-{sibling_dir}_{deprel}"] = (
                    deprel in dir_sibling_deprels
                )

        # Sibling POS presence
        for pos in all_pos:
            features[f"{prefix}_sibling-pos_{pos}"] = pos in sibling_pos

        # Sibling lemma for specific deprels
        sibling_lemmas = head2child_lemmas[node["head"]]
        for deprel in sibling_deprel_candidates:
            lemma_val = "None"
            for dep, lem in zip(head2child_deprels[node["head"]], sibling_lemmas):
                if dep == deprel and dep != node["deprel"]:
                    lemma_val = lem
                    break
            features[f"{prefix}_sibling-lemma_{deprel}"] = lemma_val

        # Sibling morphological features
        for feat in all_feats:
            for deprel in sibling_deprel_candidates:
                if deprel == node["deprel"]:
                    features[f"{prefix}_sibling-feat_{deprel}_{feat}"] = None
                else:
                    features[f"{prefix}_sibling-feat_{deprel}_{feat}"] = (
                        head2child_feat[node["head"]].get(deprel, {}).get(feat)
                    )

    return features


def extract_sen_features(tree):
    """
    Extract sentence-level features that apply to the whole sentence.

    Returns:
        dict of feature_name -> feature_value
    """
    feature2val = {}

    subj_idx = None
    verb_idx = None
    obj_idx = None

    for token in tree:
        head = token["head"]
        if head == 0:
            continue
        head_is_root = tree[head - 1]["deprel"] == "root"

        if head_is_root:
            if token["deprel"] == "nsubj":
                subj_idx = token["id"]
                verb_idx = head
            elif token["deprel"] == "obj":
                obj_idx = token["id"]

    if subj_idx is None or verb_idx is None:
        core_arg_order = None
    elif obj_idx is None:
        core_arg_order = "sv" if subj_idx < verb_idx else "vs"
    else:
        arg_order = np.argsort([subj_idx, verb_idx, obj_idx])
        core_arg_order = "".join(np.array(["s", "v", "o"])[arg_order])

    feature2val["core_args"] = core_arg_order
    feature2val["subject_idx"] = subj_idx
    feature2val["object_idx"] = obj_idx
    feature2val["verb_idx"] = verb_idx

    # Question detection
    feature2val["in_question"] = tree[-1]["form"] == "?"

    return feature2val


def extract_instances(tree, tree_idx, target: PredictionTarget, tree_metadata):
    """
    Extract training instances from a tree based on the prediction target.

    Returns:
        list of dicts, where each dict contains features for one instance
    """
    instances = []

    # Build tree structure metadata
    child2head = {}
    child2root = defaultdict(set)
    head2child_deprels = defaultdict(list)
    head2child_pos = defaultdict(set)
    head2child_lemmas = defaultdict(list)
    head2child_feats = defaultdict(dict)
    head2left_deprels = defaultdict(list)
    head2right_deprels = defaultdict(list)

    for token in tree:
        child2head[token["id"]] = token["head"]
        head2child_deprels[token["head"]].append(token["deprel"])
        head2child_pos[token["head"]].add(token["upos"])
        head2child_lemmas[token["head"]].append(token["lemma"])

        head2child_feats[token["head"]][token["deprel"]] = {}
        for feat, value in (token["feats"] or {}).items():
            head2child_feats[token["head"]][token["deprel"]][feat] = value

        if token["id"] < token["head"]:
            head2left_deprels[token["head"]].append(token["deprel"])
        else:
            head2right_deprels[token["head"]].append(token["deprel"])

    for child, head in child2head.items():
        child2root[child].add(tree[child - 1]["deprel"])
        while head != 0:
            child2root[child].add(tree[head - 1]["deprel"])
            head = child2head[head]

    all_feats, _, all_deprel, all_pos = tree_metadata

    # Extract sentence-level features (core_args, in_question, etc.)
    sen_features = extract_sen_features(tree)

    if target.mode == "child_head":
        # Original mode: predict order of one child relative to its head
        for token in tree:
            if token["deprel"] != target.child_deprel:
                continue

            head = tree[token["head"] - 1]

            # Optional filters
            if target.head_deprel is not None and head["deprel"] != target.head_deprel:
                continue
            if target.head_pos is not None and head["upos"] != target.head_pos:
                continue

            instance = {
                "sen": tree2sen(tree),
                "treebank": tree.metadata["treebank"].split("/")[0],
                "sent_id": tree.metadata["sent_id"],
                "tree_idx": tree_idx,
                "target_order": int(
                    token["id"] > token["head"]
                ),  # 0 if child before head, 1 if after
            }

            # Extract features for the child
            child_features = extract_node_features(
                token,
                tree,
                child2root,
                head2child_deprels,
                head2child_lemmas,
                head2left_deprels,
                head2right_deprels,
                head2child_pos,
                head2child_feats,
                "child",
                all_feats,
                all_deprel,
                all_pos,
                target,
            )

            # Extract features for the head
            head_features = extract_node_features(
                head,
                tree,
                child2root,
                head2child_deprels,
                head2child_lemmas,
                head2left_deprels,
                head2right_deprels,
                head2child_pos,
                head2child_feats,
                "head",
                all_feats,
                all_deprel,
                all_pos,
                target,
            )

            # Add sentence-level features
            instance.update(sen_features)
            instance.update(child_features)
            instance.update(head_features)
            instances.append(instance)

    elif target.mode == "multi_child":
        # New mode: predict order of multiple children relative to a shared head
        # Group children by head
        head2children = defaultdict(list)
        for token in tree:
            if token["deprel"] in target.child_deprels:
                head2children[token["head"]].append(token)

        # For each head that has at least 2 of the target children
        for head_id, children in head2children.items():
            if len(children) < 2:
                continue

            # Check if we have all required deprels
            child_deprels_present = {c["deprel"] for c in children}
            if not all(
                deprel in child_deprels_present for deprel in target.child_deprels
            ):
                continue

            head = tree[head_id - 1]

            # Optional filters
            if target.head_deprel is not None and head["deprel"] != target.head_deprel:
                continue
            if target.head_pos is not None and head["upos"] != target.head_pos:
                continue

            # Create a mapping from deprel to child for this head
            deprel2child = {c["deprel"]: c for c in children}

            instance = {
                "sen": tree2sen(tree),
                "treebank": tree.metadata["treebank"].split("/")[0],
                "sent_id": tree.metadata["sent_id"],
                "tree_idx": tree_idx,
            }

            # Extract features for the head
            head_features = extract_node_features(
                head,
                tree,
                child2root,
                head2child_deprels,
                head2child_lemmas,
                head2left_deprels,
                head2right_deprels,
                head2child_pos,
                head2child_feats,
                "head",
                all_feats,
                all_deprel,
                all_pos,
                target,
            )
            instance.update(head_features)

            # Extract features for each child type
            for deprel in target.child_deprels:
                if deprel not in deprel2child:
                    continue

                child = deprel2child[deprel]
                # Use deprel as prefix so features are named like "nsubj_pos", "obj_pos", etc.
                child_features = extract_node_features(
                    child,
                    tree,
                    child2root,
                    head2child_deprels,
                    head2child_lemmas,
                    head2left_deprels,
                    head2right_deprels,
                    head2child_pos,
                    head2child_feats,
                    deprel,
                    all_feats,
                    all_deprel,
                    all_pos,
                    target,
                )
                instance.update(child_features)

            # Add sentence-level features
            instance.update(sen_features)

            instances.append(instance)

    return instances


def extract_features(treebank, target: PredictionTarget):
    """
    Extract features from treebank based on prediction target.

    Args:
        treebank: Loaded treebank
        target: PredictionTarget specifying what to predict

    Returns:
        list of dicts, where each dict is one training instance
    """
    all_feats, all_lemma_freqs, all_deprel, all_pos = get_all_feats(treebank)
    tree_metadata = (all_feats, all_lemma_freqs, all_deprel, all_pos)

    all_instances = []

    for tree_idx, tree in tqdm(enumerate(treebank), total=len(treebank)):
        instances = extract_instances(tree, tree_idx, target, tree_metadata)
        all_instances.extend(instances)

    return all_instances


def create_word_order_df(
    lang: str,
    target: PredictionTarget,
    resource_dir: str | None = None,
    save_to: str | None = None,
    max_treebank_len: int | None = None,
) -> pd.DataFrame:
    """
    Create a DataFrame with word order features for a given language and prediction target.

    Args:
        lang: Language code
        target: PredictionTarget specifying what word order to predict
        resource_dir: Directory containing treebank resources
        save_to: Directory to save CSV to (optional)
        max_treebank_len: Maximum number of sentences to process

    Returns:
        DataFrame with extracted features

    Examples:
        # Predict adjective-noun order
        df = create_word_order_df(
            "en",
            PredictionTarget(mode="child_head", child_deprel="amod")
        )

        # Predict SVO order
        df = create_word_order_df(
            "en",
            PredictionTarget(
                mode="multi_child",
                child_deprels=["nsubj", "obj"],
                head_deprel="root"
            )
        )
    """
    lang = gblang2udlang.get(lang, lang).replace(" ", "_")
    treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len)

    all_instances = extract_features(treebank, target)
    df = pd.DataFrame(all_instances)

    if save_to is not None and len(df) > 0:
        # Include target info in filename
        if target.mode == "child_head":
            suffix = f"_{target.child_deprel}"
        else:
            suffix = f"_{'_'.join(target.child_deprels)}"
        df.to_csv(os.path.join(save_to, f"{lang}{suffix}.csv"), index=False)

    return df


# Backward compatibility function
def create_word_order_df_legacy(
    lang: str,
    resource_dir: str | None = None,
    save_to: str | None = None,
    max_treebank_len: int | None = None,
    omit_core_features: bool = False,
    deprels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Legacy function for backward compatibility.
    Creates multiple child-head DataFrames, one for each deprel.
    """
    if deprels is None:
        # Get all deprels from treebank
        treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len)
        _, _, all_deprel, _ = get_all_feats(treebank)
        deprels = list(all_deprel)

    dfs = []
    for deprel in deprels:
        if omit_core_features and ("nsubj" in deprel or "obj" in deprel):
            continue

        target = PredictionTarget(mode="child_head", child_deprel=deprel)
        df = create_word_order_df(
            lang, target, resource_dir, save_to=None, max_treebank_len=max_treebank_len
        )
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    if save_to is not None:
        lang_name = gblang2udlang.get(lang, lang).replace(" ", "_")
        combined_df.to_csv(os.path.join(save_to, f"{lang_name}.csv"), index=False)

    return combined_df
