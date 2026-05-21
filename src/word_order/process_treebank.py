import sys
import ast
import os
import random

from collections import defaultdict, Counter

sys.path.append("src")

from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

from multiblimp.treebank import Treebank
from multiblimp.languages import remove_diacritics_langs, gblang2udlang

from .prediction_target import PredictionTarget
from .utils import shorten_cls


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
    features[f"{prefix}_idx"] = node["id"]

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

    # Child features (node's own dependents)
    child_deprel_candidates = set(all_deprel) - set(target.child_deprels or [])
    child_deprels = head2child_deprels[node["id"]]
    child_pos = head2child_pos[node["id"]]
    child_lemmas = head2child_lemmas[node["id"]]

    for deprel in child_deprel_candidates:
        features[f"{prefix}_child-deprel_{deprel}"] = deprel in child_deprels

    for pos in all_pos:
        features[f"{prefix}_child-pos_{pos}"] = pos in child_pos

    for deprel in child_deprel_candidates:
        lemma_val = "None"
        for dep, lem in zip(child_deprels, child_lemmas):
            if dep == deprel:
                lemma_val = lem
                break
        features[f"{prefix}_child-lemma_{deprel}"] = lemma_val

    for feat in all_feats:
        for deprel in child_deprel_candidates:
            features[f"{prefix}_child-feat_{deprel}_{feat}"] = (
                head2child_feat[node["id"]].get(deprel, {}).get(feat)
            )

    return features


def extract_sen_features(tree):
    """
    Extract sentence-level features that apply to the whole sentence.
    Sets core argument structure of main clause and `in_question` features.

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

    for child in tree:
        child2head[child["id"]] = child["head"]
        head2child_deprels[child["head"]].append(child["deprel"])
        head2child_pos[child["head"]].add(child["upos"])
        head2child_lemmas[child["head"]].append(child["lemma"])

        head2child_feats[child["head"]][child["deprel"]] = {}
        for feat, value in (child["feats"] or {}).items():
            head2child_feats[child["head"]][child["deprel"]][feat] = value

        if child["id"] < child["head"]:
            head2left_deprels[child["head"]].append(child["deprel"])
        else:
            head2right_deprels[child["head"]].append(child["deprel"])

    for child, head in child2head.items():
        child2root[child].add(tree[child - 1]["deprel"])
        while head != 0:
            child2root[child].add(tree[head - 1]["deprel"])
            head = child2head[head]

    all_feats, _, all_deprel, all_pos = tree_metadata

    # Extract sentence-level features (core_args, in_question, etc.)
    sen_features = extract_sen_features(tree)

    # Group children by head, keeping only those matching target deprels
    head2children = defaultdict(list)
    for child in tree:
        if child["deprel"] in target.child_deprels:
            head2children[child["head"]].append(child)

    # For each head that has all required child deprels present
    for head_id, children in head2children.items():
        # Check if we have all required deprels
        child_deprels_present = {c["deprel"] for c in children}
        if not all(deprel in child_deprels_present for deprel in target.child_deprels):
            continue

        head = tree[head_id - 1]

        # Create a mapping from deprel to child for this head
        deprel2child = {c["deprel"]: c for c in children}

        # Optional filters
        if target.head_deprel is not None and head["deprel"] != target.head_deprel:
            continue
        if target.head_pos is not None and head["upos"] not in target.head_pos:
            continue
        if target.child_pos is not None:
            skip = False
            for deprel, child in deprel2child.items():
                if isinstance(target.child_pos, list):
                    if child["upos"] not in target.child_pos:
                        skip = True
                elif isinstance(target.child_pos, dict):
                    if child["upos"] not in target.child_pos[deprel]:
                        skip = True
            if skip:
                continue

        sen = tree2sen(tree)
        instance = {
            "sen": sen,
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

        # Extract features for each child type; use deprel as prefix (e.g. "nsubj_pos", "amod_pos")
        for deprel in target.child_deprels:
            if deprel not in deprel2child:
                continue

            child = deprel2child[deprel]
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

        deprel_ids = {
            deprel: instance[f"{deprel}_idx"]
            for deprel in target.child_deprels + ["head"]
        }
        deprel_order = "_".join(sorted(deprel_ids, key=deprel_ids.get))
        instance["deprel_order"] = shorten_cls(deprel_order, target)

        if "broedcellen" in sen and "voorraadpotje" in sen:
            print(0, instance["nmod_child-deprel_det"], sen)

        if "bewijzen" in sen and "berusten" in sen:
            print(1, instance["nmod_child-deprel_det"], sen)

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

        if len(instances) > 0 and len(instances) % 100 == 0:
            print(len(instances))

    return all_instances


def create_word_order_df(
    target: PredictionTarget,
    treebank: Treebank | None = None,
    lang: str | None = None,
    resource_dir: str | None = None,
    save_to: str | None = None,
    max_treebank_len: int | None = None,
    drop_singleton_columns: bool = False,
) -> pd.DataFrame:
    """
    Create a DataFrame with word order features for a given language and prediction target.

    Args:
        target: PredictionTarget specifying what word order to predict
        treebank: Treebank for language, can be provided optionally
        lang: Language code, must be provided is Treebank is not passed
        resource_dir: Directory containing treebank resources
        save_to: Directory to save CSV to (optional)
        max_treebank_len: Maximum number of sentences to process
        drop_singleton_columns: drop columns with only a single value

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
    if treebank is None:
        assert lang is not None
        treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len)

    all_instances = extract_features(treebank, target)
    df = pd.DataFrame(all_instances)

    if drop_singleton_columns and len(df) > 0:
        always_keep = {"deprel_order", "sen", "treebank", "sent_id", "tree_idx"}
        cols_to_check = df.columns.difference(list(always_keep))
        keep = df[cols_to_check].nunique() > 1
        kept_always = [c for c in always_keep if c in df.columns]
        df = df[[*kept_always, *keep.index[keep]]].copy()

    if save_to is not None and len(df) > 0:
        output_path = os.path.join(save_to, f"{lang.replace(' ', '_')}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
