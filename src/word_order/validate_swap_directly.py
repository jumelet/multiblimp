import os
import random as _random
from copy import deepcopy
from dataclasses import dataclass
import conllu
import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .decision_tree import fit_dt, _infer_column_types, _fill_missing
from .process_treebank import (
    extract_node_features,
    get_all_feats,
    _build_tree_maps,
    records_to_df,
)
from .tree_utils import deserialize_tree


def find_sublist_kmp(sub, lst):
    """Returns list of start indices (0-based) where sub occurs in lst."""
    if not sub:
        return [0]

    # Build failure table
    fail = [0] * len(sub)
    j = 0
    for i in range(1, len(sub)):
        while j > 0 and sub[i] != sub[j]:
            j = fail[j - 1]
        if sub[i] == sub[j]:
            j += 1
        fail[i] = j

    # Search
    matches = []
    j = 0
    for i in range(len(lst)):
        while j > 0 and lst[i] != sub[j]:
            j = fail[j - 1]
        if lst[i] == sub[j]:
            j += 1
        if j == len(sub):
            matches.append(i - len(sub) + 1)
            j = fail[j - 1]
    return matches


def pos_seq_deprels(treebank: list[conllu.models.TokenList], pos_seq: list[str]):
    """Returns all treebank trees that contain part-of-speech sequence pos_seq"""
    records = []

    for tree_idx, tree in enumerate(treebank):
        tokens = [t for t in tree if isinstance(t["id"], int)]
        tree_pos = [t["upos"] for t in tokens]

        match_starts = find_sublist_kmp(pos_seq, tree_pos)
        if not match_starts:
            continue

        # Build children map for subtree validation
        children = {t["id"]: [] for t in tokens}
        for t in tokens:
            if t["head"] and t["head"] in children:
                children[t["head"]].append(t["id"])

        sent_id = tree.metadata.get("sent_id") if tree.metadata else None
        token_by_id = {t["id"]: t for t in tokens}

        for start in match_starts:
            span_tokens = [tokens[start + i] for i in range(len(pos_seq))]
            span_ids = {t["id"] for t in span_tokens}

            # The subtree root is the unique token whose head falls outside the span
            span_roots = [t for t in span_tokens if t["head"] not in span_ids]
            if len(span_roots) != 1:
                continue

            span_root = span_roots[0]

            # Verify the full subtree rooted here equals the span exactly
            subtree_ids = set()
            queue = [span_root["id"]]
            while queue:
                nid = queue.pop()
                subtree_ids.add(nid)
                queue.extend(children[nid])

            if subtree_ids != span_ids:
                continue

            head_token = token_by_id.get(span_root["head"])
            subtree = " ".join(t["form"] for t in span_tokens)
            sen = [t["form"] for t in tokens]

            records.append(
                {
                    "deprel": span_root["deprel"],
                    "subtree_form": subtree,
                    "root_form": span_root["form"],
                    "root_pos": span_root["upos"],
                    "head_form": head_token["form"] if head_token else None,
                    "head_pos": head_token["upos"] if head_token else None,
                    "subtree_idx": span_root["id"],
                    "head_idx": span_root["head"],
                    "dir": 0 if span_root["id"] < span_root["head"] else 1,
                    "sen": sen,
                    "tree_idx": tree_idx,
                    "sent_id": sent_id,
                }
            )

    return pd.DataFrame(records)


def get_head_candidates(tree, idx, include_true_head=False):
    tokens = [t for t in tree if isinstance(t["id"], int)]

    # Collect descendants of idx to exclude: attaching idx to a descendant creates a cycle.
    children_map: dict[int, list[int]] = {t["id"]: [] for t in tokens}
    for t in tokens:
        if isinstance(t["head"], int) and t["head"] in children_map:
            children_map[t["head"]].append(t["id"])
    descendants: set[int] = set()
    queue = list(children_map.get(idx, []))
    while queue:
        nid = queue.pop()
        descendants.add(nid)
        queue.extend(children_map.get(nid, []))

    arcs = [
        (t["id"], t["head"])
        for t in tokens
        if t["id"] != idx and isinstance(t["head"], int)
        # and t['head'] > 0   # uncomment if root arc can be crossed
    ]

    def crosses(a, b, c, d):
        lo1, hi1 = min(a, b), max(a, b)
        lo2, hi2 = min(c, d), max(c, d)
        return lo1 < lo2 < hi1 < hi2 or lo2 < lo1 < hi2 < hi1

    head_candidates = [
        t["id"]
        for t in tokens
        if t["id"] != idx
        and t["id"] not in descendants
        and not any(crosses(idx, t["id"], dep, head) for dep, head in arcs)
    ]

    if include_true_head:
        true_head = tokens[idx - 1]["head"]

        if true_head not in head_candidates:
            head_candidates.append(true_head)

    return head_candidates


@dataclass
class JointClassifier:
    """
    A single decision tree per language that predicts whether a (head, candidate-child)
    pair corresponds to an actual attachment ("R"/"L") or not ("N").
    """

    model: Pipeline
    model_type: str
    all_feats: set
    all_deprel: set
    all_pos: set
    lexicalize: bool = False


def collect_training_data(
    treebank,
    n_samples: int,
    all_feats: set,
    all_deprel: set,
    all_pos: set,
    random_state: int = 42,
    lexicalize: bool = False,
    sample_from_same_head: bool = False,
) -> pd.DataFrame:
    """
    Sample n_samples balanced (positive, negative) pairs from the treebank.

    Each iteration samples a random tree and a random head node h, then:
      - positive: one actual child of h → label "{deprel}-L" or "{deprel}-R"
      - negative: one non-child that could projectivity-validly attach to h,
                  with features extracted from a hypothetical tree where that
                  node's head is set to h → label "N"

    Heads with no valid negative candidates (every non-child would cross an arc
    or create a cycle) are skipped.  Returns a DataFrame with 2*n_samples rows.
    """
    rng = _random.Random(random_state)

    # Cap n_samples to the number of unique (tree_idx, head_id) pairs that have children.
    max_valid_heads = sum(
        len({t["head"] for t in tree if isinstance(t["id"], int) and t["head"]})
        for tree in treebank
    )
    if n_samples > max_valid_heads:
        n_samples = max_valid_heads

    records = []
    attempts = 0
    max_attempts = n_samples * 20
    sampled_pairs: set[tuple[int, int]] = set()

    with tqdm(total=n_samples, desc="Collecting training pairs") as pbar:
        while len(records) < 2 * n_samples and attempts < max_attempts:
            attempts += 1

            tree_idx = rng.randrange(len(treebank))
            tree = treebank[tree_idx]

            pos_tree = deepcopy(tree)
            neg_tree = deepcopy(tree)

            tokens = [t for t in pos_tree if isinstance(t["id"], int)]
            head_choices = tokens  # [t for t in tokens if t["upos"] == "ADJ"]

            if not head_choices:
                continue

            head_token = rng.choice(head_choices)
            h_id = head_token["id"]

            if (tree_idx, h_id) in sampled_pairs:
                continue
            sampled_pairs.add((tree_idx, h_id))

            children = [t for t in tokens if t["head"] == h_id]
            if not children:
                continue

            pos_child = rng.choice(children)

            if sample_from_same_head:
                neg_h_id = h_id
                neg_head_token = next(t for t in neg_tree if t["id"] == h_id)
            else:
                neg_head_token = rng.choice(
                    [t for t in neg_tree if isinstance(t["id"], int)]
                )
                neg_h_id = neg_head_token["id"]

            neg_candidates = [
                t
                for t in neg_tree
                if t["id"] != neg_h_id
                and t["head"] != neg_h_id
                and neg_h_id in get_head_candidates(neg_tree, t["id"])
            ]
            if not neg_candidates:
                continue

            neg_child = rng.choice(neg_candidates)

            orig_pos_deprel = pos_child["deprel"]
            pos_child["deprel"] = "UNK"

            tree_maps = _build_tree_maps(pos_tree)

            # Positive: child is genuinely attached to h; use actual tree maps.
            pos_head_feats = extract_node_features(
                head_token,
                pos_tree,
                tree_maps,
                "head",
                all_feats,
                all_deprel,
                all_pos,
                lexicalize=lexicalize,
                encode_positional_features=True,
                excluded_deprels={pos_child["deprel"]},
            )
            pos_child_feats = extract_node_features(
                pos_child,
                pos_tree,
                tree_maps,
                "child",
                all_feats,
                all_deprel,
                all_pos,
                lexicalize=lexicalize,
                encode_positional_features=True,
            )

            # Negative: build a hypothetical tree where neg_child attaches to neg_h_id
            # so that sibling/head-relative features reflect the hypothetical attachment.
            for t in neg_tree:
                if t["id"] == neg_child["id"]:
                    t["head"] = neg_h_id
                    break

            hyp_neg_token = next(t for t in neg_tree if t["id"] == neg_child["id"])
            orig_neg_deprel = hyp_neg_token["deprel"]
            hyp_neg_token["deprel"] = "UNK"

            hyp_maps = _build_tree_maps(neg_tree)

            neg_head_feats = extract_node_features(
                neg_head_token,
                neg_tree,
                hyp_maps,
                "head",
                all_feats,
                all_deprel,
                all_pos,
                lexicalize=lexicalize,
                encode_positional_features=True,
                excluded_deprels={pos_child["deprel"]},
            )
            neg_child_feats = extract_node_features(
                hyp_neg_token,
                neg_tree,
                hyp_maps,
                "child",
                all_feats,
                all_deprel,
                all_pos,
                lexicalize=lexicalize,
                encode_positional_features=True,
            )

            meta = {
                "tree_idx": tree_idx,
                "sent_id": tree.metadata.get("sent_id", "") if tree.metadata else "",
                "treebank": (
                    (tree.metadata.get("treebank", "") or "").split("/")[0]
                    if tree.metadata
                    else ""
                ),
                "sen": [t["form"] for t in tokens],
            }
            records.append(
                {
                    **meta,
                    "head_form": head_token["form"],
                    "child_form": pos_child["form"],
                    "orig_deprel": orig_pos_deprel,
                    **pos_head_feats,
                    **pos_child_feats,
                    "label": "L" if pos_child["id"] < pos_child["head"] else "R",
                }
            )
            records.append(
                {
                    **meta,
                    "head_form": neg_head_token["form"],
                    "child_form": neg_child["form"],
                    "orig_deprel": orig_neg_deprel,
                    **neg_head_feats,
                    **neg_child_feats,
                    "label": "N",
                }
            )
            pbar.update(1)

    return records_to_df(records)


def fit_joint_classifier(
    treebank,
    n_samples: int,
    model_type: str,
    train_df: pd.DataFrame | None = None,
    lexicalize: bool = False,
    random_state: int = 42,
    verbose: bool = False,
    viz_dir: str | None = None,
    sample_from_same_head: bool = False,
    **pipeline_kwargs,
) -> tuple[JointClassifier | None, pd.DataFrame]:
    """
    Fit a single joint decision tree classifier for reattachment validation.

    The model predicts "L/R" (attaches left/right) or "N" (does-not-exist) for a
    (head, candidate-child) pair.  Training data is collected via
    collect_training_data with 50/50 positive/negative balance.
    """
    all_feats, _, all_deprel, all_pos = get_all_feats(treebank)

    if train_df is None:
        train_df = collect_training_data(
            treebank,
            n_samples,
            all_feats,
            all_deprel,
            all_pos,
            random_state=random_state,
            lexicalize=lexicalize,
            sample_from_same_head=sample_from_same_head,
        )

    extra_omit = {"orig_deprel", "dir_label"}
    if not lexicalize:
        extra_omit |= {"child_form", "head_form"}

    model, X_train, _ = fit_dt(
        train_df,
        model_type,
        target=None,
        predictor_var="label",
        omit_feats=extra_omit,
        verbose=int(verbose),
        **pipeline_kwargs,
    )
    if model is None:
        return None, train_df

    if viz_dir is not None and model_type == "decision_tree":
        from .viz_tree import tree2html

        os.makedirs(viz_dir, exist_ok=True)
        tree2html(
            model,
            X_train,
            train_df,
            predictor_var="label",
            out_file=os.path.join(viz_dir, "joint.html"),
            extra_columns=["orig_deprel", "child_form", "head_form"],
            meta={"model": "joint"},
        )

    classifier = JointClassifier(
        model=model,
        model_type=model_type,
        all_feats=all_feats,
        all_deprel=all_deprel,
        all_pos=all_pos,
        lexicalize=lexicalize,
    )

    return classifier, train_df


def _scoring_df(rows: list[dict] | dict, model: Pipeline) -> pd.DataFrame:
    """
    Build a DataFrame from one or more feature dicts, aligned to the columns the
    model was trained on, with NaN filled the same way as during training.
    """
    if isinstance(rows, dict):
        rows = [rows]
    expected_cols = model.named_steps["preprocessor"].feature_names_in_
    df = pd.DataFrame(rows).reindex(columns=expected_cols)
    categorical_cols, numeric_cols = _infer_column_types(list(expected_cols))
    df = _fill_missing(df, categorical_cols, numeric_cols)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


@dataclass
class HeadData:
    candidates: list[int]
    feature_rows: list[dict]
    directions: list[str]
    hyp_maps: list
    child_feats: dict


def _collect_head_features(
    tree,
    classifier: JointClassifier,
    idx: int,
    include_true_head: bool = False,
) -> HeadData:
    """
    Build per-candidate feature dicts for token `idx` in `tree` without running
    the classifier.  Returns an empty HeadData if there are no valid candidates.
    """
    candidates = get_head_candidates(tree, idx, include_true_head=include_true_head)
    if not candidates:
        return HeadData([], [], [], [], {})

    feature_rows = []
    all_hyp_maps = []
    directions = []
    child_feats: dict = {}

    for h_id in candidates:
        hyp_tree = deepcopy(tree)
        node_by_id = {t["id"]: t for t in hyp_tree if isinstance(t["id"], int)}

        child_token = node_by_id[idx]

        if h_id not in node_by_id:
            print(node_by_id)
            print(candidates, idx, tree)

        head_token = node_by_id[h_id]

        child_token["head"] = h_id
        child_token["deprel"] = "UNK"

        hyp_maps = _build_tree_maps(hyp_tree)

        head_feats = extract_node_features(
            head_token,
            hyp_tree,
            hyp_maps,
            "head",
            classifier.all_feats,
            classifier.all_deprel,
            classifier.all_pos,
            lexicalize=classifier.lexicalize,
            encode_positional_features=True,
        )
        child_feats = extract_node_features(
            child_token,
            hyp_tree,
            hyp_maps,
            "child",
            classifier.all_feats,
            classifier.all_deprel,
            classifier.all_pos,
            lexicalize=classifier.lexicalize,
            encode_positional_features=True,
        )

        feature_rows.append({**head_feats, **child_feats})
        all_hyp_maps.append(hyp_maps)
        directions.append("L" if idx < h_id else "R")

    return HeadData(candidates, feature_rows, directions, all_hyp_maps, child_feats)


def _scores_from_batch(
    data: HeadData,
    proba_matrix,
    leaf_ids,
    offset: int,
    classes: list,
    is_decision_tree: bool,
) -> tuple[dict, dict]:
    """Unpack scores and leaf ids for one token from a slice of a flat proba matrix."""
    results: dict = {}
    leaves: dict = {}
    for i, (h_id, direction) in enumerate(zip(data.candidates, data.directions)):
        prob = (
            float(proba_matrix[offset + i][classes.index(direction)])
            if direction in classes
            else 0.0
        )
        results[h_id] = prob
        leaves[h_id] = int(leaf_ids[offset + i]) if is_decision_tree else None
    return results, leaves


def score_heads(
    tree,
    classifier: JointClassifier,
    idx: int,
    include_true_head: bool = False,
):
    """
    For token `idx` in `tree`, return (head_id → prob, head_id → leaf, ...) for each
    projectivity-valid candidate head.  All candidates are scored in a single
    batched forward pass through the classifier.
    """
    data = _collect_head_features(tree, classifier, idx, include_true_head)
    if not data.candidates:
        return {}, {}, [], [], {}

    batch_df = _scoring_df(data.feature_rows, classifier.model)
    classes = list(classifier.model.named_steps["clf"].classes_)
    proba_matrix = classifier.model.predict_proba(batch_df)

    is_dt = classifier.model_type == "decision_tree"
    if is_dt:
        X_trans = classifier.model.named_steps["preprocessor"].transform(batch_df)
        leaf_ids = classifier.model.named_steps["clf"].apply(X_trans)
    else:
        leaf_ids = None

    results, leaves = _scores_from_batch(
        data, proba_matrix, leaf_ids, 0, classes, is_dt
    )
    all_df = [batch_df.iloc[[i]] for i in range(len(data.candidates))]
    return results, leaves, all_df, data.hyp_maps, data.child_feats


def score_swap_df(
    swap_df: pd.DataFrame,
    classifier: JointClassifier,
) -> pd.DataFrame:
    """
    Score all rows of swap_df (as returned by create_pairs_tree) in two phases:
      1. Collect features for every (original, swapped) tree pair (no model calls).
      2. Score the entire dataset in a single batched forward pass.

    Rows where the head/verb itself was the moved constituent (head_was_moved=True)
    are skipped, since the moved root is then the head rather than a child and the
    attachment validation is not meaningful; their score columns are set to None.

    Returns a copy of swap_df with four new columns:
      attachment_probs, head_prob, swapped_head_prob, max_swapped_head_prob.
    """
    # Phase 1: feature extraction (skip head-moving swaps)
    all_orig_data: list[HeadData | None] = []
    all_swap_data: list[HeadData | None] = []

    for _, row in tqdm(
        swap_df.iterrows(), total=len(swap_df), desc="Extracting features"
    ):
        if row.get("head_was_moved", False):
            all_orig_data.append(None)
            all_swap_data.append(None)
            continue
        orig_tree = deserialize_tree(row.original_tree)
        swap_tree = deserialize_tree(row.swapped_tree)
        all_orig_data.append(
            _collect_head_features(
                orig_tree, classifier, row.original_child_id, include_true_head=True
            )
        )
        all_swap_data.append(
            _collect_head_features(swap_tree, classifier, row.swapped_child_id)
        )

    # Phase 2: single batched forward pass
    all_feature_rows: list[dict] = []
    for orig_d, swap_d in zip(all_orig_data, all_swap_data):
        if orig_d is None or swap_d is None:
            continue
        all_feature_rows.extend(orig_d.feature_rows)
        all_feature_rows.extend(swap_d.feature_rows)

    batch_df = _scoring_df(all_feature_rows, classifier.model)
    classes = list(classifier.model.named_steps["clf"].classes_)
    proba_matrix = classifier.model.predict_proba(batch_df)

    is_dt = classifier.model_type == "decision_tree"
    if is_dt:
        X_trans = classifier.model.named_steps["preprocessor"].transform(batch_df)
        leaf_ids = classifier.model.named_steps["clf"].apply(X_trans)
    else:
        leaf_ids = None

    # Phase 3: unpack results per row
    all_attachment_probs = []
    head_probs = []
    swapped_head_probs = []
    max_swapped_head_probs = []

    offset = 0
    for (_, row), orig_d, swap_d in zip(
        swap_df.iterrows(), all_orig_data, all_swap_data
    ):
        if orig_d is None or swap_d is None:
            all_attachment_probs.append(None)
            head_probs.append(None)
            swapped_head_probs.append(None)
            max_swapped_head_probs.append(None)
            continue

        orig_scores, _ = _scores_from_batch(
            orig_d, proba_matrix, leaf_ids, offset, classes, is_dt
        )
        offset += len(orig_d.candidates)
        swap_scores, _ = _scores_from_batch(
            swap_d, proba_matrix, leaf_ids, offset, classes, is_dt
        )
        offset += len(swap_d.candidates)

        all_attachment_probs.append((orig_scores, swap_scores))
        head_probs.append(orig_scores.get(row.original_head_id, 0.0))
        swapped_head_probs.append(swap_scores.get(row.swapped_head_id, 0.0))
        max_swapped_head_probs.append(max(swap_scores.values()) if swap_scores else 0.0)

    swap_df["attachment_probs"] = all_attachment_probs
    swap_df["head_prob"] = head_probs
    swap_df["swapped_head_prob"] = swapped_head_probs
    swap_df["max_swapped_head_prob"] = max_swapped_head_probs

    return swap_df
