import os
import random as _random
from copy import deepcopy
from dataclasses import dataclass, field
import conllu
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm.notebook import tqdm

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
    verbose: bool = True,
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

    with tqdm(
        total=n_samples, desc="Collecting training pairs", disable=not verbose
    ) as pbar:
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
    save_to: str | None = None,
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
            verbose=verbose,
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

    if save_to is not None:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        joblib.dump(classifier, save_to + ".joblib")

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


def _probs_to_records(probs: dict[int, float] | None) -> list[dict] | None:
    if probs is None:
        return None
    return [{"head_id": head_id, "prob": prob} for head_id, prob in probs.items()]


def probs_from_records(records) -> dict[int, float] | None:
    """Turn an *_attachment_probs cell back into a {head_id: prob} dict."""
    if records is None:
        return None
    return {int(r["head_id"]): float(r["prob"]) for r in records}


def score_swap_df(
    swap_df: pd.DataFrame,
    classifier: JointClassifier,
    omit_orig_head: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Score all rows of swap_df (as returned by create_pairs_tree) in two phases:
      1. Collect features for every (original, swapped) tree pair (no model calls).
      2. Score the entire dataset in a single batched forward pass.

    Rows where the head/verb itself was the moved constituent (head_was_moved=True)
    are skipped, since the moved root is then the head rather than a child and the
    attachment validation is not meaningful; their score columns are set to None.

    If omit_orig_head is True, the original head (swapped_head_id) is excluded when
    computing max_swapped_head_prob, so only novel rival heads are considered.

    Returns a copy of swap_df with five new columns:
      orig_attachment_probs, swapped_attachment_probs, head_prob,
      swapped_head_prob, max_swapped_head_prob.
    The two attachment_probs columns hold [{"head_id": ..., "prob": ...}, ...]
    records; use probs_from_records to turn a cell back into a dict.
    """
    # Phase 1: feature extraction (skip head-moving swaps)
    all_orig_data: list[HeadData | None] = []
    all_swap_data: list[HeadData | None] = []

    for _, row in tqdm(
        swap_df.iterrows(),
        total=len(swap_df),
        desc="Extracting features",
        disable=not show_progress,
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

    # Can be empty, e.g. when every row moves the head; nothing to score then.
    if all_feature_rows:
        batch_df = _scoring_df(all_feature_rows, classifier.model)
        proba_matrix = classifier.model.predict_proba(batch_df)
    else:
        batch_df, proba_matrix = None, None
    classes = list(classifier.model.named_steps["clf"].classes_)

    is_dt = classifier.model_type == "decision_tree" and batch_df is not None
    if is_dt:
        X_trans = classifier.model.named_steps["preprocessor"].transform(batch_df)
        leaf_ids = classifier.model.named_steps["clf"].apply(X_trans)
    else:
        leaf_ids = None

    # Phase 3: unpack results per row
    orig_attachment_probs = []
    swapped_attachment_probs = []
    head_probs = []
    swapped_head_probs = []
    max_swapped_head_probs = []

    offset = 0
    for (_, row), orig_d, swap_d in zip(
        swap_df.iterrows(), all_orig_data, all_swap_data
    ):
        if orig_d is None or swap_d is None:
            orig_attachment_probs.append(None)
            swapped_attachment_probs.append(None)
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

        orig_attachment_probs.append(orig_scores)
        swapped_attachment_probs.append(swap_scores)
        head_probs.append(orig_scores.get(row.original_head_id, 0.0))
        swapped_head_probs.append(swap_scores.get(row.swapped_head_id, 0.0))
        if omit_orig_head:
            rival_scores = {
                cand_id: prob
                for cand_id, prob in swap_scores.items()
                if cand_id != row.swapped_head_id
            }
        else:
            rival_scores = swap_scores
        max_swapped_head_probs.append(
            max(rival_scores.values()) if rival_scores else 0.0
        )

    # Stored as a list of {"head_id", "prob"} records rather than {head_id: prob}:
    # pyarrow can't write dicts with int keys to parquet, and [head_id, prob] pairs
    # would be coerced to floats.
    swap_df["orig_attachment_probs"] = [
        _probs_to_records(d) for d in orig_attachment_probs
    ]
    swap_df["swapped_attachment_probs"] = [
        _probs_to_records(d) for d in swapped_attachment_probs
    ]
    swap_df["head_prob"] = head_probs
    swap_df["swapped_head_prob"] = swapped_head_probs
    swap_df["max_swapped_head_prob"] = max_swapped_head_probs

    return swap_df


# ── SHAP explanation of a single swap_df row ──────────────────────────────────


@dataclass
class SwapExplanation:
    """
    SHAP attributions for every candidate head of one swap_df row, on both the
    original and the swapped tree.

    Each candidate is attributed on *its own* direction class ("L" if the child
    precedes the candidate head, "R" otherwise) — the same class whose
    probability score_swap_df records, so the numbers line up with head_prob /
    swapped_head_prob / max_swapped_head_prob.

    Attributes:
        shap_df:      tidy top-k table, one row per (side, head_id, feature).
        explanations: (side, head_id) -> shap.Explanation, full feature vector.
        probs:        (side, head_id) -> probability of that candidate's direction class.
        directions:   (side, head_id) -> "L" or "R".
        highlights:   the three candidates plotted by default, as
                      (side, head_id, label) triples.
    """

    shap_df: pd.DataFrame
    explanations: dict
    probs: dict
    directions: dict
    highlights: list
    sen_str: str = ""
    swapped_sen_str: str = ""
    child_forms: dict = field(default_factory=dict)

    def waterfall(self, side: str, head_id: int, max_display: int = 15, show=True):
        """Draw the SHAP waterfall for one candidate. side is 'original'/'swapped'."""
        import matplotlib.pyplot as plt
        import shap

        key = (side, int(head_id))
        if key not in self.explanations:
            available = sorted(h for s, h in self.explanations if s == side)
            raise KeyError(
                f"No candidate head {head_id} on the {side} tree. "
                f"Available: {available}"
            )

        # waterfall draws into the current figure; give each call its own.
        fig = plt.figure()
        shap.plots.waterfall(self.explanations[key], max_display=max_display, show=False)
        _swap_waterfall_labels(plt.gca())
        label = next(
            (lab for s, h, lab in self.highlights if (s, h) == key), "candidate"
        )
        plt.title(
            f"{side}: child {self.child_forms.get(side, '')} → head {head_id}"
            f"  [{label}]  p({self.directions[key]})={self.probs[key]:.3f}",
            fontsize=10,
        )
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot(self, max_display: int = 15, show=True):
        """Draw the default waterfalls: the true head, the swapped head, and the
        best-scoring candidate on the swapped tree."""
        return [
            self.waterfall(side, head_id, max_display=max_display, show=show)
            for side, head_id, _ in self.highlights
        ]


def _swap_waterfall_labels(ax):
    """
    Rewrite shap's y-tick labels from "value = feature_name" to "feature_name = value".

    shap.plots.waterfall draws the y-ticks twice — the full label in grey, then the
    part after "=" overlaid in black — so both sets have to be rewritten, and the
    black overlay has to stay the *suffix* of the grey label for it to line up.
    """
    labels = [t.get_text() for t in ax.get_yticklabels()]
    half = len(labels) // 2
    if half == 0:
        return

    for i in range(half):
        if " = " not in labels[i]:
            continue
        # value may itself contain " = "; the feature name never does.
        value, name = labels[i].rsplit(" = ", 1)
        labels[i] = f"{name} = {value}"
        labels[i + half] = value

    ax.set_yticklabels(labels)


def _shap_feature_space(classifier: JointClassifier):
    """
    Return (ordered original feature names, cleaned names) in the column order the
    LightGBM booster sees, i.e. categorical columns followed by numeric columns.
    """
    prep = classifier.model.named_steps["preprocessor"]
    rec = prep.named_steps["reconstruct"]
    original = list(rec.categorical_cols) + list(rec.numeric_cols)
    return original, list(rec.clean_cols_)


def _shap_for_side(
    data: HeadData,
    classifier: JointClassifier,
    explainer,
    original_names: list[str],
):
    """
    Compute per-candidate shap values, base values and probabilities for one HeadData.

    Returns (values, base, proba, display) with values shaped
    (n_candidates, n_features, n_classes) and base (n_candidates, n_classes).
    """
    batch_df = _scoring_df(data.feature_rows, classifier.model)
    prep = classifier.model.named_steps["preprocessor"]
    clf = classifier.model.named_steps["clf"]

    X_t = prep.transform(batch_df)
    proba = clf.predict_proba(X_t)

    n_classes = len(clf.classes_)
    explanation = explainer(X_t)

    values = np.asarray(explanation.values)
    if values.ndim == 2:  # older shap may return a list stacked per class
        values = values.reshape(values.shape[0], -1, n_classes)

    # base_values arrives as (n, n_classes), (n_classes,) or a scalar depending on
    # the shap version; broadcast whatever we get to (n_candidates, n_classes).
    base = np.asarray(explanation.base_values, dtype=float)
    base = np.broadcast_to(
        base.reshape(-1, n_classes) if base.size % n_classes == 0 else base.reshape(1, 1),
        (values.shape[0], n_classes),
    )

    # Pre-transform values, in booster column order, for readable plot labels.
    display = batch_df[original_names].to_numpy(dtype=object)
    return values, base, proba, display


def explain_swap_row(
    row,
    classifier: JointClassifier,
    top_k: int = 15,
    plot: bool = True,
    max_display: int = 15,
) -> SwapExplanation:
    """
    Explain the model's reattachment predictions for a single row of swap_df.

    Rebuilds exactly the features score_swap_df would build for this row — every
    projectivity-valid candidate head on the original tree (including the true
    head) and on the swapped tree — then computes exact TreeSHAP attributions for
    each candidate on its own direction class.

    By default draws three waterfalls: the true head on the original tree
    (behind head_prob), the swapped head (behind swapped_head_prob), and the
    best-scoring candidate on the swapped tree (behind max_swapped_head_prob).

    Args:
        row: one row of swap_df, e.g. swap_df.iloc[42].
        classifier: the fitted JointClassifier; must be gradient_boosting.
        top_k: features kept per candidate in shap_df, ranked by |shap value|.
        plot: draw the default waterfalls.
        max_display: features shown per waterfall.
    """
    import shap

    if classifier.model_type != "gradient_boosting":
        raise ValueError(
            "explain_swap_row needs a gradient_boosting (LightGBM) classifier; "
            f"got model_type={classifier.model_type!r}."
        )
    if row.get("head_was_moved", False):
        raise ValueError(
            "This row has head_was_moved=True: the moved constituent is the head "
            "rather than a child, so attachment validation is not meaningful and "
            "score_swap_df leaves its probabilities as None."
        )

    orig_tree = deserialize_tree(row.original_tree)
    swap_tree = deserialize_tree(row.swapped_tree)

    orig_data = _collect_head_features(
        orig_tree, classifier, row.original_child_id, include_true_head=True
    )
    swap_data = _collect_head_features(swap_tree, classifier, row.swapped_child_id)

    original_names, _ = _shap_feature_space(classifier)
    clf = classifier.model.named_steps["clf"]
    classes = list(clf.classes_)
    explainer = shap.TreeExplainer(clf)

    explanations, probs, directions = {}, {}, {}
    records = []

    for side, data in (("original", orig_data), ("swapped", swap_data)):
        if not data.candidates:
            continue
        sv, base, proba, display = _shap_for_side(
            data, classifier, explainer, original_names
        )

        for i, (h_id, direction) in enumerate(zip(data.candidates, data.directions)):
            c = classes.index(direction)
            key = (side, int(h_id))
            values = sv[i, :, c]

            explanations[key] = shap.Explanation(
                values=values,
                base_values=float(base[i, c]),
                data=display[i],
                feature_names=original_names,
            )
            probs[key] = float(proba[i, c])
            directions[key] = direction

            order = np.argsort(-np.abs(values))[:top_k]
            for rank, j in enumerate(order):
                records.append(
                    {
                        "side": side,
                        "head_id": int(h_id),
                        "direction": direction,
                        "prob": probs[key],
                        "rank": rank,
                        "feature": original_names[j],
                        "value": display[i][j],
                        "shap": float(values[j]),
                    }
                )

    shap_df = pd.DataFrame(records)

    # The three candidates worth looking at, mirroring score_swap_df's columns.
    highlights = []
    true_key = ("original", int(row.original_head_id))
    if true_key in explanations:
        highlights.append((*true_key, "true head → head_prob"))

    swapped_key = ("swapped", int(row.swapped_head_id))
    swap_probs = {h: p for (s, h), p in probs.items() if s == "swapped"}
    best_h = max(swap_probs, key=swap_probs.get) if swap_probs else None

    if swapped_key in explanations:
        label = "swapped head → swapped_head_prob"
        if best_h == swapped_key[1]:
            label += " (also max_swapped_head_prob)"
        highlights.append((*swapped_key, label))
    if best_h is not None and best_h != swapped_key[1]:
        highlights.append(("swapped", best_h, "best rival → max_swapped_head_prob"))

    child_forms = {
        "original": orig_tree[int(row.original_child_id) - 1]["form"],
        "swapped": swap_tree[int(row.swapped_child_id) - 1]["form"],
    }

    result = SwapExplanation(
        shap_df=shap_df,
        explanations=explanations,
        probs=probs,
        directions=directions,
        highlights=highlights,
        sen_str=row.get("sen_str", ""),
        swapped_sen_str=row.get("swapped_sen_str", ""),
        child_forms=child_forms,
    )

    if plot:
        result.plot(max_display=max_display)

    return result
