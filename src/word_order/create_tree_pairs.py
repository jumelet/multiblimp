import copy
import os
from collections import Counter

import numpy as np
import pandas as pd
from conllu import TokenList

from .create_pairs import (
    get_subtree_indices,
    _build_swap_logic,
    balance_df,
)
from .process_treebank import PredictionTarget
from .utils import capitalize_first


def _get_space_after(token):
    return (token["misc"] or {}).get("SpaceAfter") == "No"


def _set_space_after(token, no_space):
    if no_space:
        if token["misc"] is None:
            token["misc"] = {}
        token["misc"]["SpaceAfter"] = "No"
    else:
        if token["misc"] is not None:
            token["misc"].pop("SpaceAfter", None)
            if not token["misc"]:
                token["misc"] = None


def tokens_to_sen_str(tokens):
    parts = []
    for tok in tokens:
        no_space = _get_space_after(tok)
        parts.append(tok["form"] if no_space else f"{tok['form']} ")
    return "".join(parts).strip()


def swap_tree_tokens(base_tokens, ids, pivot_idx):
    """
    Reorder `base_tokens` by moving the subtree at 0-based `ids` relative to
    `pivot_idx` (0-based).  Returns a new list of deep-copied token dicts with
    updated `id`, `head`, `misc` (SpaceAfter), and `form` (capitalisation),
    or None if the subtree is non-projective w.r.t. the pivot.
    """
    sorted_ids = sorted(ids)
    ids_set = set(sorted_ids)
    len_ids = len(sorted_ids)

    if all(i > pivot_idx for i in sorted_ids):
        insert_pos = pivot_idx  # subtree moves left of pivot
        right_to_left = True
    elif all(i < pivot_idx for i in sorted_ids):
        insert_pos = pivot_idx - len_ids + 1  # subtree moves right of pivot
        right_to_left = False
    else:
        return None

    # Capture SpaceAfter from relevant boundary tokens before reordering.
    orig_nsa_subtree_last = _get_space_after(base_tokens[sorted_ids[-1]])
    orig_nsa_pivot = _get_space_after(base_tokens[pivot_idx])

    # Reorder (still pointing at the original dicts — copy happens next).
    items = [base_tokens[i] for i in sorted_ids]
    remainder = [base_tokens[i] for i in range(len(base_tokens)) if i not in ids_set]
    new_tokens = [
        copy.deepcopy(t)
        for t in remainder[:insert_pos] + items + remainder[insert_pos:]
    ]

    # Build old 1-based id → new 1-based id mapping before we overwrite ids.
    old_id_to_new_id = {
        tok["id"]: new_pos + 1 for new_pos, tok in enumerate(new_tokens)
    }

    # Update id and head on every token.
    for new_pos, tok in enumerate(new_tokens):
        tok["id"] = new_pos + 1
        if tok["head"] != 0:
            tok["head"] = old_id_to_new_id[tok["head"]]

    # SpaceAfter: transfer the right-edge boundary marker to the new rightmost token.
    # After a right-to-left move: subtree at [pivot_idx .. pivot_idx+len_ids-1],
    #                              pivot at pivot_idx+len_ids.
    # After a left-to-right move: pivot at pivot_idx-len_ids,
    #                              subtree at [pivot_idx-len_ids+1 .. pivot_idx].
    if right_to_left:
        _set_space_after(new_tokens[pivot_idx + len_ids], orig_nsa_subtree_last)
        _set_space_after(new_tokens[pivot_idx + len_ids - 1], False)
    else:
        _set_space_after(new_tokens[pivot_idx], orig_nsa_pivot)
        _set_space_after(new_tokens[pivot_idx - len_ids], False)

    # Capitalisation: whichever token is now first should be capitalised;
    # whichever token is no longer first should be lower-cased.
    if pivot_idx == 0:
        # Pivot was first; subtree is now at the front.
        new_tokens[0]["form"] = capitalize_first(new_tokens[0]["form"])
        new_tokens[len_ids]["form"] = new_tokens[len_ids]["form"].lower()
    if 0 in ids_set:
        # Subtree started at first position; it moved right, pivot is now first.
        new_tokens[0]["form"] = capitalize_first(new_tokens[0]["form"])
        new_tokens[pivot_idx - len_ids + 1]["form"] = new_tokens[
            pivot_idx - len_ids + 1
        ]["form"].lower()

    return new_tokens


def _moved_root_original_id(base_tokens, ids) -> int:
    """Return the 1-based id of the subtree root in the original tree."""
    sorted_ids = sorted(ids)
    subtree_1based = {base_tokens[i]["id"] for i in sorted_ids}
    root_pos = next(
        (i for i in sorted_ids if base_tokens[i]["head"] not in subtree_1based),
        sorted_ids[0],
    )
    return base_tokens[root_pos]["id"]


def _moved_root_new_id(base_tokens, ids, pivot_idx) -> int:
    """
    Return the new 1-based id (in the swapped tree) of the root of the moved
    subtree — the token whose head is outside the subtree (i.e. the topmost node).
    """
    sorted_ids = sorted(ids)
    subtree_1based = {base_tokens[i]["id"] for i in sorted_ids}
    root_pos = next(
        (i for i in sorted_ids if base_tokens[i]["head"] not in subtree_1based),
        sorted_ids[0],
    )
    rank = sorted_ids.index(root_pos)
    if all(i > pivot_idx for i in sorted_ids):
        insert_pos = pivot_idx
    else:
        insert_pos = pivot_idx - len(sorted_ids) + 1
    return insert_pos + rank + 1  # 1-based


def create_all_constituent_swaps_tree(
    df, treebank, target: PredictionTarget, max_sen_len=100
):
    assert len(target.child_deprels) in (1, 2)

    result_rows = []

    for _, row in df.iterrows():
        if row.sen_len > max_sen_len:
            continue

        tree = treebank[row.tree_idx]

        dep_ids = {
            dep: get_subtree_indices(tree, int(row[f"{dep}_idx"]) - 1)
            for dep in target.child_deprels
        }
        h_ids = [int(row.head_idx) - 1]

        sen_str = tokens_to_sen_str(list(tree))
        tree_orders = {row.deprel_order: list(tree)}
        swap_logic = _build_swap_logic(target, dep_ids, h_ids)
        swap_candidates = set(row.swap_order_candidates)

        for swap_order, (base_sen_order, ids, pivot_idx) in swap_logic[
            row.deprel_order
        ].items():
            pivot_idx = int(pivot_idx)
            base_tokens = tree_orders[base_sen_order]

            # The head/verb is the moved constituent (rather than a dependent) for
            # roughly half the 2-deprel swaps; flag these since the moved root is then
            # the head, not a child, which changes what attachment validation means.
            head_was_moved = ids == h_ids

            new_tokens = swap_tree_tokens(base_tokens, ids, pivot_idx)
            if new_tokens is None:
                break

            tree_orders[swap_order] = new_tokens

            if swap_order in swap_candidates:
                new_metadata = dict(tree.metadata)
                new_metadata["text"] = tokens_to_sen_str(new_tokens)
                swapped_tree = TokenList(new_tokens, metadata=new_metadata)

                original_child_id = _moved_root_original_id(base_tokens, ids)
                swapped_child_id = _moved_root_new_id(base_tokens, ids, pivot_idx)

                result_row = row.to_dict()
                result_row["sen_str"] = sen_str
                result_row["original_order"] = row.deprel_order
                result_row["swap_order"] = swap_order
                result_row["swapped_sen_str"] = new_metadata["text"]
                result_row["original_child_id"] = original_child_id
                result_row["swapped_child_id"] = swapped_child_id
                result_row["head_was_moved"] = head_was_moved
                result_row["original_head_id"] = tree[original_child_id - 1]["head"]
                result_row["swapped_head_id"] = swapped_tree[swapped_child_id - 1][
                    "head"
                ]
                result_row["original_tree"] = tree.serialize()
                result_row["swapped_tree"] = swapped_tree.serialize()
                result_rows.append(result_row)

    return pd.DataFrame(result_rows)


def create_pairs_tree(
    dt_df: pd.DataFrame,
    treebank,
    target: PredictionTarget,
    max_total_items: int = 1000,
    max_sen_len: int = 100,
    balance_features: list[str] | None = None,
    save_to_pairs_only: str | None = None,
    save_to: str | None = None,
):
    keep_df = dt_df[dt_df.num_swaps > 0].copy()

    if len(keep_df) == 0:
        return None

    num_leaves = len(keep_df.leaf_id.unique())
    items_per_leaf = max_total_items // num_leaves

    print(items_per_leaf)

    keep_df["sen_len"] = [len(sen) for sen in keep_df.sen]

    if balance_features is not None:
        selected_idx = (
            keep_df.groupby("leaf_id", group_keys=False)
            .apply(lambda g: balance_df(g, items_per_leaf, balance_features))
            .index
        )
    else:
        selected_idx = (
            keep_df.groupby("leaf_id", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=min(items_per_leaf, len(g)),
                    replace=False,
                    random_state=42,
                )
            )
            .index
        )

    keep_df = keep_df.loc[selected_idx]

    swap_df = create_all_constituent_swaps_tree(
        keep_df, treebank, target, max_sen_len=max_sen_len
    )

    if len(swap_df) > 0:
        if save_to_pairs_only is not None:
            pair_df = swap_df[["sen_str", "swapped_sen_str", "leaf_rule"]].copy()
            pair_df = pair_df.sort_values("leaf_rule")

            os.makedirs(os.path.dirname(save_to_pairs_only), exist_ok=True)
            pair_df.to_csv(save_to_pairs_only, index=False)

        if save_to is not None:
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            swap_df.to_csv(save_to, index=False)

    return swap_df
