import os

import pandas as pd

import warnings

from .process_treebank import PredictionTarget
from .utils import capitalize_first


def get_subtree_indices(tokens, index):
    """
    Return all token indices for which `index` is an ancestor.
    `index` should be 1-based, like CoNLL-U IDs.
    """
    # Build child lists
    children = {}
    for tok in tokens:
        head = tok["head"] - 1
        tid = tok["id"] - 1
        if head is not None:
            children.setdefault(head, []).append(tid)

    # Collect descendants
    stack = children.get(index, [])[:]  # immediate children
    subtree = set(stack)

    while stack:
        node = stack.pop()
        for child in children.get(node, []):
            if child not in subtree:
                subtree.add(child)
                stack.append(child)

    subtree.add(int(index))

    return sorted(subtree)


def move_indices_relative(sen, indices, head_idx):
    """
    sen: list
    indices: iterable of integer indices (0-based)
    head_idx: integer (0-based)
    """
    indices = sorted(indices)

    # Extract the items to move
    items = [sen[i] for i in indices]

    # Remove them from the original list
    remainder = [sen[i] for i in range(len(sen)) if i not in indices]

    # Determine insertion point in the remainder
    # Case 1: all moved indices > head_idx  → insert *before* head
    # Case 2: all moved indices < head_idx  → insert *after* head
    if all(i > head_idx for i in indices):
        insert_pos = head_idx
    elif all(i < head_idx for i in indices):
        insert_pos = head_idx - len(indices) + 1
    else:
        # error = f"Non-projective subtree! head_idx: {head_idx}\nindices:{indices}"
        # warnings.warn(error)
        return None

    # Insert moved items
    return remainder[:insert_pos] + items + remainder[insert_pos:]


def get_sen_str(tree, sen):
    no_space_afters = [
        # (tok['misc'] or {}).get('SpaceAfter') == "No"
        False
        for tok in tree
    ]
    sen_str = ""
    for tok, no_space_after in zip(sen, no_space_afters):
        sen_str += tok if no_space_after else f"{tok} "
    sen_str = sen_str.strip()

    return sen_str, no_space_afters


def get_swapped_sen_str(swapped_sen, no_space_afters, ids, head_idx):
    no_space_afters_swapped = move_indices_relative(no_space_afters, ids, head_idx)
    swapped_sen_str = ""
    for tok, no_space_after in zip(swapped_sen, no_space_afters_swapped):
        swapped_sen_str += tok if no_space_after else f"{tok} "
    swapped_sen_str = swapped_sen_str.strip()

    return swapped_sen_str


def create_core_arg_swaps(df, treebank, max_sen_len=100):
    df["non_projective"] = False

    for df_idx, row in df.iterrows():
        if len(row.sen) > max_sen_len:
            continue

        tree = treebank[row.tree_idx]

        subj_ids = get_subtree_indices(tree, row.subject_idx - 1)
        obj_ids = get_subtree_indices(tree, row.object_idx - 1)
        verb_ids = [int(row.verb_idx - 1)]  # get_subtree_indices(tree, row.verb_idx)

        sen_str, no_space_afters = get_sen_str(tree, row.sen)

        df.at[df_idx, "sen_str"] = sen_str
        df.at[df_idx, f"{row.core_args}_sen_str"] = sen_str

        sen_orders = {row.core_args: row.sen}

        swap_logic = {
            "svo": {
                "sov": ("svo", obj_ids, verb_ids[0]),
                "osv": ("svo", obj_ids, subj_ids[0]),
                "vso": ("svo", subj_ids, verb_ids[-1]),
                "vos": ("svo", subj_ids, obj_ids[-1]),
                "ovs": ("vso", obj_ids, verb_ids[0] - len(subj_ids)),
            },
            "sov": {
                "svo": ("sov", obj_ids, verb_ids[-1]),
                "osv": ("sov", obj_ids, subj_ids[0]),
                "vso": ("sov", verb_ids, subj_ids[0]),
                "ovs": ("sov", subj_ids, verb_ids[-1]),
                "vos": ("osv", verb_ids, obj_ids[0] - len(subj_ids)),
            },
            "vso": {
                "svo": ("vso", subj_ids, verb_ids[0]),
                "vos": ("vso", obj_ids, subj_ids[0]),
                "ovs": ("vso", obj_ids, verb_ids[0]),
                "sov": ("vso", verb_ids, obj_ids[-1]),
                "osv": ("svo", obj_ids, subj_ids[0] - len(verb_ids)),
            },
            "vos": {
                "vso": ("vos", subj_ids, obj_ids[-1]),
                "ovs": ("vos", obj_ids, verb_ids[0]),
                "svo": ("vos", subj_ids, verb_ids[0]),
                "osv": ("vos", verb_ids, subj_ids[-1]),
                "sov": ("ovs", subj_ids, obj_ids[0] - len(verb_ids)),
            },
            "ovs": {
                "osv": ("ovs", subj_ids, verb_ids[-1]),
                "vos": ("ovs", verb_ids, obj_ids[0]),
                "vso": ("ovs", obj_ids, subj_ids[-1]),
                "sov": ("ovs", subj_ids, obj_ids[0]),
                "svo": ("vos", subj_ids, verb_ids[0] - len(obj_ids)),
            },
            "osv": {
                "ovs": ("osv", subj_ids, verb_ids[-1]),
                "sov": ("osv", subj_ids, obj_ids[0]),
                "svo": ("osv", obj_ids, verb_ids[-1]),
                "vos": ("osv", verb_ids, obj_ids[0]),
                "vso": ("sov", verb_ids, subj_ids[0] - len(obj_ids)),
            },
        }

        for swap_core_arg, (base_sen_order, ids, pivot_idx) in swap_logic[
            row.core_args
        ].items():
            pivot_idx = int(pivot_idx)
            base_sen = list(sen_orders[base_sen_order])
            swap_sen = move_indices_relative(base_sen, ids, pivot_idx)

            if swap_sen is None:
                df.at[df_idx, "non_projective"] = True
                break

            # reset capitalization
            if pivot_idx == 0:
                swap_sen[0] = capitalize_first(swap_sen[0])
                swap_sen[len(ids)] = swap_sen[len(ids)].lower()
            if 0 in ids:
                swap_sen[0] = swap_sen[0][0].upper() + swap_sen[0][1:]
                swap_sen[pivot_idx - len(ids) + 1] = swap_sen[
                    pivot_idx - len(ids) + 1
                ].lower()

            swap_sen_str = get_swapped_sen_str(
                swap_sen, no_space_afters, ids, pivot_idx
            )

            sen_orders[swap_core_arg] = swap_sen

            df.at[df_idx, f"{swap_core_arg}_sen_str"] = swap_sen_str

    # only keep projective tree swaps
    df = df[~df["non_projective"]]

    return df


def create_pairs(
    dt_df: pd.DataFrame,
    treebank,
    max_per_leaf: int = 100,
    save_to_pairs_only: str | None = None,
    save_to: str | None = None,
):
    full_swap_df = create_core_arg_swaps(dt_df, treebank)
    
    if len(full_swap_df) > 0:
        # Sample `max_per_leaf` items for each leaf_id that has entropy below threshold
        selected_idx = (
            full_swap_df[full_swap_df["keep"]]
            .groupby("leaf_id", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=min(max_per_leaf, len(g)),
                    replace=False,
                    random_state=42,
                )
            )
            .index
        )
        full_swap_df["keep"] = False
        full_swap_df.loc[selected_idx, "keep"] = True

        if save_to_pairs_only is not None:
            sub_df = full_swap_df[full_swap_df.keep]
            tight_swap_df = sub_df[["sen_str", "swapped_sen_str", "leaf_rule"]].copy()
            tight_swap_df = tight_swap_df.sort_values("leaf_rule")

            os.makedirs(os.path.dirname(save_to_pairs_only), exist_ok=True)
            tight_swap_df.to_csv(save_to_pairs_only, index=False)

        if save_to is not None:
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            full_swap_df.to_csv(save_to, index=False)

    return full_swap_df
