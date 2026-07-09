import os
from collections import Counter

import numpy as np
import pandas as pd

from .process_treebank import PredictionTarget
from .utils import capitalize_first, shorten_cls


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
    no_space_afters = [(tok["misc"] or {}).get("SpaceAfter") == "No" for tok in tree]
    sen_str = ""
    for tok, no_space_after in zip(sen, no_space_afters):
        sen_str += tok if no_space_after else f"{tok} "
    sen_str = sen_str.strip()

    return sen_str, no_space_afters


def get_swapped_sen_str(swapped_sen, no_space_afters_swapped):
    swapped_sen_str = ""
    for tok, no_space_after in zip(swapped_sen, no_space_afters_swapped):
        swapped_sen_str += tok if no_space_after else f"{tok} "
    return swapped_sen_str.strip()


def _build_swap_logic(target, dep_ids, h_ids):
    """
    Build the swap_logic dict for the current row.

    Each entry maps a target order string to (base_order, ids_to_move, pivot_idx).
    Within each source dict the insertion order matters: later entries may reference
    base orders produced by earlier entries in the same iteration.
    """
    h = shorten_cls("head", target)

    if len(target.child_deprels) == 1:
        dep1 = target.child_deprels[0]
        a = shorten_cls(dep1, target)
        a_ids = dep_ids[dep1]
        ah, ha = f"{a}{h}", f"{h}{a}"
        return {
            ah: {ha: (ah, a_ids, h_ids[-1])},
            ha: {ah: (ha, a_ids, h_ids[0])},
        }

    dep1, dep2 = target.child_deprels
    a = shorten_cls(dep1, target)
    b = shorten_cls(dep2, target)
    a_ids = dep_ids[dep1]
    b_ids = dep_ids[dep2]

    # All 6 permutations of (dep1, head, dep2), mirroring the SVO structure:
    # a ↔ subj, h ↔ verb/head, b ↔ obj
    ahb, abh = f"{a}{h}{b}", f"{a}{b}{h}"
    hab, hba = f"{h}{a}{b}", f"{h}{b}{a}"
    bha, bah = f"{b}{h}{a}", f"{b}{a}{h}"

    return {
        ahb: {
            abh: (ahb, b_ids, h_ids[0]),
            bah: (ahb, b_ids, a_ids[0]),
            hab: (ahb, a_ids, h_ids[-1]),
            hba: (ahb, a_ids, b_ids[-1]),
            bha: (hab, b_ids, h_ids[0] - len(a_ids)),
        },
        abh: {
            ahb: (abh, b_ids, h_ids[-1]),
            bah: (abh, b_ids, a_ids[0]),
            hab: (abh, h_ids, a_ids[0]),
            bha: (abh, a_ids, h_ids[-1]),
            hba: (bah, h_ids, b_ids[0] - len(a_ids)),
        },
        hab: {
            ahb: (hab, a_ids, h_ids[0]),
            hba: (hab, b_ids, a_ids[0]),
            bha: (hab, b_ids, h_ids[0]),
            abh: (hab, h_ids, b_ids[-1]),
            bah: (ahb, b_ids, a_ids[0] - len(h_ids)),
        },
        hba: {
            hab: (hba, a_ids, b_ids[-1]),
            bha: (hba, b_ids, h_ids[0]),
            ahb: (hba, a_ids, h_ids[0]),
            bah: (hba, h_ids, a_ids[-1]),
            abh: (bha, a_ids, b_ids[0] - len(h_ids)),
        },
        bha: {
            bah: (bha, a_ids, h_ids[-1]),
            hba: (bha, h_ids, b_ids[0]),
            hab: (bha, b_ids, a_ids[-1]),
            abh: (bha, a_ids, b_ids[0]),
            ahb: (hba, a_ids, h_ids[0] - len(b_ids)),
        },
        bah: {
            bha: (bah, a_ids, h_ids[-1]),
            abh: (bah, a_ids, b_ids[0]),
            ahb: (bah, b_ids, h_ids[-1]),
            hba: (bah, h_ids, b_ids[0]),
            hab: (abh, h_ids, a_ids[0] - len(b_ids)),
        },
    }


def create_all_constituent_swaps(
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

        sen_str, no_space_afters = get_sen_str(tree, row.sen)
        sen_orders = {row.deprel_order: row.sen}
        nsa_orders = {row.deprel_order: no_space_afters}
        swap_logic = _build_swap_logic(target, dep_ids, h_ids)
        swap_candidates = set(row.swap_order_candidates)

        for swap_order, (base_sen_order, ids, pivot_idx) in swap_logic[
            row.deprel_order
        ].items():
            pivot_idx = int(pivot_idx)
            base_sen = list(sen_orders[base_sen_order])
            swap_sen = move_indices_relative(base_sen, ids, pivot_idx)

            if swap_sen is None:
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

            base_nsa = nsa_orders[base_sen_order]
            swap_nsa = move_indices_relative(base_nsa, ids, pivot_idx)
            assert swap_nsa is not None  # same ids/pivot already passed for swap_sen

            # SpaceAfter=No is a right-boundary property of a constituent, not of an
            # individual token. Transfer it from the old rightmost constituent to the
            # new rightmost; reset the constituent that moved to the left to space.
            sorted_ids = sorted(ids)
            len_ids = len(sorted_ids)
            if all(i > pivot_idx for i in sorted_ids):
                # Constituent was right of pivot; now moved left. Pivot becomes rightmost.
                swap_nsa[pivot_idx + len_ids] = base_nsa[sorted_ids[-1]]
                swap_nsa[pivot_idx + len_ids - 1] = False
            else:
                # Constituent was left of pivot; now moved right. Constituent becomes rightmost.
                swap_nsa[pivot_idx] = base_nsa[pivot_idx]
                swap_nsa[pivot_idx - len_ids] = False

            sen_orders[swap_order] = swap_sen
            nsa_orders[swap_order] = swap_nsa

            if swap_order in swap_candidates:
                result_row = row.to_dict()
                result_row["sen_str"] = sen_str
                result_row["original_order"] = row.deprel_order
                result_row["swap_order"] = swap_order
                result_row["swapped_sen_str"] = get_swapped_sen_str(swap_sen, swap_nsa)
                result_rows.append(result_row)

    return pd.DataFrame(result_rows)


def balance_df(df, n_items, balance_features):
    feature_distributions = {feat: Counter(df[feat]) for feat in balance_features}
    feature_totals = {
        feat: sum(feature_distributions[feat].values()) for feat in balance_features
    }
    df["sample_weight"] = [
        1
        / np.prod(
            [
                feature_distributions[feat][row[feat]] / feature_totals[feat]
                for feat in balance_features
            ]
        )
        for _, row in df.iterrows()
    ]
    df.sample_weight /= sum(df.sample_weight)

    ids = np.random.choice(
        range(len(df)),
        size=min(n_items, len(df)),
        p=df.sample_weight,
        replace=False,
    )

    return df.iloc[ids]


def create_pairs(
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

    # Sample `max_per_leaf` items for each leaf_id that has entropy below threshold
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

    swap_df = create_all_constituent_swaps(
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
