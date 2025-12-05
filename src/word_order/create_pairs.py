from .decision_tree import set_dt_meta
import warnings

from .process_treebank import tree2sen


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


def create_swaps(df, treebank, leaf2dir, threshold=0.1):
    leaf_decision = [row.dir == leaf2dir[row.leaf_id] for _, row in df.iterrows()]
    df["leaf_decision"] = leaf_decision
    low_entropy_df = df[(df.leaf_entropy < threshold) & df.leaf_decision].copy()

    swapped_sens = []
    str_sens = []
    str_swapped_sens = []

    for df_idx, row in low_entropy_df.iterrows():
        tree = treebank[row.tree_idx]
        ids = get_subtree_indices(tree, row.child_idx)
        swapped_sen = move_indices_relative(row.sen, ids, row.head_idx)

        if 0 in ids:
            swapped_sen = None

        swapped_sens.append(swapped_sen)

        no_space_afters = [
            False
            # (tok['misc'] or {}).get('SpaceAfter') == "No"
            for tok in tree
        ]
        sen_str = ""
        for tok, no_space_after in zip(row.sen, no_space_afters):
            sen_str += tok if no_space_after else f"{tok} "
        str_sens.append(sen_str.strip())

        if swapped_sen is not None:
            # TODO: fix space after logic!
            # no_space_afters_swapped = move_indices_relative(
            #     no_space_afters, ids, row.head_idx
            # )

            swapped_sen_str = ""
            for tok, no_space_after in zip(swapped_sen, no_space_afters):
                swapped_sen_str += tok if no_space_after else f"{tok} "
            str_swapped_sens.append(swapped_sen_str.strip())
        else:
            str_swapped_sens.append(None)

    low_entropy_df["sen_str"] = str_sens
    low_entropy_df["swapped_sen_str"] = str_swapped_sens
    low_entropy_df["swapped_sen"] = swapped_sens

    low_entropy_df = low_entropy_df[
        ~low_entropy_df["swapped_sen_str"].apply(lambda x: x is None)
    ]

    return low_entropy_df


def create_pairs(model, dt_df, full_df, treebank, max_per_leaf=100, threshold=0.1, save_to: str | None = None):
    set_dt_meta(model, dt_df, full_df)

    tree = model.named_steps["clf"].tree_
    leaf2dir = {idx: direction.item() for idx, direction in enumerate(tree.value.argmax(-1))}

    full_swap_df = create_swaps(dt_df, treebank, leaf2dir, threshold=threshold)

    if len(full_swap_df) > 0:
        full_swap_df = (
            full_swap_df
            .groupby("leaf_id", group_keys=False)
            .apply(lambda g: g.sample(n=min(max_per_leaf, len(g)), replace=False, random_state=42))
        )

        if save_to is not None:
            full_swap_df.to_csv(save_to, index=False)

    return full_swap_df
