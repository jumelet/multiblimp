"""Check that the DT features of an instance do not encode its child-head order.

For every instance, the target child (with its subtree) is moved to the other side
of its head, and the instances of the swapped tree are extracted again. A feature
that the DT can see must have the same value in both trees, and the same children
must be selected as instances; otherwise the feature (or the selection) carries
information about the order that is being predicted.
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

from conllu import TokenList

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from word_order.create_tree_pairs import Orthography, get_subtree_indices, swap_tree_tokens
from word_order.decision_tree import OMIT_FEATURES
from word_order.prediction_target import get_all_deprel_targets
from word_order.process_treebank import META_FEATURES, extract_instances, get_all_feats, load_treebank
from word_order.utils import SAMPLED_PICKLE_DIR

# The swap may change capitalisation, which is irrelevant for this check.
NO_CASE_CHANGES = Orthography(sentence_case=False)


def dt_visible(feature: str) -> bool:
    """Whether fit_dt can use the feature (see the omitted columns in fit_dt)."""
    return not (
        feature in META_FEATURES
        or feature in OMIT_FEATURES
        or feature == "deprel_order"
        or "idx" in feature
        or "_dir" in feature
        or "agreement" in feature
    )


def same(a, b) -> bool:
    return a == b or (a != a and b != b)  # NaN == NaN


def swap_child(tree, child_idx: int):
    """The tree with the subtree of the 0-based child moved across its head, or None."""
    tokens = list(tree)
    for tok in tokens:
        tok["orig_id"] = tok["id"]
    ids = get_subtree_indices(tokens, child_idx)
    new_tokens = swap_tree_tokens(tokens, ids, tokens[child_idx]["head"] - 1, NO_CASE_CHANGES)
    for tok in tokens:
        del tok["orig_id"]
    if new_tokens is None:
        return None
    return TokenList(new_tokens, metadata=dict(tree.metadata))


def check(treebank, target, tree_metadata, lexicalize, max_examples=3):
    deprel = target.child_deprels[0]
    n_instances = n_swapped = n_selection_changed = 0
    leaks = Counter()
    examples = defaultdict(list)

    for tree_idx, tree in enumerate(treebank):
        for inst in extract_instances(tree, tree_idx, target, tree_metadata, lexicalize=lexicalize):
            n_instances += 1
            child_id = inst[f"{deprel}_idx"]
            swapped = swap_child(tree, child_id - 1)
            if swapped is None:
                continue
            n_swapped += 1
            new_id = next(tok["id"] for tok in swapped if tok["orig_id"] == child_id)
            for tok in swapped:
                del tok["orig_id"]

            swapped_insts = extract_instances(swapped, tree_idx, target, tree_metadata, lexicalize=lexicalize)
            match = next((s for s in swapped_insts if s[f"{deprel}_idx"] == new_id), None)
            if match is None:
                n_selection_changed += 1
                continue

            for feature in set(inst) | set(match):
                if not dt_visible(feature):
                    continue
                a, b = inst.get(feature), match.get(feature)
                if not same(a, b):
                    leaks[feature] += 1
                    if len(examples[feature]) < max_examples:
                        examples[feature].append((" ".join(inst["sen"])[:80], a, b))

    return n_instances, n_swapped, n_selection_changed, leaks, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Directory with the outputs of the earlier pipeline steps")
    parser.add_argument("--langs", nargs="+", required=True)
    parser.add_argument("--deprels", nargs="+", default=None, help="Defaults to all targets")
    parser.add_argument("--max_trees", type=int, default=500)
    parser.add_argument("--lexicalize", action="store_true")
    args = parser.parse_args()

    targets = [
        t for t in get_all_deprel_targets() if args.deprels is None or t.child_deprels[0] in args.deprels
    ]
    for lang in args.langs:
        treebank = load_treebank(lang, os.path.join(args.data_dir, SAMPLED_PICKLE_DIR))
        tree_metadata = get_all_feats(treebank)
        treebank = treebank[: args.max_trees]
        for target in targets:
            n, n_swapped, n_sel, leaks, examples = check(treebank, target, tree_metadata, args.lexicalize)
            status = "OK" if not n_sel and not leaks else "LEAK"
            print(f"[{lang} {target.child_deprels[0]}] {status}: {n} instances, {n_swapped} swapped, "
                  f"{n_sel} with a different selection, {len(leaks)} non-invariant features", flush=True)
            for feature, count in leaks.most_common(10):
                print(f"    {feature}: {count}x, e.g. {examples[feature][0]}")


if __name__ == "__main__":
    main()
