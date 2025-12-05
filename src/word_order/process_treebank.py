import sys
import ast
import os
import random

from collections import defaultdict, Counter

sys.path.append("src")

from tqdm import tqdm
import pandas as pd
import numpy as np

from multiblimp.treebank import Treebank
from multiblimp.languages import remove_diacritics_langs, gblang2udlang


META_FEATURES = ["sen", "treebank", "sent_id", "tree_idx", "child_idx", "head_idx"]


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
        treebank = random.sample(treebank, max_treebank_len)

    return treebank


def read_df(lang, word_order_dir = None) -> pd.DataFrame:
    lang = gblang2udlang.get(lang, lang).replace(" ", "_")

    df = pd.read_csv(
        os.path.join(word_order_dir or "", f"{lang}.csv"),
        low_memory=False,
        converters={"sen": ast.literal_eval},
    )
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.replace([np.inf, -np.inf], "inf")

    return df


def tree2sen(tree):
    return [x["form"] for x in tree]


def remove_first_occurrence(lst, x):
    result = []
    removed = False
    for item in lst:
        if not removed and item == x:
            removed = True
            continue
        result.append(item)
    return result


def get_all_feats(treebank, min_freq=0.005):
    all_feats = Counter()
    all_deprel = set()
    all_lemma_freqs = Counter()
    all_pos = set()
    num_tokens = 0

    for tree in treebank:
        for token in tree:
            num_tokens += 1
            all_lemma_freqs[token["lemma"]] += 1
            for feat in token["feats"] or {}:
                all_feats[feat] += 1
            all_deprel.add(token["deprel"])
            all_pos.add(token["upos"])

    all_feats = Counter(
        {x: c / num_tokens for x, c in all_feats.items() if (c / num_tokens) > min_freq}
    )

    return set(all_feats), all_lemma_freqs, all_deprel, all_pos


def get_sub_condition_value(
    sub_condition,
    child,
    head,
    num_children,
    tree,
    child2root,
    head2child_deprels,
    head2child_lemmas,
    head2left_deprels,
    head2right_deprels,
    head2child_pos,
):
    main_features = {
        "deprel": child["deprel"],
        "child_pos": child["upos"],
        "head_pos": head["upos"],
        "head_deprel": head["deprel"],
        "grandhead_deprel": tree[head["head"] - 1]["deprel"],
        "child_form": child["form"],
        "head_form": head["form"],
        "child_lemma": child["lemma"],
        "head_lemma": head["lemma"],
        # "is_neighbor": abs(head["id"] - child["id"]) == 1,
        "#_siblings": len(num_children[child["head"]]) - 1,
        "has_siblings": len(num_children[child["head"]]) > 1,
        "#_children": len(num_children[child["id"]]),
        "has_children": len(num_children[child["id"]]) > 0,
        "in_question": tree[-1]["form"] == "?",
    }

    if sub_condition in main_features:
        return main_features[sub_condition]

    if "under" in sub_condition:
        deprel = sub_condition.split("_")[1]

        return deprel in child2root[child["id"]]

    sub_condition_splits = sub_condition.split("_")

    if len(sub_condition_splits) == 2:
        child_head, sub_condition = sub_condition_splits
    elif len(sub_condition_splits) == 3:
        child_head, sub_condition, feature = sub_condition_splits
    elif len(sub_condition_splits) >= 4:
        child_head = sub_condition_splits[0]
        lemma = "_".join(sub_condition_splits[3:])

        item = head if child_head == "head" else child
        sibling_lemmas = head2child_lemmas[item["head"]]

        return lemma in sibling_lemmas
    else:
        raise ValueError("subcondition not recognized")

    item = head if child_head == "head" else child

    if sub_condition == "sibling-deprel":
        sibling_deprels = remove_first_occurrence(
            head2child_deprels[item["head"]], item["deprel"]
        )

        return feature in sibling_deprels

    if sub_condition == "sibling-deprel-R":
        sibling_deprels = head2right_deprels[item["head"]]
        if item["id"] > item["head"]:
            sibling_deprels = remove_first_occurrence(sibling_deprels, item["deprel"])

        return feature in sibling_deprels

    if sub_condition == "sibling-deprel-L":
        sibling_deprels = head2left_deprels[item["head"]]
        if item["id"] < item["head"]:
            sibling_deprels = remove_first_occurrence(sibling_deprels, item["deprel"])

        return feature in sibling_deprels

    if sub_condition == "sibling-lemma":
        sibling_deprels = head2child_deprels[item["head"]]
        sibling_lemmas = head2child_lemmas[item["head"]]

        for deprel, lemma in zip(sibling_deprels, sibling_lemmas):
            if deprel == feature:
                return lemma

        return "None"

    if sub_condition == "sibling-pos":
        sibling_pos = head2child_pos[item["head"]]

        return feature in sibling_pos

    if sub_condition == "dir":
        return head["id"] < head["head"]

    if sub_condition == "position":
        rel_position = item["id"] / len(tree)
        if rel_position < (1 / 3):
            return "start"
        elif rel_position < (2 / 3):
            return "mid"
        else:
            return "end"

    return (item["feats"] or {}).get(sub_condition)


def extract_features(treebank, condition_on=None):
    condition_on = condition_on or []
    deprel_directions = []

    for tree_idx, tree in tqdm(enumerate(treebank), total=len(treebank)):
        # Initialize tree information
        num_children = defaultdict(list)
        child2head = {}
        child2root = defaultdict(set)
        head2child_deprels = defaultdict(list)
        head2child_pos = defaultdict(set)
        head2child_lemmas = defaultdict(list)
        head2left_deprels = defaultdict(list)
        head2right_deprels = defaultdict(list)

        for token in tree:
            num_children[token["head"]].append(token["id"])
            child2head[token["id"]] = token["head"]
            head2child_deprels[token["head"]].append(token["deprel"])
            head2child_pos[token["head"]].add(token["upos"])
            head2child_lemmas[token["head"]].append(token["lemma"])
            if token["id"] < token["head"]:
                head2left_deprels[token["head"]].append(token["deprel"])
            else:
                head2right_deprels[token["head"]].append(token["deprel"])

        for child, head in child2head.items():
            child2root[child].add(tree[child - 1]["deprel"])

            while head != 0:
                child2root[child].add(tree[head - 1]["deprel"])
                head = child2head[head]

        # Collect sub_condition values
        for child_idx, token in enumerate(tree):
            head_idx = token["head"] - 1
            deprel = token["deprel"]
            head = tree[head_idx]

            if deprel == "root":
                continue

            sen = tree2sen(tree)
            treebank = tree.metadata["treebank"].split("/")[0]
            sent_id = tree.metadata["sent_id"]

            # "Meta-features", make sure this order aligns with the META_FEATURES var on top of this file!
            condition = [sen, treebank, sent_id, tree_idx, child_idx, head_idx]

            for sub_condition in condition_on:
                sub_condition_value = get_sub_condition_value(
                    sub_condition,
                    token,
                    head,
                    num_children,
                    tree,
                    child2root,
                    head2child_deprels,
                    head2child_lemmas,
                    head2left_deprels,
                    head2right_deprels,
                    head2child_pos,
                )
                condition.append(sub_condition_value)

            if token["id"] < token["head"]:
                deprel_directions.append(condition + [0])
            elif token["id"] > token["head"]:
                deprel_directions.append(condition + [1])

    return deprel_directions


def create_word_order_df(
    lang: str,
    resource_dir: str | None = None,
    save_to: str | None = None,
    max_treebank_len: int | None = None,
) -> pd.DataFrame:
    lang = gblang2udlang.get(lang, lang).replace(" ", "_")
    treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len)

    condition_on = [
        "deprel",
        "child_pos",
        "head_pos",
        "head_deprel",
        # "grandhead_deprel",
        "child_form",
        "head_form",
        "child_lemma",
        "head_lemma",
        # "child_freq",
        # "head_freq",
        # "is_neighbor",
        # "#_siblings",
        "has_siblings",
        # "#_children",
        "has_children",
        "head_dir",
        "in_question",
        # "head_position",
        # "child_position",
    ]
    all_feats, all_lemma_freqs, all_deprel, all_pos = get_all_feats(treebank)
    condition_on.extend([f"child_{feat}" for feat in all_feats])
    condition_on.extend([f"head_{feat}" for feat in all_feats])
    condition_on.extend({f"under_{deprel}" for deprel in all_deprel})

    for sibling_dir in ["L", "R"]:
        condition_on.extend(
            {f"child_sibling-deprel-{sibling_dir}_{deprel}" for deprel in all_deprel}
        )
        condition_on.extend(
            {f"head_sibling-deprel-{sibling_dir}_{deprel}" for deprel in all_deprel}
        )

    condition_on.extend({f"child_sibling-pos_{pos}" for pos in all_pos})
    condition_on.extend({f"head_sibling-pos_{pos}" for pos in all_pos})

    condition_on.extend({f"child_sibling-deprel_{deprel}" for deprel in all_deprel})
    condition_on.extend({f"head_sibling-deprel_{deprel}" for deprel in all_deprel})

    condition_on.extend({f"child_sibling-lemma_{deprel}" for deprel in all_deprel})
    condition_on.extend({f"head_sibling-lemma_{deprel}" for deprel in all_deprel})

    deprel_dirs = extract_features(treebank, condition_on=condition_on)

    df_columns = META_FEATURES + condition_on + ["dir"]
    df = pd.DataFrame(deprel_dirs, columns=df_columns)

    if save_to is not None:
        df.to_csv(os.path.join(save_to or "", f"{lang}.csv"), index=False)

    return df
