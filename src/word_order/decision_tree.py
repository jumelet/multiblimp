import joblib
import os
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
from sklearn import set_config
from sklearn.tree import _tree

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .process_treebank import META_FEATURES
from .entropy import order_entropy
from .utils import ALL_CORE_ARGS

RND = 42
OMIT_FEATURES = ["subject_idx", "object_idx", "verb_idx", "obj_dir", "nsubj_dir"]


def fit_dt(
    df,
    predictor_var="dir",
    deprel_kwargs=None,
    omit_feats=None,
    save_to=None,
    max_depth=12,
    min_samples_leaf=25,
    min_impurity_decrease=0.005,
    verbose=1,
    min_df_len=10,
):
    sub_df = df.copy()
    if deprel_kwargs is not None:
        for k, v in deprel_kwargs.items():
            sub_df = sub_df[sub_df[k] == v]

    if len(sub_df) < min_df_len:
        return None, None, None

    omit_feats = (
        (omit_feats or set()).union(set(META_FEATURES)).union(set(OMIT_FEATURES))
    )
    omit_feats.union({col for col in df.columns if "idx" in col})

    sub_condition_on = list(set(df.columns) - omit_feats - {predictor_var})

    X = sub_df[sub_condition_on].copy()

    # drop columns with only a single value
    X = X.loc[:, X.nunique() > 1].copy()
    sub_condition_on = [col for col in sub_condition_on if col in X.columns]

    y = sub_df[predictor_var].copy()

    y[pd.isna(y)] = "None"

    numeric_cols = [
        feat
        for feat in sub_condition_on
        if (
            "freq" in feat
            or "#" in feat
            or "under" in feat
            or "question" in feat
            or ("sibling" in feat and not ("lemma" in feat or "feat" in feat))
            or "has_" in feat
        )
    ]
    categorical_cols = list(set(sub_condition_on) - set(numeric_cols))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RND
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                DecisionTreeClassifier(
                    criterion="entropy",
                    random_state=RND,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_impurity_decrease=min_impurity_decrease,
                    # ccp_alpha=0.01,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    if save_to is not None:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        joblib.dump(model, save_to + ".joblib")
        X_train.to_csv(save_to + ".csv")

    if verbose > 0:
        print("Train acc", model.score(X_train, y_train))
        print("Test acc ", model.score(X_test, y_test))

    return model, X_train, y_train


def pprint_node(txt):
    new_txt = (
        txt.replace(" <= 0.5", "")
        .replace("000", "")
        .replace(".0,", ",")
        .replace(".0]", "]")
        .strip()
    )

    if len(new_txt.split("_")) == 3:
        splits = new_txt.split("_")
        new_txt = f"{splits[0]}_{splits[1]} = {splits[2]}"
    elif len(new_txt.split("_")) == 4:
        splits = new_txt.split("_")
        new_txt = f"{splits[0]}_{splits[1]}_{splits[2]} = {splits[3]}"
    elif "True" in new_txt and not "\n" in new_txt:
        new_txt = "     False     "  # labels must be flipped, trust me it is right
    elif "False" in new_txt and not "\n" in new_txt:
        new_txt = "     True     "

    if " = nan" in new_txt:
        new_txt = new_txt.replace(" = nan", " is not set")

    return new_txt


def plot_dt(model, save_to=None, show_plot=True, class_names=None):
    set_config(transform_output="default")

    clf = model.named_steps["clf"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = [x.split("__")[1] for x in preprocessor.get_feature_names_out()]
    class_names = class_names or model.classes_

    fig = plt.figure(figsize=(25, 15))
    artists = plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        node_ids=True,
    )

    for node_id, artist in enumerate(artists):
        txt = artist.get_text()
        m = re.match(r"node #(\d+)\n(.*)", txt, re.S)
        if m is not None:
            node_id = int(m.group(1))
            rest = m.group(2)
            new_txt = pprint_node(rest)

            artist.set_text(f"[{node_id}] {new_txt}")
        else:
            new_txt = pprint_node(txt)
            artist.set_text(new_txt)

    if save_to is not None:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def set_dt_meta(model, df, full_df, predictor_var="dir", additional_vars=None, threshold=0.1):
    X_trans = model.named_steps["preprocessor"].transform(df)
    leaf_ids = model.named_steps["clf"].apply(X_trans)

    tree = model.named_steps["clf"].tree_
    impurity = tree.impurity
    children_left = tree.children_left
    children_right = tree.children_right

    # Map leaf to entropy
    is_leaf = (children_left == -1) & (children_right == -1)
    leaf_node_ids = np.where(is_leaf)[0]
    leaf_entropy_map = dict(zip(leaf_node_ids, impurity[leaf_node_ids]))

    # Vectorized lookup
    leaf_full_entropy = np.array([leaf_entropy_map[node] for node in leaf_ids])

    # Add columns from full_df that were omitted when fitting the DT
    additional_vars = additional_vars or []
    additional_vars.extend(META_FEATURES + OMIT_FEATURES + [predictor_var])
    for col in additional_vars:
        df[col] = full_df[col]

    df["leaf_id"] = leaf_ids
    df["leaf_full_entropy"] = leaf_full_entropy

    unseen_classes = list(set(ALL_CORE_ARGS) - set(model.classes_))

    leaf_top1_entropies = []
    for leaf_id, core_arg in zip(df["leaf_id"], df["core_args"]):
        core_arg_idx = list(model.classes_).index(core_arg)

        leaf_distribution = tree.value[leaf_id][0] * tree.n_node_samples[leaf_id]

        n_right = leaf_distribution[core_arg_idx]
        n_wrong = sum(leaf_distribution) - n_right

        leaf_entropy = order_entropy(n_right, n_wrong)
        leaf_top1_entropies.append(leaf_entropy)
    df["leaf_top1_entropy"] = leaf_top1_entropies

    for swapped_idx, swapped_class in enumerate(list(model.classes_) + unseen_classes):
        class_entropies = []

        for leaf_id, core_arg in zip(df["leaf_id"], df["core_args"]):
            core_arg_idx = list(model.classes_).index(core_arg)
            leaf_distribution = tree.value[leaf_id][0] * tree.n_node_samples[leaf_id]
            n_swapped = (
                leaf_distribution[swapped_idx]
                if swapped_idx < len(leaf_distribution)
                else 0
            )
            class_entropy = order_entropy(leaf_distribution[core_arg_idx], n_swapped)
            class_entropies.append(class_entropy)

        df[f"{swapped_class}_entropy"] = class_entropies

    tree_rules = get_tree_rules(model, tight=True)
    df["leaf_rule"] = df["leaf_id"].map(tree_rules)

    tree = model.named_steps["clf"].tree_
    leaf2var = {
        idx: model.classes_[var.item()] for idx, var in enumerate(tree.value.argmax(-1))
    }

    leaf_decision = [
        row[predictor_var] == leaf2var[row.leaf_id] for _, row in df.iterrows()
    ]
    df["leaf_decision"] = leaf_decision
    df["keep"] = (df.leaf_top1_entropy < threshold) & df.leaf_decision

    set_num_swaps(df, threshold)


def set_num_swaps(df, threshold):
    swap_so = str.maketrans({"s": "o", "o": "s"})

    correct_num_swaps = []
    all_swap_order_candidates = []

    all_orders = {"svo", "ovs", "osv", "sov", "vos", "vso"}

    for _, row in df.iterrows():
        core_arg = row.core_args
        swap_orders = all_orders - {core_arg, core_arg.translate(swap_so)}

        swap_order_candidates = [
            arg_order
            for arg_order in swap_orders
            if (row[f"{arg_order}_entropy"] < threshold) and row.leaf_decision
        ]

        num_swaps = len(swap_order_candidates)

        correct_num_swaps.append(num_swaps)
        all_swap_order_candidates.append(swap_order_candidates)

    df["num_swaps"] = correct_num_swaps
    df["swap_order_candidates"] = all_swap_order_candidates


def get_rule_for_leaf(model, leaf_id, tight=False):
    """Returns a human-readable rule for a leaf node."""
    tree = model.named_steps["clf"].tree_
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    children_left = tree.children_left
    children_right = tree.children_right
    threshold = tree.threshold
    feature = tree.feature

    # Build parent pointers
    parent_position = {}
    stack = [0]  # root node
    while stack:
        node = stack.pop()
        for child in [children_left[node], children_right[node]]:
            if child != -1:
                parent_position[child] = node
                stack.append(child)

    path = []
    node = leaf_id

    while node in parent_position:
        parent = parent_position[node]

        # Determine direction
        direction = "<=" if children_left[parent] == node else ">"

        feat = feature[parent]
        if feat != _tree.TREE_UNDEFINED:
            rule = f"{feature_names[feat]} {direction} {threshold[parent]:.4f}"
            rule = rule.replace("num__", "").replace("cat__", "")
            items = rule.split("_")
            if "<=" in rule:
                rule = (
                    "_".join(items[:-1]) + " != " + items[-1].replace(" <= 0.5000", "")
                )
            elif ">" in rule:
                rule = "_".join(items[:-1]) + " = " + items[-1].replace(" > 0.5000", "")

            if tight:
                path.append(f"[{parent}]")
            else:
                path.append(f"[{parent}] {rule}")

        node = parent

    return path[::-1]


def get_tree_rules(model, tight=False):
    tree = model.named_steps["clf"].tree_
    leaf_ids = np.where((tree.children_left == -1) & (tree.children_right == -1))[0]

    leaf_rules = {
        int(leaf): get_rule_for_leaf(model, leaf, tight=tight) for leaf in leaf_ids
    }

    if tight:
        leaf_rules = {
            leaf: " -> ".join(rule + [f"[{leaf}]"]) for leaf, rule in leaf_rules.items()
        }

    return leaf_rules
