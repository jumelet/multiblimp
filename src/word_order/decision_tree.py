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

RND = 42


def fit_dt(
    df,
    deprel,
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
    sub_df = df[(df.deprel == deprel)]
    if deprel_kwargs is not None:
        for k, v in deprel_kwargs.items():
            sub_df = sub_df[sub_df[k] == v]

    if len(sub_df) < min_df_len:
        return None, None

    omit_feats = (omit_feats or set()).union(set(META_FEATURES))

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
            or ("sibling" in feat and not "lemma" in feat)
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
        joblib.dump(model, save_to)

    if verbose > 0:
        print("Train acc", model.score(X_train, y_train))
        print("Test acc ", model.score(X_test, y_test))

    return model, X


def pprint_node(txt):
    new_txt = txt.replace(" <= 0.5", "").replace("000", "").replace(".0,", ",").replace(".0]", "]").strip()

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


def plot_dt(model, save_to=None, show_plot=True):
    set_config(transform_output="default")

    clf = model.named_steps["clf"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = [x.split("__")[1] for x in preprocessor.get_feature_names_out()]

    fig = plt.figure(figsize=(25, 15))
    artists = plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["L", "R"],
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

def set_dt_meta(model, df, full_df, predictor_var="dir"):
    X_trans = model.named_steps["preprocessor"].transform(df)
    leaf_ids = model.named_steps["clf"].apply(X_trans)

    tree = model.named_steps["clf"].tree_
    impurity = tree.impurity
    children_left = tree.children_left
    children_right = tree.children_right

    # Map leaf → entropy
    is_leaf = (children_left == -1) & (children_right == -1)
    leaf_node_ids = np.where(is_leaf)[0]
    leaf_entropy_map = dict(zip(leaf_node_ids, impurity[leaf_node_ids]))

    # Vectorized lookup
    leaf_entropy = np.array([leaf_entropy_map[node] for node in leaf_ids])

    # Add columns
    for col in [
        "sen",
        "sent_id",
        "treebank",
        "child_idx",
        "tree_idx",
        "head_idx",
        predictor_var,
    ]:
        df[col] = full_df[col]

    df["leaf_id"] = leaf_ids
    df["leaf_entropy"] = leaf_entropy

    tree_rules = get_tree_rules(model, tight=True)
    df["leaf_rule"] = df["leaf_id"].map(tree_rules)


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
