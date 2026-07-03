import joblib
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import _tree

import numpy as np
import pandas as pd

from .entropy import order_entropy
from .prediction_target import PredictionTarget
from .process_treebank import META_FEATURES
from .utils import get_all_orders


RND = 42
OMIT_FEATURES = ["subject_idx", "object_idx", "verb_idx"] # TODO should be passed via pipeline to fit_dt kwarg omit_feats


def fit_dt(
    full_df,
    target: PredictionTarget,
    deprel_kwargs=None,
    omit_feats=None,
    save_to=None,
    max_depth=12,
    min_samples_leaf=25,
    min_impurity_decrease=0.005,
    verbose=1,
    min_df_len=10,
    leaf_threshold=0.1,
    predictor_var="deprel_order"
):
    sub_df = full_df.copy()
    if deprel_kwargs is not None:
        for k, v in deprel_kwargs.items():
            sub_df = sub_df[sub_df[k] == v]

    if len(sub_df) < min_df_len:
        return None, None, None

    omit_feats = (
        (omit_feats or set()).union(set(META_FEATURES)).union(set(OMIT_FEATURES))
    )
    omit_feats.update({col for col in full_df.columns if "idx" in col})
    omit_feats.update({col for col in full_df.columns if "_dir" in col})
    omit_feats.update({col for col in full_df.columns if "agreement" in col and not predictor_var in col})

    sub_condition_on = list(set(full_df.columns) - omit_feats - {predictor_var})

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

    X[categorical_cols] = X[categorical_cols].fillna("_missing").astype(str)
    X[numeric_cols] = X[numeric_cols].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RND
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
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

    X_train = set_dt_features_in_df(model, X_train, full_df, target, predictor_var, threshold=leaf_threshold)

    if save_to is not None:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        joblib.dump(model, save_to + ".joblib")
        X_train.to_csv(save_to + ".csv")

    if verbose > 0:
        print("Train acc", model.score(X_train, y_train))
        print("Test acc ", model.score(X_test, y_test))

    return model, X_train, y_train


def set_dt_features_in_df(
    model, df, full_df, target: PredictionTarget, predictor_var:str, additional_vars=None, threshold=0.1,
):
    """
    Augments df with DT-derived columns and returns the enriched DataFrame.

    Args:
        model: fitted sklearn Pipeline with 'preprocessor' and 'clf' steps.
        df: feature DataFrame (e.g. X_train) to assign leaf ids and entropies for.
        full_df: original full DataFrame used to re-attach omitted columns (e.g. META_FEATURES).
        target: PredictionTarget used to enumerate all possible word orders.
        predictor_var: column name of the word-order variable being predicted.
        additional_vars: extra columns from full_df to carry over into the result.
        threshold: entropy threshold below which a leaf prediction is considered confident.

    Returns:
        A new DataFrame with added columns: leaf_id, leaf_full_entropy, leaf_top1_entropy,
        per-class entropies, leaf_rule, leaf_decision, keep, num_swaps, swap_order_candidates.
    """
    X_trans = model.named_steps["preprocessor"].transform(df)
    leaf_ids = model.named_steps["clf"].apply(X_trans)

    tree = model.named_steps["clf"].tree_
    impurity = tree.impurity
    children_left = tree.children_left
    children_right = tree.children_right

    is_leaf = (children_left == -1) & (children_right == -1)
    leaf_node_ids = np.where(is_leaf)[0]
    leaf_entropy_map = dict(zip(leaf_node_ids, impurity[leaf_node_ids]))
    leaf_full_entropy = np.array([leaf_entropy_map[node] for node in leaf_ids])

    additional_vars = additional_vars or []
    additional_vars.extend(META_FEATURES + OMIT_FEATURES + [predictor_var])

    new_cols = {col: full_df[col] for col in additional_vars if col in full_df.columns}
    new_cols["leaf_id"] = pd.Series(leaf_ids, index=df.index)
    new_cols["leaf_full_entropy"] = pd.Series(leaf_full_entropy, index=df.index)

    all_orders = set(get_all_orders(predictor_var, target))
    unseen_classes = list(all_orders - set(model.classes_))

    classes_list = list(model.classes_)
    predictor_series = full_df.loc[df.index, predictor_var]

    leaf_top1_entropies = []
    for leaf_id, deprel_order in zip(leaf_ids, predictor_series):
        deprel_order_idx = classes_list.index(deprel_order)
        leaf_distribution = tree.value[leaf_id][0] * tree.n_node_samples[leaf_id]
        n_right = leaf_distribution[deprel_order_idx]
        n_wrong = sum(leaf_distribution) - n_right
        leaf_top1_entropies.append(order_entropy(n_right, n_wrong))
    new_cols["leaf_top1_entropy"] = leaf_top1_entropies

    for swapped_idx, swapped_class in enumerate(classes_list + unseen_classes):
        class_entropies = []
        for leaf_id, deprel_order in zip(leaf_ids, predictor_series):
            deprel_order_idx = classes_list.index(deprel_order)
            leaf_distribution = tree.value[leaf_id][0] * tree.n_node_samples[leaf_id]
            n_swapped = (
                leaf_distribution[swapped_idx]
                if swapped_idx < len(leaf_distribution)
                else 0
            )
            class_entropies.append(order_entropy(leaf_distribution[deprel_order_idx], n_swapped))
        new_cols[f"{swapped_class}_entropy"] = class_entropies

    tree_rules = get_tree_rules(model, tight=True)
    new_cols["leaf_rule"] = [tree_rules[node] for node in leaf_ids]

    leaf2var = {
        idx: model.classes_[var.item()] for idx, var in enumerate(tree.value.argmax(-1))
    }
    leaf_decision = [order == leaf2var[leaf_id] for leaf_id, order in zip(leaf_ids, predictor_series)]
    new_cols["leaf_decision"] = leaf_decision
    new_cols["keep"] = [e < threshold and d for e, d in zip(new_cols["leaf_top1_entropy"], leaf_decision)]

    correct_num_swaps = []
    all_swap_order_candidates = []
    for i, (deprel_order, decision) in enumerate(zip(predictor_series, leaf_decision)):
        if predictor_var == "core_args":
            swap_so = str.maketrans({"s": "o", "o": "s"})
            swap_orders = all_orders - {deprel_order, deprel_order.translate(swap_so)}
        else:
            swap_orders = all_orders - {deprel_order}
        swap_order_candidates = [
            arg_order
            for arg_order in swap_orders
            if new_cols[f"{arg_order}_entropy"][i] < threshold and decision
        ]
        correct_num_swaps.append(len(swap_order_candidates))
        all_swap_order_candidates.append(swap_order_candidates)
    new_cols["num_swaps"] = correct_num_swaps
    new_cols["swap_order_candidates"] = all_swap_order_candidates

    new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return new_df


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
