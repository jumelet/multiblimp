import math
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline


def order_entropy(
    n_ab: float,
    n_ba: float,
    smoothing_a: float = 0.5,  # Jeffreys prior
) -> float:
    """
    Compute Shannon entropy (in bits) for word order frequencies.

    Args:
        n_ab (int): Count of A>B order
        n_ba (int): Count of B>A order
        smoothing_a (float): Smoothing factor

    Returns:
        float: Entropy in bits
    """
    n_ab += smoothing_a
    n_ba += smoothing_a

    total = n_ab + n_ba
    if total == 0:
        return 0.0

    p_ab = n_ab / total
    p_ba = n_ba / total

    # avoid log(0) issues
    def safe_term(p):
        return -p * math.log2(p) if p > 0 else 0.0

    return safe_term(p_ab) + safe_term(p_ba)


def calculate_base_entropy(
    df: pd.DataFrame, target_col: str, binary: bool = False, smoothing: float = 0.5
) -> float:
    """Calculate entropy of word order distribution.

    Args:
        df: DataFrame containing the data
        target_col: Column name containing word order labels
        binary: If True, calculate binary entropy (majority-class vs rest).
                If False, calculate six-class entropy.
        smoothing: Smoothing factor for entropy calculation (Jeffreys prior)

    Returns:
        Entropy value in bits
    """
    value_counts = df[target_col].value_counts()

    if binary:
        # Binary entropy: majority class vs. rest
        n_majority = value_counts.iloc[0]  # Most frequent class
        n_rest = len(df) - n_majority
        return order_entropy(n_majority, n_rest, smoothing_a=smoothing)
    else:
        # Six-class entropy with smoothing
        counts = value_counts.values
        smoothed_counts = counts + smoothing
        total = smoothed_counts.sum()
        probabilities = smoothed_counts / total

        return sum(-p * math.log2(p) if p > 0 else 0.0 for p in probabilities)


def calculate_tree_entropy(
    dt: Pipeline,
    df: pd.DataFrame,
    target_col: str,
    binary: bool = False,
    smoothing: float = 0.5,
) -> float:
    """Calculate weighted entropy after decision tree split.

    Args:
        dt: Fitted sklearn Pipeline containing the decision tree
        df: DataFrame containing the features
        target_col: Column name containing word order labels
        binary: If True, calculate binary entropy. If False, six-class entropy.
        smoothing: Smoothing factor for entropy calculation

    Returns:
        Weighted average entropy of leaf nodes
    """
    # Get the decision tree classifier from the pipeline
    tree_model = dt.named_steps["clf"]

    # Prepare features (drop target column)
    X = df.drop(columns=[target_col])

    # Transform features through preprocessor and get leaf assignments
    leaf_ids = tree_model.apply(dt.named_steps["preprocessor"].transform(X))

    # Calculate entropy for each leaf
    weighted_entropy = 0.0
    total_samples = len(df)

    for leaf_id in np.unique(leaf_ids):
        leaf_mask = leaf_ids == leaf_id
        leaf_df = df[leaf_mask]
        leaf_weight = len(leaf_df) / total_samples
        leaf_entropy = calculate_base_entropy(
            leaf_df, target_col, binary=binary, smoothing=smoothing
        )
        weighted_entropy += leaf_weight * leaf_entropy

    return weighted_entropy
