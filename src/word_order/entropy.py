import math
from collections import Counter

from scipy.stats import entropy
import numpy as np


def order_entropy(
    n_ab: int,
    n_ba: int,
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


def calc_dep_entropy(df):
    deprel_dirs = dict(df.groupby("deprel").sum("dir")["dir"])
    deprel_counts = Counter(df.deprel)
    deprel_entropy = {}

    for deprel, count in deprel_counts.items():
        deprel_entropy[deprel] = order_entropy(
            deprel_dirs[deprel].item(),
            count - deprel_dirs[deprel].item(),
        )

    return deprel_entropy


def calc_model_entropy(model, df):
    """Calculate the entropy of the data in `df` for a DecisionTreeClassifier."""
    probs = model.predict_proba(df)
    entropies = entropy(probs.T, base=2)
    mean_entropy = np.mean(entropies)

    return mean_entropy
