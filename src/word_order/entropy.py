import math
from collections import Counter


def order_entropy(n_ab: int, n_ba: int) -> float:
    """
    Compute Shannon entropy (in bits) for word order frequencies.

    Args:
        n_ab (int): Count of A>B order
        n_ba (int): Count of B>A order

    Returns:
        float: Entropy in bits
    """
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
