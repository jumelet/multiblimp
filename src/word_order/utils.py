import itertools

from .prediction_target import PredictionTarget


ALL_CORE_ARGS = ["vos", "vso", "ovs", "svo", "osv", "sov"]

MAX_TREEBANK_LEN = 30_000


def capitalize_first(word: str) -> str:
    if len(word) == 0:
        return ""
    return word[0].upper() + word[1:]


def shorten_cls(cls, target):
    if isinstance(target.head_pos, list) and len(target.head_pos) == 1:
        head_pos = target.head_pos[0][0]
    else:
        head_pos = "H"

    cls_swap = {
        "det": "d",
        "nummod": "n",
        "amod": "a",
        "head": head_pos,
        "nsubj": "s",
        "obj": "o",
        "_": "",
        "acl:relcl": "v",
        "obl": "c",
        "iobj": "i",
        "advmod": "a",
        "nmod": "n",
    }

    result = cls
    for old, new in cls_swap.items():
        result = result.replace(old, new)

    return result


def get_all_orders(predictor_var: str, target: PredictionTarget):
    if predictor_var == "core_args":
        all_orders = ALL_CORE_ARGS
    else:
        deprels = target.child_deprels + ["head"]
        all_orders = [
            shorten_cls("_".join(permutation), target)
            for permutation in sorted(
                itertools.permutations(deprels, len(deprels)),
                key=lambda p: (p.index("head"), "_".join(p)),
            )
        ]

    return all_orders
