from collections import Counter
from tqdm import tqdm
from typing import Optional

from .unimorph import allval2um, UNDEFINED


def get_feature_combinations(
    corpus,
    inflector,
    context_inflector,
    ufeat: Optional[str] = None,
    head_ud_features={},
    child_ud_features={},
    verbose=False,
    tqdm_progress=True,
    only_try_ud_if_no_um=True,
    discard_undefined=True,
    allow_undefined=False,
):
    feature_combinations = Counter()

    head2feat = {}
    child2feat = {}

    iterator = tqdm(corpus) if tqdm_progress else corpus

    for _, item in iterator:
        if item["head"] not in head2feat:
            head_features = inflector.get_form_features(
                item["head"], head_ud_features, ufeat=ufeat, only_try_ud_if_no_um=only_try_ud_if_no_um
            )
            if discard_undefined:
                head_features.discard(UNDEFINED)
            head2feat[item["head"]] = head_features
        else:
            head_features = head2feat[item["head"]]

        if item["child"] not in child2feat:
            child_features = context_inflector.get_form_features(
                item["child"], child_ud_features, ufeat=ufeat, only_try_ud_if_no_um=only_try_ud_if_no_um
            )
            if discard_undefined:
                child_features.discard(UNDEFINED)
            child2feat[item["child"]] = child_features
        else:
            child_features = child2feat[item["child"]]

        if len(child_features) == 0:
            ud_child_feat = item["child_features"].get(ufeat)
            if ud_child_feat is not None:
                ud_child_feat = allval2um(ud_child_feat)
                child_features = set(ud_child_feat)

        if len(head_features) == 0:
            ud_head_feat = item["head_features"].get(ufeat)
            if ud_head_feat is not None:
                ud_child_feat = allval2um(ud_head_feat)
                head_features = set(ud_child_feat)

        head_features = list(head_features)
        child_features = list(child_features)

        order = "SV" if item["distance"] > 0 else "VS"

        if (
            (len(head_features) == 1)
            and (len(child_features) == 1)
            and (UNDEFINED not in [child_features[0], head_features[0]])
        ):
            feature_combinations[child_features[0], head_features[0], order] += 1
        elif (
            (len(head_features) == 1)
            and (head_features[0] != UNDEFINED)
            and (head_features[0] in child_features)
        ):
            feature_combinations[head_features[0], head_features[0], order] += 1
        elif (
            (len(child_features) == 1)
            and (child_features[0] != UNDEFINED)
            and (child_features[0] in head_features)
        ):
            feature_combinations[child_features[0], child_features[0], order] += 1
        elif (
            allow_undefined
            and (len(head_features) == 1)
            and (head_features[0] != UNDEFINED)
            and (UNDEFINED in child_features or len(child_features) == 0)
        ):
            feature_combinations[head_features[0], head_features[0], order] += 1
        elif (
            allow_undefined
            and (len(child_features) == 1)
            and (child_features[0] != UNDEFINED)
            and (UNDEFINED in head_features or len(head_features) == 0)
        ):
            feature_combinations[child_features[0], child_features[0], order] += 1

    tot = sum(feature_combinations.values())

    child_tot = Counter()
    head_tot = Counter()
    for (childfeat, headfeat, order), freq in feature_combinations.items():
        child_tot[childfeat, order] += freq
        head_tot[headfeat, order] += freq

    rel_combs = Counter()
    child_rel_combs = Counter()
    head_rel_combs = Counter()
    for (childfeat, headfeat, order), freq in feature_combinations.items():
        rel_combs[childfeat, headfeat, order] = freq / tot
        child_rel_combs[childfeat, headfeat, order] = freq / child_tot[childfeat, order]
        head_rel_combs[childfeat, headfeat, order] = freq / head_tot[headfeat, order]

    if verbose:
        if len(rel_combs) > 0:
            print(
                f"{'φ_N':<8}{'φ_V':<8}{'SV/VS':<8}"
                f"{'P(φ_N,φ_V,SV)':<17}{'P(φ_V|φ_N,SV)':<17}{'P(φ_N|φ_V,SV)':<17}"
            )
        for childfeat, headfeat, order in sorted(rel_combs.keys()):
            print(
                f"{childfeat:<8}{headfeat:<8}{order:<8}"
                f"{rel_combs[childfeat, headfeat, order] * 100:<17.1f}"
                f"{child_rel_combs[childfeat, headfeat, order] * 100:<17.1f}"
                f"{head_rel_combs[childfeat, headfeat, order] * 100:<17.1f}"
            )

    print(feature_combinations)

    return feature_combinations, rel_combs, child_rel_combs, head_rel_combs
