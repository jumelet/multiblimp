from collections import Counter
from tqdm import tqdm

from .unimorph import allval2um, UNDEFINED


def measure_orders(treebank):
    dep_orders = Counter()

    iterator = tqdm(corpus) if tqdm_progress else corpus

    for _, item in iterator:
        order = "SV" if item["distance"] > 0 else "VS"
        cop_lemma = item["cop_features"].get("lemma") if condition_cop_lemma else None

        head_features = set()
        child_features = set()

        if collocate_ud_features:
            if head_ufeat in item["head_features"]:
                head_features = allval2um(item["head_features"][head_ufeat])
            if child_ufeat in item["child_features"]:
                child_features = allval2um(item["child_features"][child_ufeat])

            if len(head_features) > 0 and len(child_features) > 0:
                feature_combinations[child_features, head_features, order, cop_lemma] += 1
                continue

        if only_collocate_ud_features:
            continue

        if len(head_features) == 0:
            if item["head"] not in head2feat:
                head_features = head_inflector.get_form_features(
                    item["head"].lower(),
                    head_ud_features,
                    only_try_ud_if_no_um=only_try_ud_if_no_um,
                )
                if discard_undefined:
                    head_features.discard(UNDEFINED)
                head2feat[item["head"]] = head_features
            else:
                head_features = head2feat[item["head"]]

        if len(child_features) == 0:
            if item["child"] not in child2feat:
                child_features = child_inflector.get_form_features(
                    item["child"].lower(),
                    child_ud_features,
                    only_try_ud_if_no_um=only_try_ud_if_no_um,
                )
                if discard_undefined:
                    child_features.discard(UNDEFINED)
                child2feat[item["child"]] = child_features
            else:
                child_features = child2feat[item["child"]]

        if isinstance(head_features, str):
            head_features = [head_features]
        else:
            head_features = list(head_features)

        if isinstance(child_features, str):
            child_features = [child_features]
        else:
            child_features = list(child_features)

        if (
            (len(head_features) == 1)
            and (len(child_features) == 1)
            and (UNDEFINED not in [child_features[0], head_features[0]])
        ):
            feature_combinations[child_features[0], head_features[0], order, cop_lemma] += 1
        elif (
            allow_multiple
            and (len(head_features) == 1)
            and (head_features[0] != UNDEFINED)
            and (head_features[0] in child_features)
        ):
            feature_combinations[head_features[0], head_features[0], order, cop_lemma] += 1
        elif (
            allow_multiple
            and (len(child_features) == 1)
            and (child_features[0] != UNDEFINED)
            and (child_features[0] in head_features)
        ):
            feature_combinations[child_features[0], child_features[0], order, cop_lemma] += 1
        elif (
            allow_undefined
            and (len(head_features) == 1)
            and (head_features[0] != UNDEFINED)
            and (UNDEFINED in child_features or len(child_features) == 0)
        ):
            feature_combinations[head_features[0], head_features[0], order, cop_lemma] += 1
        elif (
            allow_undefined
            and (len(child_features) == 1)
            and (child_features[0] != UNDEFINED)
            and (UNDEFINED in head_features or len(head_features) == 0)
        ):
            feature_combinations[child_features[0], child_features[0], order, cop_lemma] += 1
        else:
            if (child_ufeat in item["child_features"]) and (head_ufeat in item["head_features"]):
                ud_head_ufeat = allval2um(item["head_features"][head_ufeat])
                ud_child_ufeat = allval2um(item["child_features"][child_ufeat])

                print(item['child'], child_features, ud_child_ufeat, item['head'], head_features, ud_head_ufeat)

    tot = sum(feature_combinations.values())

    child_tot = Counter()
    head_tot = Counter()
    for (childfeat, headfeat, order, cop_lemma), freq in feature_combinations.items():
        child_tot[childfeat, order, cop_lemma] += freq
        head_tot[headfeat, order, cop_lemma] += freq

    rel_combs = Counter()
    child_rel_combs = Counter()
    head_rel_combs = Counter()
    for (childfeat, headfeat, order, cop_lemma), freq in feature_combinations.items():
        rel_combs[childfeat, headfeat, order, cop_lemma] = freq / tot
        child_rel_combs[childfeat, headfeat, order, cop_lemma] = freq / child_tot[childfeat, order, cop_lemma]
        head_rel_combs[childfeat, headfeat, order, cop_lemma] = freq / head_tot[headfeat, order, cop_lemma]

    if verbose:
        if len(rel_combs) > 0:
            print(
                f"{'φ_N':<8}{'φ_V':<8}{'SV/VS':<8}{'lemma':<8}"
                f"{'P(φ_N,φ_V,SV)':<17}{'P(φ_V|φ_N,SV)':<17}{'P(φ_N|φ_V,SV)':<17}"
            )
        for childfeat, headfeat, order, cop_lemma in sorted(rel_combs.keys()):
            print(
                f"{childfeat:<8}{headfeat:<8}{order:<8}{cop_lemma or '':<8}"
                f"{rel_combs[childfeat, headfeat, order, cop_lemma] * 100:<17.1f}"
                f"{child_rel_combs[childfeat, headfeat, order, cop_lemma] * 100:<17.1f}"
                f"{head_rel_combs[childfeat, headfeat, order, cop_lemma] * 100:<17.1f}"
            )

    print(feature_combinations)

    return feature_combinations, rel_combs, child_rel_combs, head_rel_combs
