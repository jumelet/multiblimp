import os
import pickle
from typing import *

from iso639 import Lang
import numpy as np
import pandas as pd
from scipy.stats import binom_test

from .unimorph import allval2um


BALANCE_FEATURES = [
    "child",
    "head",
    "distance",
    # "num_congruent_attractors",
    # "num_incongruent_attractors",
    "has_attractors",
    "only_has_congruent_attractors",
    "only_has_incongruent_attractors",
    "sen",
]


def downsample_pairs(
    lang_probs,
    lang: str,
    phenomenon: str,
    n_items: int,
    output_dir: str,
    minimal_pair_dir: str = "../minimal_pairs",
    balance_features: List[str] = BALANCE_FEATURES,
    split_aux: bool = True,
    min_total_pairs: int = 5,
):
    filename = os.path.join(minimal_pair_dir, phenomenon, f"{lang}.tsv")
    lang_dir = os.path.join(output_dir, lang)
    ufeat = {"#": "Number", "G": "Gender", "P": "Person"}[phenomenon[2]]

    if not os.path.exists(filename):
        return

    df = pd.read_csv(filename, sep="\t")

    if len(df) < min_total_pairs:
        return

    if split_aux:
        aux_df = df[~pd.isna(df.cop)].copy()
        v_df = df[pd.isna(df.cop)].copy()

        create_conditions(
            lang_probs,
            v_df,
            n_items,
            balance_features,
            lang_dir,
            "verb",
            ufeat,
            min_total_pairs=min_total_pairs,
        )
        create_conditions(
            lang_probs,
            aux_df,
            n_items,
            balance_features,
            lang_dir,
            "aux",
            ufeat,
            min_total_pairs=min_total_pairs,
        )
    else:
        create_conditions(
            lang_probs,
            df,
            n_items,
            balance_features,
            lang_dir,
            "ptcp",
            ufeat,
            min_total_pairs=min_total_pairs,
        )


def create_conditions(
    lang_probs,
    df,
    n_items,
    balance_features,
    lang_dir: str,
    suffix: str,
    ufeat: str,
    p0: float = 0.9,
    alpha: float = 0.1,
    min_total_pairs: int = 10,
    min_cond_pairs: int = 1,
):
    counts, cond_probs = lang_probs[0], lang_probs[2]

    for condition in df.feature_vals.unique():
        if "|" in condition:
            continue

        for order in ["SV", "VS"]:
            gram_val, ungram_val = condition.split(" -> ")

            gram_key = gram_val, gram_val, order

            child_order_count = sum(freq for (c, _, o), freq in counts.items() if (c == gram_val) and (o == order))
            p_agreement = binom_test(counts[gram_key], n=child_order_count, p=p0, alternative='greater')
            p_no_agreement = binom_test(counts[gram_key], n=child_order_count, p=p0, alternative='less')

            if counts[gram_key] == 0:
                asterisk = "?"
            else:
                if p_agreement <= (alpha/2):
                    asterisk = "+"
                elif (counts[gram_key] > min_total_pairs) and (counts[gram_key]/child_order_count) > p0:
                    asterisk = "~"
                elif p_no_agreement <= (alpha/2):
                    asterisk = None
                else:
                    asterisk = "?"

            # This mask also takes feature_vals of shape "3 -> 1|2" into account
            condition_mask = [
                (
                    gram_val in feature_val.split(' -> ')[0]
                    and ungram_val in feature_val.split(' -> ')[1]
                )
                for feature_val in df.feature_vals
            ]
            sub_df = df[condition_mask]

            if order == "SV":
                sub_df = sub_df[sub_df.distance > 0]
            else:
                sub_df = sub_df[sub_df.distance < 0]

            if (
                asterisk is None
                or (len(sub_df) < min_cond_pairs)
            ):
                if len(sub_df) > 0:
                    if asterisk is None:
                        print(f"SKIPPING {len(sub_df)} items | low prob:", lang_dir, gram_val, ungram_val, order)
                    else:
                        print(f"SKIPPING {len(sub_df)} items | little data:", lang_dir, gram_val, ungram_val, order)
            else:
                sub_df = set_attractors(sub_df, ufeat, gram_val)

                balanced_df = balance_df(sub_df, n_items, balance_features)

                filename = f"{gram_val}_{ungram_val}_{order}_{suffix}_{len(balanced_df)}{asterisk}.tsv"
                output_file = os.path.join(lang_dir, filename)

                if not os.path.exists(lang_dir):
                    os.makedirs(lang_dir, exist_ok=True)

                balanced_df = balanced_df.loc[:, ~balanced_df.columns.str.contains('^Unnamed')]
                balanced_df.to_csv(output_file, sep="\t", index=False)

                print(filename)


def set_attractors(df, ufeat, gram_val):
    df["attractors"] = [eval(x) for x in df.attractors]
    df["congruent_attractors"] = [
        [
            attr
            for attr in attractors
            if allval2um(attr['feats'][ufeat]) == gram_val
        ]
        for attractors in df.attractors
    ]
    df["incongruent_attractors"] = [
        [
            attr
            for attr in attractors
            if allval2um(attr['feats'][ufeat]) != gram_val
        ]
        for attractors in df.attractors
    ]

    df["num_attractors"] = [len(x) for x in df.attractors]
    df["num_congruent_attractors"] = [len(x) for x in df.congruent_attractors]
    df["num_incongruent_attractors"] = [len(x) for x in df.incongruent_attractors]

    df["has_attractors"] = df.num_attractors > 0
    df["only_has_congruent_attractors"] = (df.num_congruent_attractors > 0) & (df.num_incongruent_attractors == 0)
    df["only_has_incongruent_attractors"] = (df.num_incongruent_attractors > 0) & (df.num_congruent_attractors == 0)

    return df


def balance_df(df, n_items, balance_features):
    feature_distributions = {
        feat: Counter(df[feat])
        for feat in balance_features
    }
    feature_totals = {
        feat: sum(feature_distributions[feat].values())
        for feat in balance_features
    }
    df['sample_weight'] = [
        1 / np.prod([feature_distributions[feat][row[feat]] / feature_totals[feat] for feat in balance_features])
        for _, row in df.iterrows()
    ]
    df.sample_weight /= sum(df.sample_weight)

    ids = np.random.choice(
        range(len(df)),
        size=min(n_items, len(df)),
        p=df.sample_weight,
        replace=False,
    )

    return df.iloc[ids]
