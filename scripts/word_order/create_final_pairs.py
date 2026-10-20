"""Create the final word order minimal pairs for all languages.

Per (lang, deprel), pairs follow the cascade in ORDER_STRENGTHS: strict and
near-strict leaves of the (non-lexical) DT in dt_df/, and only if those yield fewer
than --min_total_items pairs, the strict leaves of a lexical DT (fitted with
form/lemma features). Combos for which no step reaches --min_total_items get no
file. Lexical DTs are stored in dt_df_lexical/ (with their trees in
html_lexical/) and reused when they already exist.

Expects the outputs of the earlier pipeline steps in --data_dir: resources/ (UD
pickles), treebank_features/, dt_df/ and joint_classifiers/.

Resumable: every finished (lang, deprel), including those without pairs, is
recorded in done.txt in the output directory and skipped on a rerun.
"""

import argparse
import math
import os
import sys
import traceback
from glob import glob

import joblib
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from word_order.create_tree_pairs import create_pairs_tree
from word_order.decision_tree import fit_dt
from word_order.prediction_target import get_all_deprel_targets
from word_order.process_treebank import load_treebank, read_df
from word_order.utils import MAX_TREEBANK_LEN
from word_order.viz_tree import tree2html


def get_impurity(n, min_n=300, max_n=4000, max_val=0.1, min_val=0.01):
    """min_impurity_decrease for a DT on n rows (same as for the non-lexical DTs)."""
    if n <= min_n:
        return max_val
    if n >= max_n:
        return min_val
    t = (math.log(n) - math.log(min_n)) / (math.log(max_n) - math.log(min_n))
    return max_val - t * (max_val - min_val)


def lexical_dt_df(data_dir, lang, target):
    """Load the lexical DT of (lang, deprel), or fit and save it if it doesn't exist."""
    deprel = "_".join(target.child_deprels)
    path = os.path.join(data_dir, "dt_df_lexical", deprel, lang)
    if os.path.exists(path + ".parquet"):
        return pd.read_parquet(path + ".parquet")

    full_df = read_df(lang, word_order_dir=os.path.join(data_dir, "treebank_features", deprel))
    full_df = full_df[full_df["deprel_order"].notnull()]
    model, dt_df, _ = fit_dt(
        full_df,
        "decision_tree",
        target=target,
        verbose=0,
        min_impurity_decrease=get_impurity(len(full_df)),
        min_samples_leaf=10,
        save_to=path,
        omit_feats=set(),  # keep the form/lemma features
    )
    if model is None:
        return None
    tree2html(
        model,
        dt_df,
        full_df,
        "deprel_order",
        os.path.join(data_dir, "html_lexical", deprel, f"{lang}.html"),
        target=target,
        max_rows=15,
        meta={"Language": f"{lang} (lexical)"},
        only_show_real_orders=True,
        correlate_features=True,
    )
    return dt_df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="/media/jaap/81b6ce8a-28e5-4eda-9c68-b13e0637cc4f/WORD_ORDER")
    parser.add_argument("--out_dir", default=None, help="Defaults to {data_dir}/final_pairs")
    parser.add_argument("--langs", nargs="+", default=None, help="Defaults to all languages with a DT")
    parser.add_argument("--max_total_items", type=int, default=500)
    parser.add_argument("--min_total_items", type=int, default=10)
    parser.add_argument("--validate_threshold", type=float, default=0.9)
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir or os.path.join(data_dir, "final_pairs")
    os.makedirs(out_dir, exist_ok=True)
    done_path = os.path.join(out_dir, "done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()

    targets = list(get_all_deprel_targets())
    langs = args.langs or sorted(
        {os.path.basename(p)[: -len(".parquet")] for p in glob(os.path.join(data_dir, "dt_df", "*", "*.parquet"))}
    )

    for lang in langs:
        todo = [
            t
            for t in targets
            if f"{lang}/{'_'.join(t.child_deprels)}" not in done
            and os.path.exists(os.path.join(data_dir, "dt_df", "_".join(t.child_deprels), f"{lang}.parquet"))
        ]
        if not todo:
            continue

        treebank = load_treebank(lang, os.path.join(data_dir, "resources"), max_treebank_len=MAX_TREEBANK_LEN)
        classifier = joblib.load(os.path.join(data_dir, "joint_classifiers", f"{lang}.joblib"))

        for target in todo:
            deprel = "_".join(target.child_deprels)
            kwargs = dict(
                max_total_items=args.max_total_items,
                min_total_items=args.min_total_items,
                balance_features=["sen_len", "head_form", "leaf_id"] + [f"{d}_form" for d in target.child_deprels],
                classifier=classifier,
                validate_threshold=args.validate_threshold,
                save_to=os.path.join(out_dir, deprel, f"{lang}.parquet"),
            )
            try:
                dt_df = pd.read_parquet(os.path.join(data_dir, "dt_df", deprel, f"{lang}.parquet"))
                swap_df = create_pairs_tree(dt_df, treebank, target, **kwargs)
                if swap_df is None:
                    lexical = lexical_dt_df(data_dir, lang, target)
                    if lexical is not None:
                        swap_df = create_pairs_tree(dt_df, treebank, target, lexical_dt_df=lexical, **kwargs)
            except Exception:
                print(f"[{lang} {deprel}] FAILED\n{traceback.format_exc()}", flush=True)
                continue

            n_pairs = 0 if swap_df is None else len(swap_df)
            strength = swap_df.order_strength.iloc[0] if n_pairs else "-"
            print(f"[{lang} {deprel}] {n_pairs} pairs ({strength})", flush=True)
            with open(done_path, "a") as f:
                f.write(f"{lang}/{deprel}\n")

        print(f"[{lang}] done", flush=True)


if __name__ == "__main__":
    main()
