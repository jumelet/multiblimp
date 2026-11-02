"""Create the final word order minimal pairs for all languages.

Per (lang, deprel), pairs follow the cascade in ORDER_STRENGTHS: strict and
near-strict leaves of the (non-lexical) DT in dt_df/, and only if those yield fewer
than --min_total_items pairs, the strict leaves of a lexical DT (fitted with
form/lemma features). Combos for which no step reaches --min_total_items get no
file. Lexical DTs are stored in dt_df_lexical/ and reused when they already exist.

Expects the outputs of the earlier pipeline steps in --data_dir: resources/ (UD
pickles), treebank_features/, dt_df/ and joint_classifiers/.

Resumable: every finished (lang, deprel), including those without pairs, is
recorded in done.txt in the output directory and skipped on a rerun. With
--overwrite, the given languages are redone from scratch, including their lexical
DTs; to resume an interrupted overwrite, rerun without --overwrite.
"""

import argparse
import os
import sys
import traceback
from glob import glob

import joblib
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from word_order.parallel import map_languages, parse_workers
from word_order.create_tree_pairs import create_pairs_tree
from word_order.decision_tree import fit_word_order_dt
from word_order.prediction_target import get_all_deprel_targets
from word_order.process_treebank import load_treebank
from word_order.utils import SAMPLED_PICKLE_DIR


def lexical_dt_df(data_dir, lang, target, refit=False):
    """Load the lexical DT of (lang, deprel), or fit and save it if it doesn't exist."""
    path = os.path.join(data_dir, "dt_df_lexical", "_".join(target.child_deprels), f"{lang}.parquet")
    if os.path.exists(path) and not refit:
        return pd.read_parquet(path)
    _, dt_df = fit_word_order_dt(data_dir, lang, target, lexical=True)
    return dt_df


def create_language_pairs(
    lang: str,
    data_dir: str,
    out_dir: str,
    done: set,
    max_total_items: int,
    min_total_items: int,
    validate_threshold: float,
    overwrite: bool,
) -> list[str]:
    """Create the pairs of every deprel of one language; returns the finished (lang/deprel) keys."""
    todo = [
        t
        for t in get_all_deprel_targets()
        if f"{lang}/{'_'.join(t.child_deprels)}" not in done
        and os.path.exists(os.path.join(data_dir, "dt_df", "_".join(t.child_deprels), f"{lang}.parquet"))
    ]
    if not todo:
        print(f"[{lang}] done", flush=True)
        return []

    treebank = load_treebank(lang, os.path.join(data_dir, SAMPLED_PICKLE_DIR))
    classifier = joblib.load(os.path.join(data_dir, "joint_classifiers", f"{lang}.joblib"))

    finished = []
    for target in todo:
        deprel = "_".join(target.child_deprels)
        save_to = os.path.join(out_dir, deprel, f"{lang}.parquet")
        if os.path.exists(save_to):
            os.remove(save_to)  # combos without enough pairs write no new file
        kwargs = dict(
            max_total_items=max_total_items,
            min_total_items=min_total_items,
            balance_features=["sen_len", "head_form", "leaf_id"] + [f"{d}_form" for d in target.child_deprels],
            classifier=classifier,
            validate_threshold=validate_threshold,
            save_to=save_to,
        )
        try:
            dt_df = pd.read_parquet(os.path.join(data_dir, "dt_df", deprel, f"{lang}.parquet"))
            swap_df = create_pairs_tree(dt_df, treebank, target, **kwargs)
            if swap_df is None:
                lexical = lexical_dt_df(data_dir, lang, target, refit=overwrite)
                if lexical is not None:
                    swap_df = create_pairs_tree(dt_df, treebank, target, lexical_dt_df=lexical, **kwargs)
        except Exception:
            print(f"[{lang} {deprel}] FAILED\n{traceback.format_exc()}", flush=True)
            continue

        n_pairs = 0 if swap_df is None else len(swap_df)
        strength = swap_df.order_strength.iloc[0] if n_pairs else "-"
        print(f"[{lang} {deprel}] {n_pairs} pairs ({strength})", flush=True)
        finished.append(f"{lang}/{deprel}")

    print(f"[{lang}] done", flush=True)
    return finished


def create_pairs(
    data_dir: str,
    langs: list[str] | None = None,
    out_dir: str | None = None,
    max_total_items: int = 500,
    min_total_items: int = 10,
    validate_threshold: float = 0.9,
    overwrite: bool = False,
    workers: int = 1,
) -> None:
    out_dir = out_dir or os.path.join(data_dir, "final_pairs")
    os.makedirs(out_dir, exist_ok=True)
    if langs is None:
        langs = sorted(
            {os.path.basename(p)[: -len(".parquet")] for p in glob(os.path.join(data_dir, "dt_df", "*", "*.parquet"))}
        )
    langs = [lang.replace(" ", "_") for lang in langs]

    done_path = os.path.join(out_dir, "done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    if overwrite:
        done = {key for key in done if key.split("/")[0] not in langs}
        with open(done_path, "w") as f:
            f.writelines(f"{key}\n" for key in sorted(done))
        for lang in langs:
            for path in glob(os.path.join(out_dir, "*", f"{lang}.parquet")):
                os.remove(path)

    def record_done(lang, keys):
        with open(done_path, "a") as f:
            f.writelines(f"{key}\n" for key in keys)

    map_languages(
        create_language_pairs,
        langs,
        workers,
        on_result=record_done,
        step="create_pairs",
        data_dir=data_dir,
        out_dir=out_dir,
        done=done,
        max_total_items=max_total_items,
        min_total_items=min_total_items,
        validate_threshold=validate_threshold,
        overwrite=overwrite,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Directory with the outputs of the earlier pipeline steps")
    parser.add_argument("--out_dir", default=None, help="Defaults to {data_dir}/final_pairs")
    parser.add_argument("--langs", nargs="+", default=None, help="Defaults to all languages with a DT")
    parser.add_argument("--max_total_items", type=int, default=500)
    parser.add_argument("--min_total_items", type=int, default=10)
    parser.add_argument("--validate_threshold", type=float, default=0.9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=parse_workers,
        default=1,
        help='Maximum languages processed in parallel, or "auto" (one per CPU core); also limited by memory',
    )
    args = parser.parse_args()
    create_pairs(
        args.data_dir,
        args.langs,
        args.out_dir,
        args.max_total_items,
        args.min_total_items,
        args.validate_threshold,
        args.overwrite,
        args.workers,
    )


if __name__ == "__main__":
    main()
