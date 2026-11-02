"""Fit the (non-lexical) word order decision tree per language and deprel.

Writes {data_dir}/dt_df/{deprel}/{lang}.joblib (the model) and .parquet (the
training rows with their leaf), from the features in treebank_features/.
Existing DTs are skipped unless --overwrite is given.
"""

import argparse
import os
import sys
from glob import glob

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from word_order.parallel import map_languages, parse_workers
from word_order.decision_tree import fit_word_order_dt
from word_order.prediction_target import get_all_deprel_targets

def fit_language_dts(lang: str, data_dir: str, deprels: list[str] | None = None, overwrite: bool = False):
    for target in get_all_deprel_targets():
        deprel = "_".join(target.child_deprels)
        if deprels is not None and deprel not in deprels:
            continue
        has_features = os.path.exists(os.path.join(data_dir, "treebank_features", deprel, f"{lang}.parquet"))
        model_path = os.path.join(data_dir, "dt_df", deprel, f"{lang}.joblib")
        if not has_features:
            if overwrite and os.path.exists(model_path):
                os.remove(model_path)
                os.remove(model_path[: -len(".joblib")] + ".parquet")
            continue
        if os.path.exists(model_path) and not overwrite:
            continue
        model, _ = fit_word_order_dt(data_dir, lang, target, lexical=False)
        print(f"[{lang} {deprel}] {'fitted' if model is not None else 'no DT'}", flush=True)
    print(f"[{lang}] done", flush=True)


def fit_dts(
    data_dir: str,
    langs: list[str] | None = None,
    deprels: list[str] | None = None,
    overwrite: bool = False,
    workers: int = 1,
) -> None:
    if langs is None:
        paths = glob(os.path.join(data_dir, "treebank_features", "*", "*.parquet"))
        langs = sorted({os.path.basename(p)[: -len(".parquet")] for p in paths})
    langs = [lang.replace(" ", "_") for lang in langs]
    map_languages(
        fit_language_dts,
        langs,
        workers,
        step="fit_dts",
        data_dir=data_dir,
        deprels=deprels,
        overwrite=overwrite,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Directory with treebank_features/")
    parser.add_argument("--langs", nargs="+", default=None, help="Defaults to all languages with features")
    parser.add_argument("--deprels", nargs="+", default=None, help="Defaults to all targets")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=parse_workers,
        default=1,
        help='Maximum languages processed in parallel, or "auto" (one per CPU core); also limited by memory',
    )
    args = parser.parse_args()
    fit_dts(args.data_dir, args.langs, args.deprels, args.overwrite, args.workers)


if __name__ == "__main__":
    main()
