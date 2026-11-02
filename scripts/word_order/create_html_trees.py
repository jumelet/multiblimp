"""Render the fitted word order DTs as HTML pages.

Writes {data_dir}/html/{deprel}/{lang}.html for the DTs in dt_df/, and
html_lexical/ for those in dt_df_lexical/. Existing pages are skipped unless
--overwrite is given.
"""

import argparse
import os
import sys
from glob import glob

import joblib
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from word_order.prediction_target import get_all_deprel_targets
from word_order.process_treebank import read_df
from word_order.viz_tree import tree2html

PREDICTOR_VAR = "deprel_order"
DT_DIRS = {"dt_df": "html", "dt_df_lexical": "html_lexical"}


def create_html_trees(
    data_dir: str,
    langs: list[str] | None = None,
    deprels: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    selected = None if langs is None else {lang.replace(" ", "_") for lang in langs}
    for target in get_all_deprel_targets():
        deprel = "_".join(target.child_deprels)
        if deprels is not None and deprel not in deprels:
            continue
        for dt_dir, html_dir in DT_DIRS.items():
            for model_path in sorted(glob(os.path.join(data_dir, dt_dir, deprel, "*.joblib"))):
                lang = os.path.basename(model_path)[: -len(".joblib")]
                html_file = os.path.join(data_dir, html_dir, deprel, f"{lang}.html")
                if (selected is not None and lang not in selected) or (os.path.exists(html_file) and not overwrite):
                    continue

                full_df = read_df(lang, word_order_dir=os.path.join(data_dir, "treebank_features", deprel))
                tree2html(
                    joblib.load(model_path),
                    pd.read_parquet(model_path[: -len(".joblib")] + ".parquet"),
                    full_df[full_df[PREDICTOR_VAR].notnull()],
                    PREDICTOR_VAR,
                    html_file,
                    target=target,
                    max_rows=15,
                    meta={"Language": lang if dt_dir == "dt_df" else f"{lang} (lexical)"},
                    only_show_real_orders=True,
                    correlate_features=True,
                )
                print(f"[{lang} {deprel}] {html_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Directory with treebank_features/ and the DTs")
    parser.add_argument("--langs", nargs="+", default=None, help="Defaults to all languages with a DT")
    parser.add_argument("--deprels", nargs="+", default=None, help="Defaults to all targets")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    create_html_trees(args.data_dir, args.langs, args.deprels, args.overwrite)


if __name__ == "__main__":
    main()
