"""Run (parts of) the word order minimal pair pipeline.

Steps, always run in this order:
  pickle_treebanks           UD treebanks -> resources/ud/ud_pickles/ (and _max30000/)
  extract_features           -> treebank_features/
  fit_validation_classifiers -> joint_classifiers/
  fit_dts                    -> dt_df/
  create_pairs               -> final_pairs/ (and dt_df_lexical/ where needed)
  create_html_trees          -> html/, html_lexical/
  upload_to_hf               final_pairs/ -> HuggingFace dataset (--dataset)

Each step skips outputs that already exist, unless --overwrite is given.

Example:
  python pipeline.py --data_dir /path/to/WORD_ORDER \\
      --steps extract_features fit_dts create_pairs --overwrite --workers 6
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "../pickle"))

from multiblimp.languages import get_ud_langs
from word_order.parallel import parse_workers

from create_final_pairs import create_pairs
from create_html_trees import create_html_trees
from fit_dts import fit_dts
from fit_validation_classifiers import fit_validation_classifiers
from pickle_treebanks import pickle_treebanks, subsample_pickles
from process_all_treebanks import extract_features

STEPS = [
    "pickle_treebanks",
    "extract_features",
    "fit_validation_classifiers",
    "fit_dts",
    "create_pairs",
    "create_html_trees",
    "upload_to_hf",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Root directory of all pipeline data, with resources/")
    parser.add_argument("--steps", nargs="+", required=True, choices=STEPS)
    parser.add_argument("--langs", nargs="+", default=None, help="Defaults to all languages in the UD folder")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=parse_workers,
        default=1,
        help='Maximum languages processed in parallel, or "auto" (one per CPU core); also limited by memory',
    )
    parser.add_argument("--max_total_items", type=int, default=500, help="Maximum pairs per (lang, deprel)")
    parser.add_argument("--min_total_items", type=int, default=10, help="Minimum pairs per (lang, deprel)")
    parser.add_argument("--validate_threshold", type=float, default=0.9)
    parser.add_argument("--dataset", default=None, help="HuggingFace dataset name, required for upload_to_hf")
    args = parser.parse_args()

    if "upload_to_hf" in args.steps and args.dataset is None:
        parser.error("upload_to_hf requires --dataset")

    resource_dir = os.path.join(args.data_dir, "resources")
    langs = args.langs or sorted(get_ud_langs(resource_dir))

    for step in STEPS:
        if step not in args.steps:
            continue
        print(f"=== {step} === {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        if step == "pickle_treebanks":
            pickle_treebanks(resource_dir, langs, args.overwrite)
            subsample_pickles(resource_dir, langs, overwrite=args.overwrite)
        elif step == "extract_features":
            extract_features(args.data_dir, langs, overwrite=args.overwrite, workers=args.workers)
        elif step == "fit_validation_classifiers":
            fit_validation_classifiers(args.data_dir, langs, args.overwrite, args.workers)
        elif step == "fit_dts":
            fit_dts(args.data_dir, langs, overwrite=args.overwrite, workers=args.workers)
        elif step == "create_pairs":
            create_pairs(
                args.data_dir,
                langs,
                max_total_items=args.max_total_items,
                min_total_items=args.min_total_items,
                validate_threshold=args.validate_threshold,
                overwrite=args.overwrite,
                workers=args.workers,
            )
        elif step == "create_html_trees":
            create_html_trees(args.data_dir, langs, overwrite=args.overwrite)
        elif step == "upload_to_hf":
            from word_order.huggingface import upload

            upload(os.path.join(args.data_dir, "final_pairs"), args.dataset, delete_prior_files=args.overwrite)


if __name__ == "__main__":
    main()
