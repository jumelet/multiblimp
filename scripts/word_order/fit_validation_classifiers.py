"""Fit the joint attachment classifier per language, used to validate swaps.

Writes {data_dir}/joint_classifiers/{lang}.joblib; existing files are skipped
unless --overwrite is given.
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from multiblimp.languages import get_ud_langs
from word_order.parallel import map_languages, parse_workers
from word_order.process_treebank import load_treebank
from word_order.utils import SAMPLED_PICKLE_DIR
from word_order.validate_swap import fit_joint_classifier


def fit_language_classifier(
    lang: str,
    data_dir: str,
    overwrite: bool = False,
    n_samples: int = 10_000,
    max_depth: int = 25,
    min_impurity_decrease: float = 0.001,
    min_samples_leaf: int = 25,
):
    lang = lang.replace(" ", "_")
    save_to = os.path.join(data_dir, "joint_classifiers", lang)
    if os.path.exists(save_to + ".joblib") and not overwrite:
        return

    treebank = load_treebank(lang, os.path.join(data_dir, SAMPLED_PICKLE_DIR))
    fit_joint_classifier(
        treebank,
        n_samples,
        "gradient_boosting",
        max_depth=max_depth,
        min_impurity_decrease=min_impurity_decrease,
        min_samples_leaf=min_samples_leaf,
        sample_from_same_head=False,
        save_to=save_to,
    )
    print(f"[{lang}] fitted", flush=True)


def fit_validation_classifiers(
    data_dir: str,
    langs: list[str] | None = None,
    overwrite: bool = False,
    workers: int = 1,
) -> None:
    langs = langs or sorted(get_ud_langs(os.path.join(data_dir, "resources")))
    map_languages(
        fit_language_classifier,
        langs,
        workers,
        step="fit_validation_classifiers",
        data_dir=data_dir,
        overwrite=overwrite,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Directory with resources/ (UD pickles)")
    parser.add_argument("--langs", nargs="+", default=None, help="Defaults to all UD languages")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=parse_workers,
        default=1,
        help='Maximum languages processed in parallel, or "auto" (one per CPU core); also limited by memory',
    )
    args = parser.parse_args()
    fit_validation_classifiers(args.data_dir, args.langs, args.overwrite, args.workers)


if __name__ == "__main__":
    main()
