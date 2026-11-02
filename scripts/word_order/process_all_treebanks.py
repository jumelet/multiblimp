"""Extract the word order features of all treebanks, for every deprel target.

Writes {data_dir}/treebank_features/{deprel}/{lang}.parquet; existing files are
skipped unless --overwrite is given.
"""

import argparse
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from multiblimp.languages import get_ud_langs
from word_order.parallel import map_languages, parse_workers
from word_order.prediction_target import get_all_deprel_targets
from word_order.process_treebank import create_word_order_df, load_treebank
from word_order.utils import SAMPLED_PICKLE_DIR


def extract_language_features(lang: str, data_dir: str, deprels: list[str] | None = None, overwrite: bool = False):
    random.seed(42)
    resource_dir = os.path.join(data_dir, "resources")
    todo = []
    for target in get_all_deprel_targets():
        deprel = "_".join(target.child_deprels)
        if deprels is not None and deprel not in deprels:
            continue
        word_order_dir = os.path.join(data_dir, "treebank_features", deprel)
        path = os.path.join(word_order_dir, f"{lang.replace(' ', '_')}.parquet")
        if overwrite or not os.path.exists(path):
            todo.append((target, word_order_dir, path))
    if not todo:
        print(f"[{lang}] done", flush=True)
        return

    treebank = load_treebank(lang, os.path.join(data_dir, SAMPLED_PICKLE_DIR))
    for target, word_order_dir, path in todo:
        if os.path.exists(path):
            os.remove(path)  # a target without instances writes no new file
        df = create_word_order_df(
            lang=lang,
            treebank=treebank,
            target=target,
            resource_dir=resource_dir,
            save_to=word_order_dir,
            drop_singleton_columns=True,
            lexicalize=True,
        )
        print(f"[{lang} {'_'.join(target.child_deprels)}] {len(df)} instances", flush=True)
    print(f"[{lang}] done", flush=True)


def extract_features(
    data_dir: str,
    langs: list[str] | None = None,
    deprels: list[str] | None = None,
    overwrite: bool = False,
    workers: int = 1,
) -> None:
    langs = langs or sorted(get_ud_langs(os.path.join(data_dir, "resources")))
    map_languages(
        extract_language_features,
        langs,
        workers,
        step="extract_features",
        data_dir=data_dir,
        deprels=deprels,
        overwrite=overwrite,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Directory with resources/ (UD treebanks and pickles)")
    parser.add_argument("--langs", nargs="+", default=None, help="Defaults to all UD languages")
    parser.add_argument("--deprels", nargs="+", default=None, help="Defaults to all targets")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=parse_workers,
        default=1,
        help='Maximum languages processed in parallel, or "auto" (one per CPU core); also limited by memory',
    )
    args = parser.parse_args()
    extract_features(args.data_dir, args.langs, args.deprels, args.overwrite, args.workers)


if __name__ == "__main__":
    main()
