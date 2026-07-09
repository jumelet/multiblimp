"""Upload word-order pair CSVs to a HuggingFace dataset.

Directory layout expected:
    {pair_dir}/{deprel}/{lang}.csv

HuggingFace layout produced:
    subset  = ISO-639 language code
    split   = deprel name
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, disable_progress_bar
from huggingface_hub import HfApi
from tqdm import tqdm

from multiblimp.languages import lang2langcode

KEEP_COLUMNS = [
    "sen_str",
    "original_order",
    "swap_order",
    "swapped_sen_str",
    "leaf_rule",
]


def load_pairs(pair_dir: str) -> dict[str, dict[str, pd.DataFrame]]:
    """Return {lang: {deprel: df}} from the pair directory tree."""
    root = Path(pair_dir)
    data: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)

    csv_paths = sorted(p for d in root.iterdir() if d.is_dir() for p in d.glob("*.csv"))

    for csv_path in tqdm(csv_paths, desc="Loading CSVs"):
        deprel = csv_path.parent.name
        lang = lang2langcode(csv_path.stem)
        data[lang][deprel] = pd.read_csv(
            csv_path, low_memory=False, usecols=KEEP_COLUMNS
        )

    return data


def upload(pair_dir: str, dataset_name: str) -> None:
    disable_progress_bar()

    print(f"Loading pairs from {pair_dir} ...")
    lang_data = load_pairs(pair_dir)

    api = HfApi()
    api.create_repo(
        repo_id=dataset_name, repo_type="dataset", private=True, exist_ok=True
    )

    for lang, deprel_map in tqdm(sorted(lang_data.items()), desc="Uploading subsets"):
        subset = DatasetDict(
            {
                deprel: Dataset.from_pandas(df, preserve_index=False)
                for deprel, df in deprel_map.items()
            }
        )
        subset.push_to_hub(dataset_name, config_name=lang)

    print(f"Done. Dataset available at https://huggingface.co/datasets/{dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload word-order pairs to HuggingFace"
    )
    parser.add_argument(
        "pair_dir", help="Root directory containing {deprel}/{lang}.csv files"
    )
    parser.add_argument(
        "dataset_name", help="HuggingFace dataset name, e.g. 'username/my-dataset'"
    )
    args = parser.parse_args()

    upload(args.pair_dir, args.dataset_name)


if __name__ == "__main__":
    main()
