"""Upload word-order pair parquet files to a HuggingFace dataset.

Directory layout expected:
    {pair_dir}/{deprel}/{lang}.parquet

HuggingFace layout produced:
    subset  = ISO-639 language code
    split   = deprel name
"""

import argparse
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset, disable_progress_bar
from huggingface_hub import HfApi
from tqdm import tqdm

from multiblimp.languages import lang2langcode

KEEP_COLUMNS = [
    "sen_str",
    "original_order",
    "swap_order",
    "swapped_sen_str",
    "leaf_rule",
    "max_swapped_head_prob",
    "discontinuous",
    "order_strength",
]


def load_pairs(pair_dir: str) -> dict[str, dict[str, pd.DataFrame]]:
    """Return {lang: {deprel: df}} from the pair directory tree."""
    root = Path(pair_dir)
    data: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)

    parquet_paths = sorted(
        p for d in root.iterdir() if d.is_dir() for p in d.glob("*.parquet")
    )

    for parquet_path in tqdm(parquet_paths, desc="Loading parquet files"):
        deprel = parquet_path.parent.name
        lang = lang2langcode(parquet_path.stem)
        print(parquet_path)
        df = pd.read_parquet(parquet_path, columns=KEEP_COLUMNS)
        # Stored per row so the splits of a config can be merged, e.g. with
        # load_dataset(repo, lang, split="all"), without losing where rows came from.
        df["lang"] = lang
        df["deprel"] = deprel
        data[lang][deprel] = df

    return data


def upload(pair_dir: str, dataset_name: str, delete_prior_files: bool = False) -> None:
    """
    If delete_prior_files is True, all parquet files already in the repo are
    removed in the same commit, so the repo exactly mirrors pair_dir.
    """
    disable_progress_bar()

    print(f"Loading pairs from {pair_dir} ...")
    lang_data = load_pairs(pair_dir)

    api = HfApi()
    # create_repo resolves a bare name to "{user}/{name}"; upload_folder doesn't.
    dataset_name = api.create_repo(
        repo_id=dataset_name, repo_type="dataset", private=True, exist_ok=True
    ).repo_id

    # Write everything locally and upload in a single commit: pushing each subset
    # separately with push_to_hub hits the HF limit of 128 commits per hour.
    # File layout matches push_to_hub: {lang}/{deprel}-00000-of-00001.parquet
    with tempfile.TemporaryDirectory() as tmp_dir:
        configs = []
        for lang, deprel_map in tqdm(
            sorted(lang_data.items()), desc="Writing subsets"
        ):
            os.makedirs(os.path.join(tmp_dir, lang), exist_ok=True)
            data_files = []
            for deprel, df in sorted(deprel_map.items()):
                path_in_repo = f"{lang}/{deprel}-00000-of-00001.parquet"
                Dataset.from_pandas(df, preserve_index=False).to_parquet(
                    os.path.join(tmp_dir, path_in_repo)
                )
                data_files.append({"split": deprel, "path": path_in_repo})
            configs.append({"config_name": lang, "data_files": data_files})

        # The YAML header of the dataset card tells HF which files form which
        # subset/split.
        with open(os.path.join(tmp_dir, "README.md"), "w") as f:
            f.write("---\n")
            yaml.safe_dump({"configs": configs}, f, sort_keys=False)
            f.write("---\n")

        print("Uploading ...")
        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=dataset_name,
            repo_type="dataset",
            commit_message="Upload dataset",
            delete_patterns="*.parquet" if delete_prior_files else None,
        )

    print(f"Done. Dataset available at https://huggingface.co/datasets/{dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload word-order pairs to HuggingFace"
    )
    parser.add_argument(
        "pair_dir", help="Root directory containing {deprel}/{lang}.parquet files"
    )
    parser.add_argument(
        "dataset_name", help="HuggingFace dataset name, e.g. 'username/my-dataset'"
    )
    parser.add_argument(
        "--delete-prior-files",
        action="store_true",
        help="Remove all parquet files already in the repo before uploading",
    )
    args = parser.parse_args()

    upload(args.pair_dir, args.dataset_name, args.delete_prior_files)


if __name__ == "__main__":
    main()
