"""Score word-order minimal pairs with a causal LM using minicons.

Collects all sentences across every (lang, deprel) combination first,
then scores them in a single batched pass for GPU efficiency.

Usage:
    python -m src.lm_eval.lm_eval_harness \\
        --model jumelet/my-model \\
        --dataset jumelet/word-order-pairs \\
        --output results.csv \\
        [--langs eng nld deu] \\
        [--deprels nsubj amod]
"""

import argparse
from typing import Optional

import pandas as pd
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from minicons import scorer
from tqdm import tqdm

KEEP_COLUMNS = ["sen_str", "original_order", "swap_order", "swapped_sen_str", "leaf_rule"]


def get_task_specs(
    dataset_name: str,
    langs: Optional[list[str]] = None,
    deprels: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    """Return (lang, deprel) pairs available in the dataset, filtered by args."""
    available_langs = get_dataset_config_names(dataset_name)
    if langs:
        available_langs = [l for l in available_langs if l in langs]

    specs = []
    for lang in tqdm(available_langs, desc="Fetching splits"):
        available_splits = get_dataset_split_names(dataset_name, lang)
        for split in available_splits:
            if deprels is None or split in deprels:
                specs.append((lang, split))

    return specs


def run_eval(
    model_name: str,
    dataset_name: str,
    output_path: str,
    langs: Optional[list[str]] = None,
    deprels: Optional[list[str]] = None,
    batch_size: int = 32,
    device: str = "cuda",
) -> pd.DataFrame:
    print(f"Enumerating tasks for dataset '{dataset_name}' ...")
    specs = get_task_specs(dataset_name, langs, deprels)
    print(f"Found {len(specs)} (lang, deprel) combinations.")

    print(f"Loading model '{model_name}' ...")
    lm = scorer.IncrementalLMScorer(model_name, device=device)

    all_dfs, all_sens, all_swapped = [], [], []
    for lang, deprel in tqdm(specs, desc="Loading data"):
        ds = load_dataset(dataset_name, lang, split=deprel)
        df = ds.to_pandas()[list(KEEP_COLUMNS)]
        df["lang"] = lang
        df["deprel"] = deprel
        all_dfs.append(df)
        all_sens.extend(df["sen_str"].tolist())
        all_swapped.extend(df["swapped_sen_str"].tolist())

    def score_batched(sentences):
        tok_lens = [len(ids) for ids in lm.tokenizer(sentences, add_special_tokens=True).input_ids]
        order = sorted(range(len(sentences)), key=lambda i: tok_lens[i])
        restore = [0] * len(sentences)
        for rank, orig in enumerate(order):
            restore[orig] = rank
        sorted_sents = [sentences[i] for i in order]

        raw = []
        for i in tqdm(range(0, len(sorted_sents), batch_size), desc="Scoring"):
            raw.extend(lm.token_score(sorted_sents[i : i + batch_size]))
        return [raw[restore[i]] for i in range(len(sentences))]

    print("Scoring sentences ...")
    raw_sen = score_batched(all_sens)
    raw_swapped = score_batched(all_swapped)

    sen_probs = [[s for _, s in sent] for sent in raw_sen]
    swapped_probs = [[s for _, s in sent] for sent in raw_swapped]

    result_df = pd.concat(all_dfs, ignore_index=True)
    result_df["sen_probs"] = sen_probs
    result_df["swapped_sen_probs"] = swapped_probs
    result_df["sen_nll"] = [-sum(p) for p in sen_probs]
    result_df["swapped_sen_nll"] = [-sum(p) for p in swapped_probs]

    result_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Score word-order minimal pairs with a causal LM")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--output", required=True, help="Path for the output CSV")
    parser.add_argument("--langs", nargs="+", default=None, help="ISO-639 language codes (default: all)")
    parser.add_argument("--deprels", nargs="+", default=None, help="Dependency relations (default: all)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_eval(
        model_name=args.model,
        dataset_name=args.dataset,
        output_path=args.output,
        langs=args.langs,
        deprels=args.deprels,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
