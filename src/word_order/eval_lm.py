"""Score the word-order pairs dataset (see huggingface.py) with an LM.

Uses the batched loglikelihood of lm-eval's HF model directly, rather than
lm-eval tasks: with one task per (lang, deprel) split, building the tasks
takes far longer than the scoring itself.

Scores are stored per pair as (lang, deprel, doc_id, sen_logprob,
swapped_sen_logprob), where doc_id is the row index within the split. This
matches the row order of the {deprel}/{lang}.parquet pair files, so the scores
can be merged back onto the full pair data for analysis.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm


def load_pairs(dataset: str, langs: list[str] | None = None) -> pd.DataFrame:
    """Download the {lang}/{deprel}-00000-of-00001.parquet files of the dataset
    in a single snapshot and concatenate them."""
    patterns = [f"{lang}/*.parquet" for lang in langs] if langs else ["*/*.parquet"]
    local_dir = snapshot_download(dataset, repo_type="dataset", allow_patterns=patterns)

    dfs = []
    for path in tqdm(sorted(Path(local_dir).glob("*/*.parquet"))):
        df = pd.read_parquet(
            path, columns=["sen_str", "swapped_sen_str", "lang", "deprel"]
        )
        df["doc_id"] = range(len(df))
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def score_sentences(lm: HFLM, sentences: list[str]) -> dict[str, float]:
    """Log-probability of each unique sentence, conditioned only on BOS/EOT."""
    unique_sentences = list(dict.fromkeys(sentences))
    requests = [
        Instance(request_type="loglikelihood", doc={}, arguments=("", sen), idx=0)
        for sen in unique_sentences
    ]
    logprobs = [logprob for logprob, _ in lm.loglikelihood(requests)]

    return dict(zip(unique_sentences, logprobs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", default="model_results/word_order")
    parser.add_argument("--langs", nargs="+", default=None)
    parser.add_argument(
        "--batch_size",
        default="auto:8",
        help="'auto:8' re-evaluates the optimal batch size 8 times during eval",
    )
    parser.add_argument("--max_batch_size", type=int, default=1024)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--hf_token", help="File containing an HF token", default="hf_token.txt"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)

    if os.path.exists(args.hf_token):
        with open(args.hf_token) as f:
            # datasets, transformers and huggingface_hub all read HF_TOKEN from the env.
            os.environ["HF_TOKEN"] = f.read().strip()

    pairs = load_pairs(args.dataset, args.langs)
    print(f"Loaded {len(pairs)} pairs")

    lm = HFLM(
        pretrained=args.model,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        dtype=args.dtype,
    )

    sen2logprob = score_sentences(
        lm, pairs.sen_str.tolist() + pairs.swapped_sen_str.tolist()
    )
    pairs["sen_logprob"] = pairs.sen_str.map(sen2logprob)
    pairs["swapped_sen_logprob"] = pairs.swapped_sen_str.map(sen2logprob)

    accuracy = (pairs.sen_logprob > pairs.swapped_sen_logprob).mean()
    print(f"Overall accuracy: {accuracy:.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, args.model.replace("/", "__") + ".parquet"
    )
    pairs.drop(columns=["sen_str", "swapped_sen_str"]).to_parquet(
        output_path, index=False
    )
    print(f"Saved scores to {output_path}")
