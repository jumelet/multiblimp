"""Score the word-order pairs dataset (see word_order/huggingface.py) with an LM.

Uses the batched loglikelihood of lm-eval's HF model directly, rather than
lm-eval tasks: with one task per (lang, deprel) split, building the tasks
takes far longer than the scoring itself.

Each sentence is scored by its log-probability conditioned only on BOS/EOT, and
a pair counts as correct if the original sentence is more likely than the
swapped one. Scores are stored per pair as (lang, deprel, sen_logprob,
swapped_sen_logprob).

HF_HOME must be set before importing this module, as it imports transformers.
"""

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
    local_dir = snapshot_download(
        dataset,
        repo_type="dataset",
        allow_patterns=patterns,
        max_workers=32,
        token=os.environ.get("HF_TOKEN"),
    )

    dfs = []
    for path in tqdm(sorted(Path(local_dir).glob("*/*.parquet"))):
        df = pd.read_parquet(path, columns=["sen_str", "swapped_sen_str", "lang", "deprel"])
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


def eval_lm(
    model: str,
    dataset: str,
    output_dir: str,
    langs: list[str] | None = None,
    batch_size: int | str = "auto:8",
    max_batch_size: int = 1024,
    dtype: str = "auto",
) -> pd.DataFrame:
    """Score all pairs of the dataset with the model, and save the scores to
    {output_dir}/{model with / replaced by __}.parquet."""
    pairs = load_pairs(dataset, langs)
    print(f"Loaded {len(pairs)} pairs", flush=True)

    lm = HFLM(
        pretrained=model,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        dtype=dtype,
        softmax_dtype="float32",
    )

    sen2logprob = score_sentences(lm, pairs.sen_str.tolist() + pairs.swapped_sen_str.tolist())
    pairs["sen_logprob"] = pairs.sen_str.map(sen2logprob)
    pairs["swapped_sen_logprob"] = pairs.swapped_sen_str.map(sen2logprob)

    accuracy = (pairs.sen_logprob > pairs.swapped_sen_logprob).mean()
    print(f"Overall accuracy: {accuracy:.3f}", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, model.replace("/", "__") + ".parquet")
    scores = pairs.drop(columns=["sen_str", "swapped_sen_str"])
    scores.to_parquet(output_path, index=False)
    print(f"Saved scores to {output_path}", flush=True)

    return scores
