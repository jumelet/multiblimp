"""Score the word order minimal pairs of a HuggingFace dataset with a causal LM.

See src/eval_lm/word_order.py. Scores are saved to
{output_dir}/{model with / replaced by __}.parquet.

Example:
  python eval_model.py --model swiss-ai/Apertus-8B-2509 --langs nld deu
"""

import argparse
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="e.g. swiss-ai/Apertus-8B-2509 or google/gemma-3-27b-pt")
    parser.add_argument("--dataset", default="jumelet/multiblimp-v2")
    parser.add_argument("--langs", nargs="+", default=None, help="e.g. nld deu; defaults to all languages")
    parser.add_argument(
        "--batch_size",
        default="auto:8",
        help='Batch size, or "auto:N" to re-evaluate the optimal batch size N times during eval',
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=256,
        help="Upper bound of the auto-detected batch size; with 1024, the float32 logits of a batch "
        "of short sentences took 18.55 GiB for salamandra-2b (256k vocab) and went OOM",
    )
    parser.add_argument("--hf_token", default="hf_token.txt", help="File containing the token, or the token itself")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output_dir", default="model_results/word_order")
    parser.add_argument("--hf_home", default="/scratch-shared/jumelet/hf")
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Only download the dataset into the HF cache (e.g. on the login node before submitting jobs), "
        "with the same HF_HOME and token as a job",
    )
    args = parser.parse_args()
    if args.model is None and not args.download_only:
        parser.error("--model is required, unless --download_only is given")

    logging.basicConfig(level=logging.INFO, force=True)

    # HF_HOME must be set before transformers is imported
    os.environ["HF_HOME"] = args.hf_home

    if os.path.exists(args.hf_token):
        with open(args.hf_token) as f:
            os.environ["HF_TOKEN"] = f.read().strip()
    elif args.hf_token != parser.get_default("hf_token"):
        os.environ["HF_TOKEN"] = args.hf_token

    if args.download_only:
        from eval_lm.word_order import download_pairs

        local_dir = download_pairs(args.dataset, args.langs)
        print(f"Downloaded {args.dataset} (HF_HOME={os.environ['HF_HOME']}) to {local_dir}", flush=True)
        return

    print("Importing lm-eval", flush=True)
    from eval_lm.word_order import eval_lm
    print("Imports loaded", flush=True)

    eval_lm(
        model=args.model,
        dataset=args.dataset,
        output_dir=args.output_dir,
        langs=args.langs,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
