from glob import glob
import argparse
import sys
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="LM to evaluate")
    parser.add_argument(
        "--data_dir", help="Minimal pair directory", default="final_pairs"
    )
    parser.add_argument("--src_dir", help="Source directory", default="./src")
    parser.add_argument("--results_dir", help="Dir to write results to", default=None)
    parser.add_argument(
        "--cache_dir", help="(optional) HF cache dir", default="/scratch-shared/jumelet"
    )
    parser.add_argument(
        "--hf_token",
        help="Huggingface token (file or token itself)",
        default="hf_token.txt",
    )
    args = parser.parse_args()

    sys.path.append(args.src_dir)

    from lm_eval.load_model import load_hf_model
    from lm_eval.score import score_tse

    if os.path.exists(args.hf_token):
        with open(args.hf_token) as f:
            hf_token = f.read().strip()
    else:
        hf_token = args.hf_token

    lm = load_hf_model(
        args.model_name, no_cache=False, token=hf_token, cache_dir=args.cache_dir
    )

    pair_files = glob(os.path.join(args.data_dir, "*/*/*.tsv"))
    for fn in sorted(pair_files):
        _, phenomenon, lang, condition = fn.split("/")

        df = score_tse(lm, fn=fn)

        print(phenomenon, lang, condition, f"{(df.sen_nll < df.wrong_nll).mean():.3f}")

        results_dir = (
            os.path.join("model_results", args.model_name)
            if args.results_dir is None
            else args.results_dir
        )
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)

        score_fn = os.path.join(
            results_dir, phenomenon + "_" + lang + "_" + condition + ".tsv"
        )
        df.to_csv(score_fn, sep="\t")
