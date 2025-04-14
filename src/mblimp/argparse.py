import argparse
from typing import List

from .languages import get_ud_langs


def fetch_lang_candidates(resource_dir: str) -> List[str]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--langs', nargs='*', help='Languages to process', default=[])
    args = parser.parse_args()

    if len(args.langs) > 0:
        lang_candidates = args.langs
    else:
        lang_candidates = get_ud_langs(resource_dir)

    return lang_candidates
