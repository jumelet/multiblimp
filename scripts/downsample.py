import pickle
import sys
sys.path.append("../src")

from mblimp.downsample import downsample_pairs
from mblimp.languages import get_ud_langs, lang2langcode


if __name__ == "__main__":
    resource_dir = "../resources"

    lang_candidates = get_ud_langs(resource_dir)
    phenomena = [
        "sv#a",
        "sp#a",
        "svGa",
        "spGa",
        "svPa",
        "spPa",
    ]

    for phenomenon in phenomena:
        n_items = 100

        with open(f'lang_stats/lang_{phenomenon}.pickle', 'rb') as f:
            all_lang_probs = pickle.load(f)

        for lang in lang_candidates:
            print(lang)
            langcode = lang2langcode(lang)

            downsample_pairs(
                all_lang_probs[lang],
                langcode,
                phenomenon,
                n_items,
                f"../final_pairs2/{phenomenon}",
                split_aux=("sv" in phenomenon),
            )
