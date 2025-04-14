import sys

sys.path.append("../../src")

from mblimp.pipeline import Pipeline
from mblimp.swap_features import *
from mblimp.filters import NsubjFilter
from mblimp.argparse import fetch_lang_candidates


if __name__ == "__main__":
    resource_dir = "../../resources"
    lang_candidates = fetch_lang_candidates(resource_dir)

    pipeline = Pipeline(
        NsubjFilter,
        swap_gender_any,
        unimorph_inflect_args={
            "filter": {
                "upos": ["V"],
            },
            "combine_um_ud": True,
            "remove_multiword_forms": True,
        },
        unimorph_context_args={
            "filter": {
                "upos": ["N", "PRO", "PRON"],
            },
            "combine_um_ud": True,
            "remove_multiword_forms": True,
        },
        treebank_args={
            "treebank_size": None,
            "test_files_only": False,
            "shuffle_treebank": True,
        },
        filter_args={
            "lang_head_features": {},
            "lang_child_features": {},
            "noun_upos": ["NOUN", "PRON"],
            "ufeat": "Gender",
        },
        take_features_from="head",
        um_strategies={},
        max_num_of_pairs=5000000,
        save_dir="../../minimal_pairs/svGa",
        resource_dir=resource_dir,
        load_from_pickle=True,
        balance_features=True,
        store_diagnostics=False,
    )
    scores, all_minimal_pairs, all_diagnostics = pipeline(lang_candidates)
