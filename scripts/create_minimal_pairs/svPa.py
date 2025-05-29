import sys

sys.path.append("../../src")

from multiblimp.pipeline import Pipeline
from multiblimp.swap_features import *
from multiblimp.filters import NsubjFilter
from multiblimp.argparse import fetch_lang_candidates


if __name__ == "__main__":
    resource_dir = "../../resources"
    lang_candidates = fetch_lang_candidates(resource_dir)

    pipeline = Pipeline(
        NsubjFilter,
        swap_any_person,
        unimorph_inflect_args={
            "filter_entries": {
                "upos": ["V"],
            },
            "combine_um_ud": True,
            "remove_multiword_forms": True,
        },
        unimorph_context_args={
            "filter_entries": {
                "upos": ["N", "PRO", "PRON"],
            },
            "combine_um_ud": True,
            "remove_multiword_forms": True,
        },
        treebank_args={
            "treebank_size": None,
            "test_files_only": False,
            "shuffle_treebank": False,
        },
        filter_args={
            "lang_head_features": {},
            "lang_child_features": {},
            "noun_upos": ["NOUN", "PRON"],
            "ufeat": "Person",
        },
        take_features_from="head",
        um_strategies={},
        max_num_of_pairs=5000000,
        save_dir="../../minimal_pairs/svPa",
        resource_dir=resource_dir,
        load_from_pickle=True,
        balance_features=True,
        store_diagnostics=False,
    )
    scores, all_minimal_pairs, all_diagnostics = pipeline(lang_candidates)
