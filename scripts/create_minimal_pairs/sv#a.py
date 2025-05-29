import sys

sys.path.append("../../src")

from multiblimp.pipeline import Pipeline
from multiblimp.swap_features import swap_number_subj_any
from multiblimp.filters import NsubjFilter
from multiblimp.argparse import fetch_lang_candidates


if __name__ == "__main__":
    resource_dir = "../../resources"
    lang_candidates = fetch_lang_candidates(resource_dir)

    lang_head_features = {
        "English": {"Tense": "Pres", "Person": "3"},
    }

    pipeline = Pipeline(
        NsubjFilter,
        swap_number_subj_any,
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
            "lang_head_features": lang_head_features,
            "lang_child_features": {},
            "noun_upos": ["NOUN", "PRON"],
            "ufeat": "Number",
        },
        take_features_from="head",
        max_num_of_pairs=10000000,
        save_dir="../../minimal_pairs/sv#a",
        resource_dir=resource_dir,
        load_from_pickle=True,
        balance_features=True,
        store_diagnostics=False,
    )
    scores, all_minimal_pairs, all_diagnostics = pipeline(lang_candidates)
