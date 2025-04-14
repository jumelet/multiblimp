import sys
sys.path.append("../src")

from mblimp.pipeline import pipeline, swap_number
from mblimp.filters import NsubjFilter
from mblimp.languages import get_ud_langs


if __name__ == "__main__":
    lang_head_features = {
        "English": {"Tense": "Pres", "Person": "3"},
    }
    um_strategies = {
        "Latvian": [
            {"Mood": None},
        ],
        "English": [{"Mood": None, "Person": None, "Tense": None}],
        "Galician": [
            {"Person": "3", "Mood": "Ind"},
        ],
    }
    resource_dir = "../resources"
    lang_candidates = get_ud_langs(resource_dir)[:10]
    lang_candidates = ["German"]

    scores, all_minimal_pairs, all_diagnostics = pipeline(
        lang_candidates,
        NsubjFilter,
        swap_number,
        upos="V",
        lang_head_features=lang_head_features,
        take_features_from="head",
        um_strategies=um_strategies,
        max_corpus_size=500,
        use_ud_inflections=False,
        test_files_only=False,
        save_dir="minimal_pairs/sv#a_um",
        resource_dir=resource_dir,
        load_from_pickle=True,
        shuffle_treebank=False,
        combine_um_ud=True,
    )
    print(all_diagnostics)
