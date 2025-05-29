import pickle
import sys

sys.path.append("../../src")

from multiblimp.pipeline import Pipeline
from multiblimp.filters import NsubjFilter
from multiblimp.languages import get_ud_langs, lang2langcode
from multiblimp.measure_agreement import get_feature_combinations
from multiblimp.swap_features import *


if __name__ == "__main__":
    resource_dir = "../resources"

    lang_candidates = get_ud_langs(resource_dir)
    lang_head_features = {
        "English": {"Tense": "Pres", "Person": "3"},
    }

    all_results = {}

    pipeline = Pipeline(
        NsubjFilter,
        swap_number_subj_any,
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
            "lang_head_features": lang_head_features,
            "transfer_head_child_features": False,
            "lang_child_features": {},
            "noun_upos": ["NOUN", "PRON"],
        },
        take_features_from="head",
        max_num_of_pairs=0,
        resource_dir=resource_dir,
        load_from_pickle=True,
        balance_features=True,
        store_diagnostics=False,
    )

    for lang in ["French"]:  # lang_candidates:
        print(lang)
        langcode = lang2langcode(lang)

        inflector = pipeline.load_inflector(
            lang, langcode, pipeline.unimorph_inflect_args
        )[0]
        context_inflector = pipeline.load_inflector(
            lang, langcode, pipeline.unimorph_context_args
        )[0]
        corpus = pipeline.load_corpus(lang)

        results = get_feature_combinations(
            corpus,
            inflector,
            context_inflector,
            head_ud_features={"Mood": "Ind"},
            child_ud_features={"Case": "Nom"},
            verbose=True,
            tqdm_progress=False,
            only_try_ud_if_no_um=False,
            discard_undefined=True,
            allow_undefined=False,
        )
        all_results[lang] = results

    # with open("lang_sv#a.pickle", "rb") as f:
    #     prev_results = pickle.load(f)
    #
    # prev_results.update(all_results)

    # with open('lang_sv#a.pickle', 'wb') as f:
    #     pickle.dump(all_results, f)
