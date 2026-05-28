import pickle
import sys

sys.path.append("../src")

from multiblimp.pipeline import Pipeline
from multiblimp.filters import AdjNounFilter
from multiblimp.languages import get_ud_langs, lang2langcode
from multiblimp.measure_agreement import get_feature_combinations
from multiblimp.swap_features import *


if __name__ == "__main__":
    resource_dir = "../resources"

    lang_candidates = ["French"]  # get_ud_langs(resource_dir)

    all_results = {}

    pipeline = Pipeline(
        AdjNounFilter,
        swap_number_subj_any,
        unimorph_inflect_args={
            "filter_entries": {
                "upos": ["N"],
            },
            "combine_um_ud": True,
            "remove_multiword_forms": True,
        },
        unimorph_context_args={
            "filter_entries": {
                "upos": ["Adj", "ADJ"],
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

    for lang in lang_candidates:
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
            verbose=True,
            tqdm_progress=False,
            only_try_ud_if_no_um=True,
            discard_undefined=True,
            allow_undefined=False,
            allow_multiple=True,
            collocate_ud_features=True,
            only_collocate_ud_features=True,
        )
        all_results[lang] = results

    # with open("lang_svPa.pickle", "rb") as f:
    #     prev_results = pickle.load(f)
    #
    # prev_results.update(all_results)

    #with open("lang_svPa.pickle", "wb") as f:
    #    pickle.dump(all_results, f)
