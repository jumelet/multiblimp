import os
import random
from collections import Counter
from glob import glob
from itertools import permutations
from math import ceil
from typing import *

import pandas as pd
from tqdm import tqdm

from .filters import UDFilter
from .filters.utils import set_minimal_pairs
from .languages import (
    lang2langcode,
    remove_diacritics_langs,
    remove_multiples_langs,
    use_udlex_langs,
    gblang2udlang,
    lang2unimorph_lang,
)
from .treebank import Treebank
from .udlex import UDLexInflector
from .unimorph import UnimorphInflector, allval2um, UNDEFINED


class Pipeline:
    def __init__(
        self,
        filter_class: Type[UDFilter],
        inflection_map: Tuple[str, Dict[str, str]],
        save_dir=None,
        resource_dir=None,
        load_from_pickle=False,
        unimorph_inflect_args={},
        unimorph_context_args={},
        treebank_args={},
        filter_args={},
        take_features_from="child",
        um_strategies={},
        max_num_of_pairs=None,
        balance_features=False,
        store_diagnostics=True,
    ):
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        self.filter_class = filter_class
        self.inflection_map = inflection_map
        self.save_dir = save_dir
        self.resource_dir = resource_dir
        self.unimorph_inflect_args = unimorph_inflect_args
        self.unimorph_context_args = unimorph_context_args
        self.treebank_args = treebank_args
        self.filter_args = filter_args
        self.take_features_from = take_features_from
        self.um_strategies = um_strategies
        self.load_from_pickle = load_from_pickle
        self.max_num_of_pairs = max_num_of_pairs
        self.balance_features = balance_features
        self.store_diagnostics = store_diagnostics

    def __call__(self, lang_candidates: List[Tuple[str, str]]):
        scores = []
        all_minimal_pairs = {}
        all_diagnostics = {}

        try:
            for lang in lang_candidates:
                print(lang)
                lang_score, lang_minimal_pairs, diagnostics = self.process_language(
                    lang
                )

                scores.append(lang_score)
                all_minimal_pairs[lang] = lang_minimal_pairs
                all_diagnostics[lang] = diagnostics
        except KeyboardInterrupt:
            return scores, all_minimal_pairs, all_diagnostics

        with open(os.path.join(self.save_dir, "log.txt"), 'a') as f:
            for score in scores:
                f.write("\t".join(map(str, score)) + '\n')

        return scores, all_minimal_pairs, all_diagnostics

    def process_language(self, lang: str):
        corpus_size = 0
        items_seen = 0
        diagnostics = {
            "correct_swaps": [],
            "same_forms": [],
            "same_features": [],
            "undefined_features": [],
            "no_inflections": [],
            "no_candidates": [],
            "multi_now_valid": [],
            "ambiguous_subjects": [],
            "extra_pairs": [],
        }
        feature_distribution = Counter()

        umlang = lang2unimorph_lang.get(lang, lang)
        langcode = lang2langcode(umlang)

        inflector, skip_lang, num_lemma, num_form = self.load_inflector(lang, langcode, self.unimorph_inflect_args)

        ufeat, inflection_map = inflector.inflection_map
        if inflector.ud_inflector is not None:
            ud_ufeat, _ = inflector.ud_inflector.inflection_map
            ufeat = ud_ufeat or ufeat

        if inflection_map is None:
            unique_features = inflector.unique_values(ufeat)
            feature_combinations = permutations(unique_features, 2)

            if len(unique_features) < 2:
                skip_lang = True
        else:
            feature_combinations = inflection_map.items()

        feature_keys = []
        for feat1, feat2 in feature_combinations:
            key = f"{feat1} -> {feat2}"
            feature_keys.append(key)
            feature_distribution[key] = 0

        # Start inflection procedure
        if not skip_lang:
            corpus = self.load_corpus(lang)
            corpus_size = len(corpus)
            context_inflector = self.load_inflector(lang, langcode, self.unimorph_context_args)[0]

            strategies = self.um_strategies.get(lang, [])

            for _, item in tqdm(corpus):
                form_features = item[f"{self.take_features_from}_features"]
                form = item[self.take_features_from]

                if self.balance_features:
                    ud_ufeat = form_features.get(ufeat)
                    if ud_ufeat is not None:
                        um_ufeat = allval2um(ud_ufeat)
                        if feature_distribution[um_ufeat] >= self.max_num_of_pairs:
                            continue

                swap_forms, feature_vals = inflector.inflect(
                    form,
                    form_features,
                    strategies=strategies,
                )

                item[f"swap_{self.take_features_from}"] = swap_forms
                item["feature_vals"] = feature_vals

                if swap_forms is not None:
                    if len(swap_forms) > 0:
                        for swap_form in swap_forms:
                            items_seen += 1
                            item_type, item = self.process_item(
                                dict(item),
                                form,
                                swap_form,
                                form_features,
                                ufeat,
                                feature_vals,
                                feature_distribution,
                                inflector,
                                context_inflector,
                            )
                            diagnostics[item_type].append(item)

                            if (item_type == "correct_swaps") and (len(swap_forms) > 0):
                                multi_item = dict(item)
                                multi_item['alternatives'] = swap_forms
                                diagnostics["multi_now_valid"].append(multi_item)
                    else:
                        items_seen += 1
                        diagnostics["no_inflections"].append(item)
                else:
                    items_seen += 1
                    diagnostics["no_candidates"].append(item)

                if self.max_num_of_pairs is not None:
                    do_break = True
                    for key in feature_keys:
                        if feature_distribution[key] < self.max_num_of_pairs:
                            do_break = False
                    if do_break:
                        break

        minimal_pair_df = pd.DataFrame(diagnostics["correct_swaps"])
        minimal_pair_df = set_minimal_pairs(minimal_pair_df, self.take_features_from)

        diagnostics = {
            item_type: pd.DataFrame(items)
            for item_type, items in diagnostics.items()
        }

        coverage = f"{len(diagnostics['correct_swaps'])/max(items_seen,1)*100:.1f}"
        feature_distribution = Counter(
            {key: value for key, value in feature_distribution.items() if value > 0}
        )

        lang_score = (
            lang,
            len(diagnostics['correct_swaps']),
            coverage,
            corpus_size,
            num_lemma,
            num_form,
            len(diagnostics['no_candidates']),
            f"{len(diagnostics['no_candidates'])/max(items_seen,1)*100:.1f}",
            len(diagnostics['no_inflections']),
            f"{len(diagnostics['no_inflections']) / max(items_seen, 1) * 100:.1f}",
            len(diagnostics['same_forms']),
            f"{len(diagnostics['same_forms']) / max(items_seen, 1) * 100:.1f}",
            len(diagnostics['same_features']),
            f"{len(diagnostics['same_features']) / max(items_seen, 1) * 100:.1f}",
            len(diagnostics['undefined_features']),
            f"{len(diagnostics['undefined_features']) / max(items_seen, 1) * 100:.1f}",
            len(diagnostics['ambiguous_subjects']),
            f"{len(diagnostics['ambiguous_subjects']) / max(items_seen, 1) * 100:.1f}",
            len(diagnostics['multi_now_valid']),
            f"{len(diagnostics['multi_now_valid']) / max(len(diagnostics['correct_swaps']), 1) * 100:.1f}",
            feature_distribution,
        )
        print(*lang_score, sep="\t")

        if self.save_dir is not None:
            self.save_lang(langcode, minimal_pair_df, diagnostics)

        if self.store_diagnostics:
            return lang_score, minimal_pair_df, diagnostics
        else:
            return lang_score, None, None

    def process_item(
        self,
        item,
        form,
        swap_form,
        form_features,
        ufeat,
        feature_vals,
        feature_distribution,
        inflector,
        context_inflector,
    ) -> Tuple[str, Dict[str, str]]:
        item[f"swap_{self.take_features_from}"] = swap_form

        if swap_form == form:
            return "same_forms", item
        else:
            swap_feature_vals = inflector.get_form_features(swap_form, form_features, ufeat)
            feature_key = '|'.join(sorted(feature_vals)) + ' -> ' + '|'.join(sorted(swap_feature_vals))

            add_item = False
            for feat1 in feature_vals:
                for feat2 in swap_feature_vals:
                    if feature_distribution[f"{feat1} -> {feat2}"] < self.max_num_of_pairs:
                        add_item = True
                    else:
                        add_item = None

            if (
                (len(feature_vals & swap_feature_vals) == 0)
                and (UNDEFINED not in swap_feature_vals)
            ):
                if (len(swap_feature_vals) > 0) and add_item:
                    # todo: take the child features from an argument
                    child_features = context_inflector.get_form_features(
                        item["child"],
                        {"Case": "Nom"},  # ergative?
                        ufeat,
                        only_try_ud_if_no_um=True,
                    )
                    if len(child_features & swap_feature_vals) > 0:
                        return "ambiguous_subjects", item
                    else:
                        item["feature_vals"] = feature_key
                        num_combinations = len(feature_vals) * len(swap_feature_vals)
                        for feat1 in feature_vals:
                            for feat2 in swap_feature_vals:
                                feature_distribution[f"{feat1} -> {feat2}"] += 1 / num_combinations
                        if num_combinations > 1:
                            feature_distribution[feature_key] += 1
                        if item['swap_head'] == 'hebbe':
                            print(item)
                        return "correct_swaps", item
                elif add_item is None:
                    return "extra_pairs", item
                else:
                    return "undefined_features", item
            elif len(feature_vals & swap_feature_vals) > 0:
                wrong_item = dict(item)
                wrong_item[f"swap_{self.take_features_from}"] = swap_form
                wrong_item["feature_vals"] = feature_key

                return "same_features", wrong_item
            else:
                wrong_item = dict(item)
                wrong_item[f"swap_{self.take_features_from}"] = swap_form
                wrong_item["feature_vals"] = feature_key

                return "undefined_features", wrong_item

    def load_inflector(self, lang: str, langcode: str, unimorph_args):
        num_form = 0
        num_lemma = 0
        skip_lang = False

        remove_diacritics = lang in remove_diacritics_langs
        remove_multiples = lang in remove_multiples_langs
        use_udlex = lang in use_udlex_langs

        if use_udlex:
            inflector = UDLexInflector(
                lang,
                upos,
                self.inflection_map,
                ignore_ufeats={"VerbForm", "Subcat"},
                resource_dir=self.resource_dir,
            )
        else:
            inflector = UnimorphInflector(
                langcode,
                self.inflection_map,
                resource_dir=self.resource_dir,
                load_from_pickle=self.load_from_pickle,
                remove_diacritics=remove_diacritics,
                remove_multiples=remove_multiples,
                **unimorph_args,
            )

            if len(inflector) == 0:
                skip_lang = True
                num_lemma = 0
                num_form = 0
            else:
                num_lemma = inflector.num_lemmas
                num_form = inflector.num_forms

            if not inflector.can_feature_swap:
                skip_lang = True

        return inflector, skip_lang, num_lemma, num_form

    def load_corpus(self, lang: str):
        treebank = self.load_treebank(lang)

        lang_head_features = self.filter_args.get("lang_head_features", {})
        lang_child_features = self.filter_args.get("lang_child_features", {})

        head_features = lang_head_features.get("*", {})
        child_features = lang_child_features.get("*", {})
        head_features.update(lang_head_features.get(lang, {}))
        child_features.update(lang_child_features.get(lang, {}))

        filtered_corpus = self.filter_class(
            treebank,
            head_features=head_features,
            child_features=child_features,
            **self.filter_args
        )
        filtered_corpus.filter_treebank(verbose=False)

        if self.treebank_args.get("shuffle_treebank", False):
            random.shuffle(filtered_corpus.filtered_items)

        return filtered_corpus.filtered_items[:self.treebank_args.get("treebank_size")]

    def load_treebank(self, lang: str):
        udlang = gblang2udlang.get(lang, lang).replace(" ", "_")
        remove_diacritics = udlang in remove_diacritics_langs

        treebank = Treebank(
            udlang,
            remove_diacritics=remove_diacritics,
            load_from_pickle=self.load_from_pickle,
            resource_dir=self.resource_dir,
            test_files_only=self.treebank_args.get("test_files_only", False),
        )

        return treebank

    def save_lang(
        self,
        langcode: str,
        minimal_pair_df: pd.DataFrame,
        diagnostics: Dict[str, pd.DataFrame],
    ) -> None:
        if len(minimal_pair_df) > 0:
            save_file = os.path.join(self.save_dir, f"{langcode}.tsv")
            minimal_pair_df = minimal_pair_df.drop(columns=['tree'])
            minimal_pair_df.to_csv(save_file, sep="\t")

        diagnostics_dir = os.path.join(self.save_dir, "diagnostics", langcode)
        os.makedirs(diagnostics_dir, exist_ok=True)

        for name, item_df in diagnostics.items():
            if len(item_df) == 0:
                continue
            save_file = os.path.join(diagnostics_dir, f"{name}.tsv")
            item_df = item_df.drop(columns=['tree'])
            item_df.to_csv(save_file, sep="\t")
