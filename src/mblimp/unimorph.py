import os
from typing import *
from unidecode import unidecode

import numpy as np
import pandas as pd

from .languages import latin_to_cyrillic
from .ud2um import load_ud_features
from .unimorph_features import load_um_features

unmarked_features = {
    "Degree",
}
implicit_features = {
    "Number",
    "Gender",
    "Person",
}
BASE = "BASE"
UNDEFINED = "UNDEFINED"
DEFAULTS = {
    "Mood": "IND",
    "Voice": "ACT",
}

UD2UM = {
    "Plur": "PL",
    "Sing": "SG",
    "Pres": "PRS",
    "Past": "PST",
    "Sub": "SBJV",
    "Cnd": "COND",
    "Perf": "PRF",
    "Dual": "DU",
    "Part": "V.PTCP",
}
UM2UD = {v: k for k, v in UD2UM.items()}


def allval2um(val):
    if val == UNDEFINED:
        return UNDEFINED

    val = val.split("_")[-1]

    return UD2UM.get(val, val).upper()


class UnimorphInflector:
    def __init__(
        self,
        langcode: str,
        inflection_map: Tuple[Union[str, List[str]], Optional[Dict[str, str]]],
        resource_dir: Optional[str] = None,
        remove_diacritics: bool = False,
        filter: Dict[str, List[str]] = {},
        remove_multiples: bool = False,
        use_ud_inflections: bool = False,
        verbose: bool = False,
        load_from_pickle: bool = False,
        fill_unk_values: bool = True,
        combine_um_ud: bool = False,
        inflect_wo_ud_features: bool = False,
        prefer_tight_match: bool = True,
        remove_multiword_forms: bool = False,
    ):
        """
        UnimorphInflector loads in a UniMorph file and provides utility
        for inflecting features of a word.

        :param langcode: ISO-639-3 code of the language to be loaded in
        :param inflection_map:
        :param resource_dir:
        :param remove_diacritics:
        :param filter:
        :param remove_multiples:
        :param use_ud_inflections:
        :param verbose:
        :param load_from_pickle:
        :param fill_unk_values:
        :param combine_um_ud:
        :param inflect_wo_ud_features:
        :param prefer_tight_match:
        :param remove_multiword_forms:
        """
        self.langcode = langcode
        self.resource_dir = resource_dir or "."

        if use_ud_inflections:
            self.feat2val, self.val2feat = load_ud_features(
                os.path.join(self.resource_dir, "ud/ud_feats.txt")
            )
        else:
            self.feat2val, self.val2feat = load_um_features()

        self.inflection_map = inflection_map
        self.use_ud_inflections = use_ud_inflections
        self.verbose = verbose
        self.fill_unk_values = fill_unk_values
        self.combine_um_ud = combine_um_ud
        self.inflect_wo_ud_features = inflect_wo_ud_features
        self.prefer_tight_match = prefer_tight_match
        self.remove_multiword_forms = remove_multiword_forms

        self.ud_inflector = None
        self.unimorph_df = self.load_unimorph(
            remove_diacritics,
            filter,
            remove_multiples,
            load_from_pickle=load_from_pickle,
        )
        self.lemma_groups, self.form_groups = self.create_df_groups()

        if inflection_map is not None:
            self.update_inflection_map()

        # (form, ud_features) -> (inflected_forms, inflected_features)
        self.prev_inflections: Dict[Tuple[str, Dict[str, str]], Tuple[List[str], List[str]]] = {}

        if combine_um_ud:
            assert (
                not use_ud_inflections
            ), "Set `use_ud_inflections` to False if combining UM & UD"
            self.ud_inflector = UnimorphInflector(
                langcode,
                inflection_map,
                remove_diacritics=remove_diacritics,
                filter=filter,
                remove_multiples=remove_multiples,
                use_ud_inflections=True,
                verbose=verbose,
                resource_dir=resource_dir,
                load_from_pickle=load_from_pickle,
                fill_unk_values=fill_unk_values,
                combine_um_ud=False,
                inflect_wo_ud_features=inflect_wo_ud_features,
                prefer_tight_match=prefer_tight_match,
                remove_multiword_forms=remove_multiword_forms,
            )
        else:
            self.ud_inflector = None

    def __len__(self) -> int:
        um_len = 0 if self.unimorph_df is None else len(self.unimorph_df)
        ud_len = 0 if self.ud_inflector is None else len(self.ud_inflector)

        return um_len + ud_len

    @property
    def columns(self) -> Set[str]:
        return set() if self.unimorph_df is None else set(self.unimorph_df.columns)

    @property
    def all_columns(self) -> Set[str]:
        um_columns = (
            set() if self.unimorph_df is None else set(self.unimorph_df.columns)
        )
        ud_columns = set() if self.ud_inflector is None else self.ud_inflector.columns

        return um_columns.union(ud_columns)

    @property
    def unique_lemmas(self) -> Set[str]:
        um_lemmas = (
            set() if (self.unimorph_df is None or len(self.unimorph_df) == 0)
            else set(self.unimorph_df.lemma.unique())
        )
        ud_lemmas = (
            set()
            if self.ud_inflector is None
            else self.ud_inflector.unique_lemmas
        )

        return um_lemmas.union(ud_lemmas)

    @property
    def num_lemmas(self) -> int:
        return len(self.unique_lemmas)

    @property
    def unique_forms(self) -> Set[str]:
        um_forms = (
            set() if (self.unimorph_df is None or len(self.unimorph_df) == 0)
            else set(self.unimorph_df.form.unique())
        )
        ud_forms = (
            set()
            if self.ud_inflector is None
            else self.ud_inflector.unique_forms
        )

        return um_forms.union(ud_forms)

    @property
    def num_forms(self) -> int:
        return len(self.unique_forms)

    def unique_values(self, column: str, upos: Optional[str] = None) -> List[str]:
        unique_values = set()
        if column in self.columns:
            df = self.unimorph_df
            if upos is not None:
                df = df[df.upos == upos]

            unique_values.update(df[column].unique())

        if self.ud_inflector is not None:
            unique_values.update(self.ud_inflector.unique_values(column, upos=upos))

        unique_values = [allval2um(val) for val in unique_values if not (val == UNDEFINED or pd.isna(val))]

        return unique_values

    @property
    def can_feature_swap(self) -> bool:
        """ Returns True if the feature of the inflection map is present
        in either the UM or UD dataframe columns.
        """
        um_can_feature_swap = isinstance(self.inflection_map[0], str)

        ud_can_feature_swap = False
        if self.ud_inflector is not None:
            ud_can_feature_swap = isinstance(self.ud_inflector.inflection_map[0], str)

        return um_can_feature_swap or ud_can_feature_swap

    @property
    def has_unimorph_df(self) -> bool:
        return (self.unimorph_df is not None) and (len(self.unimorph_df) > 0)

    def load_unimorph(
        self,
        remove_diacritics: bool,
        filter: Dict[str, List[str]],
        remove_multiples: bool,
        load_from_pickle: bool = False,
    ) -> pd.DataFrame:
        if load_from_pickle:
            if self.use_ud_inflections:
                return self.load_unimorph_pickle("ud_unimorph/ud_pickles", filter)
            else:
                return self.load_unimorph_pickle("unimorph/um_pickles", filter)

        if self.use_ud_inflections:
            path = os.path.join(self.resource_dir, f"ud_unimorph/{self.langcode}")
            if not os.path.isfile(path):
                return None
            df = pd.read_csv(
                path, sep="\t", names=["lemma", "form", "ufeat", "frequency"], header=1
            )
            df = df.drop("frequency", axis=1)
        else:
            path = os.path.join(self.resource_dir, f"unimorph/{self.langcode}/{self.langcode}")
            if not os.path.isfile(path):
                return None
            df = pd.read_csv(path, sep="\t", names=["lemma", "form", "ufeat"])

            if os.path.isfile(path + ".segmentations"):
                path += ".segmentations"
                df_seg = pd.read_csv(
                    path, sep="\t", names=["lemma", "form", "ufeat", "segmentation"]
                )
                df_seg = df_seg.drop("segmentation", axis=1)
                df_seg["ufeat"] = [str(ufeat).replace("|", ";") for ufeat in df_seg.ufeat]

                df = pd.concat([df, df_seg], ignore_index=True).drop_duplicates()

        if remove_diacritics:
            df.lemma = [unidecode(lemma) for lemma in df.lemma]
            df.form = [unidecode(lemma) for lemma in df.form]

        if (self.langcode == "tat") and (not self.use_ud_inflections):
            df.lemma = [latin_to_cyrillic(lemma) for lemma in df.lemma]
            df.form = [latin_to_cyrillic(lemma) for lemma in df.form]

        if remove_multiples:
            df.form = [form.split(", ")[0] for form in df.form]

        ufeat_cols = {x: [] for x in self.feat2val}

        for ufeat in df.ufeat:
            row_ufeats = self.ufeats2dict(ufeat)
            for ufeat_col in self.feat2val:
                ufeat_cols[ufeat_col].append(row_ufeats.get(ufeat_col))

        # Only set the columns for ufeats that *do* occur
        for ufeat_col, ufeat_vals in ufeat_cols.items():
            num_not_none = len([x for x in ufeat_vals if x])
            if num_not_none > 0:
                df[ufeat_col] = ufeat_vals

        df = self.filter_entries(df, filter)
        df = self.expand_multiple_values(df)
        df = self.set_unk_values(df)

        for column in df.columns:
            df[column] = df[column].astype("category")

        return df

    def load_unimorph_pickle(self, pickle_path: str, filter: Dict[str, List[str]]) -> pd.DataFrame:
        path = os.path.join(
            self.resource_dir, pickle_path, f"{self.langcode}.pickle"
        )
        if not os.path.isfile(path):
            if self.verbose:
                print(f"UM Pickle not found at {path}")
            return None
        df = pd.read_pickle(path)

        df = self.filter_entries(df, filter)
        df = self.set_unk_values(df)

        for column in df.columns:
            df[column] = df[column].cat.remove_unused_categories()

        return df

    def pickle_unimorph_df(self, path: str) -> None:
        self.unimorph_df.to_pickle(path)

    def filter_entries(self, df, filter: Dict[str, List[str]]):
        """Only keep entries of a particular feature tag to speed up inflections later."""
        if self.remove_multiword_forms:
            df = df[~df.form.str.contains(" ", na=False)]

        if len(filter) == 0:
            return df

        for ufeat, values in filter.items():
            if ufeat in df.columns:
                values = self.val2ud_um(ufeat, values)
                pos_values = [val for val in values if not val.startswith('-')]
                neg_values = [val[1:] for val in values if val.startswith('-')]

                if len(pos_values) > 0:
                    df = df[df[ufeat].isin(pos_values)]
                if len(neg_values) > 0:
                    df = df[~df[ufeat].isin(neg_values)]
                nan_cols = [col for col in df.columns if sum(pd.isna(df[col])) == len(df)]
                for column in nan_cols:
                    df = df.drop(column, axis=1)

        return df

    def set_unk_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.columns:
            if isinstance(df[column].dtype, pd.CategoricalDtype):
                if BASE not in df[column].cat.categories:
                    df[column] = df[column].cat.add_categories([BASE])
                if UNDEFINED not in df[column].cat.categories:
                    df[column] = df[column].cat.add_categories([UNDEFINED])

        if self.fill_unk_values:
            # Missingness of a feature in UniMorph either indicates that
            # i) the feature is an (unmarked) base class that is distinct from marked ones (e.g. ADJ Degree in Dutch)
            # ii) the form is the same for all instantiations of the feature (ADJ Gender in Dutch)
            for column in unmarked_features & set(df.columns):
                df.loc[pd.isna(df[column]), column] = BASE
            for column in implicit_features & set(df.columns):
                df.loc[pd.isna(df[column]), column] = UNDEFINED
            for column in set(DEFAULTS.keys()) & set(df.columns):
                column_val = self.val2ud_um(column, DEFAULTS[column])

                if (
                    isinstance(df[column].dtype, pd.CategoricalDtype)
                    and column_val not in df[column].cat.categories
                ):
                    df[column] = df[column].cat.add_categories([column_val])
                df.loc[pd.isna(df[column]), column] = column_val

        return df

    def expand_multiple_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_rows = df.to_dict(orient="records")

        idx = 0
        delete_rows = []

        while idx < len(df_rows):
            row = df_rows[idx]
            for ufeat, val in row.items():
                if (ufeat not in {"lemma", "form", "ufeat"}) and isinstance(val, str) and ("+" in val):
                    all_vals = val.split('+')
                    for extra_val in all_vals:
                        extra_row = row.copy()
                        extra_row[ufeat] = self.val2ud_um(ufeat, extra_val)
                        df_rows.append(extra_row)

                    # We expand only one ufeat per loop, if multiple ufeats are disjunctions we get to those
                    # later since the row is appended to the end of the list of extra_rows.
                    # The row with the "x+y+z" value is removed.
                    delete_rows.append(idx)
                    break

            idx += 1

        df = pd.DataFrame(df_rows)
        df = df.drop(index=delete_rows)

        return df

    def create_df_groups(self):
        lemma_groups = None
        form_groups = None

        if self.has_unimorph_df:
            lemma_groups = self.unimorph_df.groupby("lemma")
            form_groups = self.unimorph_df.groupby("form")

        return lemma_groups, form_groups

    def update_inflection_map(self) -> None:
        """UD morphological features are encoded different than UM
        features (e.g. Sing vs. SG). We need to ensure the inflection
        map uses the right feature names, which are set here."""

        # Some languages have subcategories for features (e.g. Number[subj])
        # To use a universal inflection map, we look for the feature
        # that is present the *most* in the unimorph_df columns for inflection.
        if isinstance(self.inflection_map[0], list):
            inflection_column = None
            min_feats_per_column = 0
            for column in self.inflection_map[0]:
                if column in self.columns:
                    feature_counts = Counter(self.unimorph_df[column])
                    non_unk_counts = []
                    for ufeat, counts in feature_counts.items():
                        if (not pd.isna(ufeat)) and (ufeat != UNDEFINED):
                            non_unk_counts.append(counts)

                    if (len(non_unk_counts) > 1) and (min(non_unk_counts) > min_feats_per_column):
                        min_feats_per_column = min(non_unk_counts)
                        inflection_column = column

            self.inflection_map = (inflection_column, self.inflection_map[1])

        swap_ufeat, swap_map = self.inflection_map
        if isinstance(swap_ufeat, list):
            # None of the inflection map features are present in this language
            swap_ufeat = None
            self.inflection_map = (swap_ufeat, swap_map)

        # The UD inflections use a different feature format than UM.
        # To allow the use of a single inflection_map, we map the UM features
        # to UD-compatible features here (e.g. SG becomes Number_Sing).
        if self.use_ud_inflections and (swap_ufeat is not None) and (swap_map is not None):
            new_swap_map = {}
            for feat1, feat2 in swap_map.items():
                new_feat1 = f"{swap_ufeat}_{UM2UD.get(feat1, feat1).title()}"
                new_feat2 = f"{swap_ufeat}_{UM2UD.get(feat2, feat1).title()}"
                assert (
                    new_feat1 in self.val2feat
                ), f"Swapped feature not present {new_feat1}"
                new_swap_map[new_feat1] = new_feat2
            self.inflection_map = (swap_ufeat, new_swap_map)

    def ufeats2dict(self, ufeats: str) -> Dict[str, str]:
        """Translates the unimorph X;Y;Z format to a dictionary"""
        ufeats = (
            str(ufeats)
            .replace("/", "+")
            .replace(",", "+")
            .replace("{", "")
            .replace("}", "")
            .replace("V.PTCP", "V;V.PTCP")
            .replace("V.PCTP", "V;V.PTCP")
        )
        ufeats = ufeats.strip()
        ufeat_vals = ufeats.split(";")
        ufeat_dict = {}

        for val in ufeat_vals:
            if len(val) == 0:
                continue
            elif val in self.val2feat:
                ufeat = self.val2feat[val]
            elif val.strip() in self.val2feat:
                ufeat = self.val2feat[val.strip()]  # Livvi
            elif "." in val:
                val1, val2 = val.split(".")
                ufeat1 = self.val2feat[val1]
                ufeat2 = self.val2feat[val2]
                ufeat_dict[ufeat1] = val1
                ufeat_dict[ufeat2] = val2
                continue
            elif "+" in val:
                subval = val.split("+")[0]
                ufeat = self.val2feat[subval]
            elif val.upper() in self.val2feat:
                ufeat = self.val2feat[val.upper()]
            else:
                ufeat = "UNK"

            if ufeat in ufeat_dict:
                ufeat_dict[ufeat] += f"+{val}"
            else:
                ufeat_dict[ufeat] = val

        return ufeat_dict

    def inflect(
        self,
        form: str,
        ud_features: Dict[str, str],
        strategies: List[Dict[str, Optional[str]]] = [],
    ) -> Tuple[Union[None, str, List[str]], Optional[Set[str]]]:
        prev_inflect_key = (form, frozenset(ud_features.items()))
        if prev_inflect_key in self.prev_inflections:
            return self.prev_inflections[prev_inflect_key]

        if self.unimorph_df is None or len(self.unimorph_df) == 0:
            swap_forms = None
            feature_vals = None
        else:
            swap_forms, feature_vals = self.inflect_features(
                form, ud_features, strategies
            )

            if not self.form_found(swap_forms) and self.inflect_wo_ud_features:
                swap_forms, feature_vals = self.inflect_features(form, {}, strategies)

        if self.ud_inflector is not None and not self.form_found(swap_forms):
            ud_forms, ud_feature_vals = self.ud_inflector.inflect(
                form, ud_features, strategies=strategies
            )
            if swap_forms is None:
                swap_forms = ud_forms
                feature_vals = ud_feature_vals
            elif ud_forms is not None:
                swap_forms.update(ud_forms)
                feature_vals.update(ud_feature_vals)

        if isinstance(swap_forms, set):
            swap_forms = list(swap_forms)

        self.prev_inflections[prev_inflect_key] = swap_forms, feature_vals

        return swap_forms, feature_vals

    def inflect_features(
        self,
        form: str,
        ud_features: Dict[str, str],
        strategies: List[Dict[str, Optional[str]]]
    ) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        """Inflect form based on provided `ud_features`."""
        swap_forms, feature_vals = self.inflect_strategy(form, ud_features)
        if not self.form_found(swap_forms):
            for strat in strategies:
                swap_forms, feature_vals = self.inflect_strategy(
                    form, ud_features, strategy=strat
                )
                if self.form_found(swap_forms):
                    break

        return swap_forms, feature_vals

    @staticmethod
    def form_found(forms: Optional[List[str]]) -> bool:
        return (forms is not None) and (len(forms) == 1)

    def inflect_strategy(
        self,
        form: str,
        ud_features: Dict[str, str],
        strategy: Dict[str, Optional[str]] = {},
    ) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        um_features = self.ud2um_features(ud_features, strategy)
        form_rows = self.form2rows(form, um_features)

        # We skip inflection if no matching rows were found
        if len(form_rows) == 0:
            return None, None

        forms = set()
        feature_vals = set()

        for row_features, feature_val in self.yield_row_features(
            um_features, form_rows, strategy
        ):
            inflected_forms = self.lemma2form(row_features)

            forms.update(inflected_forms)
            feature_vals.add(feature_val)

        return forms, feature_vals

    def ud2um_features(self, ud_features: Dict[str, str], strategy={}) -> Dict[str, str]:
        um_features = {}

        for ufeat, val in ud_features.items():
            if ufeat not in self.columns:
                continue
            elif ufeat == "lemma":
                um_features[ufeat] = val
            elif ufeat in strategy:
                um_features[ufeat] = strategy[ufeat]
            else:
                um_features[ufeat] = self.val2ud_um(ufeat, val)

        for ufeat, val in DEFAULTS.items():
            if (ufeat in self.columns) and (ufeat not in um_features):
                um_features[ufeat] = self.val2ud_um(ufeat, val)

        if self.verbose:
            print("UM Features:", um_features)

        return um_features

    def val2ud_um(self, feat, val) -> Union[str, List[str]]:
        """ Maps a value from a UD/UM feature map to the compatible
        feature format of the inflector (which can be either UM/UD)
        """
        if ("," in val) or ("|" in val) or ("/" in val):
            val = val.replace(",", "+").replace("|", "+").replace("/", "+")
            subvals = val.split("+")
            um_vals = []
            for subval in subvals:
                um_vals.append(self.val2ud_um(feat, subval))
            return um_vals

        if f"{feat}_{val}" in self.val2feat:
            return f"{feat}_{val}"
        elif feat == "lemma":
            return val
        elif isinstance(val, list):
            return [self.val2ud_um(feat, subval) for subval in val]
        elif val in self.val2feat:
            return val
        elif val in UD2UM:
            return self.val2ud_um(feat, UD2UM[val])
        elif val in UM2UD:
            return self.val2ud_um(feat, UM2UD[val])
        elif val.upper() in self.val2feat:
            return val.upper()
        elif val.startswith("-"):
            return "-" + self.val2ud_um(feat, val[1:])
        elif f"{feat}_{val.title()}" in self.val2feat:
            return f"{feat}_{val.title()}"
        else:
            raise ValueError(f"Unknown {feat}: {val}")

    def form2rows(self, form, um_features: Dict[str, str]):
        sub_df = self.partial_df_match(self.form_groups, form, um_features)

        if len(sub_df) > 1:
            sub_df = sub_df.drop_duplicates()

        if self.verbose:
            print("Matched Rows:", sub_df)

        return sub_df

    def lemma2form(self, row_features: Dict[str, str]):
        lemma = row_features.pop("lemma")
        sub_df = self.partial_df_match(self.lemma_groups, lemma, row_features)
        inflected_forms = set(sub_df.form)

        if self.verbose:
            print("Matched Inflected Rows:", sub_df)
            print("Matched Forms:", inflected_forms)

        return inflected_forms

    def partial_df_match(
        self,
        groups,
        group_index,
        features: Dict[str, str],
        prefer_tight_match: Optional[bool] = None
    ):
        """ Find all rows in the morphology dataframe that match the
        group_index (lemma or form) and the features in the provided
        `features` dictionary.
        """
        if len(groups.indices.get(group_index, [])) == 0:
            empty_df = self.unimorph_df.iloc[0:0]
            return empty_df

        sub_df = groups.get_group(group_index)

        if len(sub_df) == 0:
            return sub_df

        mask = np.ones(len(sub_df), dtype=bool)

        for col, val in features.items():
            if isinstance(val, list):
                sub_mask = np.zeros_like(mask)
                for subval in val:
                    sub_mask |= sub_df[col] == subval
                mask &= sub_mask
            elif (val is not None) and (pd.notna(val)):
                if val.startswith("-"):
                    mask &= (sub_df[col] != val[1:]) & (sub_df[col] != UNDEFINED)
                elif col == self.inflection_map[0]:
                    mask &= (sub_df[col] == val)
                else:
                    mask &= (sub_df[col] == val) | (
                        sub_df[col] == UNDEFINED
                    ) | pd.isna(sub_df[col])

        candidate_rows = sub_df[mask]

        if prefer_tight_match is None:
            prefer_tight_match = self.prefer_tight_match
        if (not prefer_tight_match) or (len(candidate_rows) < 2):
            return candidate_rows

        features_added = np.zeros(len(candidate_rows))
        for idx, (_, row) in enumerate(candidate_rows.iterrows()):
            for ufeat, val in row.items():
                if (not pd.isna(val)) or (val == UNDEFINED):
                    features_added[idx] += ufeat not in features

        min_features_added = min(features_added)
        min_features_added_mask = features_added == min_features_added

        return candidate_rows[min_features_added_mask]

    def yield_row_features(
        self,
        um_features: Dict[str, str],
        form_rows: pd.DataFrame,
        strategy: Dict[str, Optional[str]],
    ):
        """
        Based on all matching rows, set and yield the (swapped) features
        we use for finding the inflected form.
        """
        swap_ufeat, swap_map = self.inflection_map

        for _, row in form_rows.iterrows():
            row_features = dict(um_features)  # make a copy
            skip_row = False
            feature_val = None

            if (
                (swap_ufeat not in row.keys())
                or (row[swap_ufeat] == UNDEFINED)
                or pd.isna(row[swap_ufeat])
            ):
                skip_row = True

            for ufeat, val in row.items():
                if (ufeat in {"ufeat", "form"}) or (pd.isna(val)):
                    continue

                if ufeat == swap_ufeat:
                    if swap_map is None:
                        feature_val = allval2um(val)
                        row_features[ufeat] = f"-{val}"
                    elif val not in swap_map:
                        skip_row = True
                        continue
                    else:
                        feature_val = allval2um(val)
                        row_features[ufeat] = self.val2ud_um(ufeat, swap_map[val])
                elif val == UNDEFINED:
                    continue
                else:
                    row_features[ufeat] = val

            # Override values if strategy is set
            for ufeat, val in strategy.items():
                row_features[ufeat] = val

            if self.verbose:
                print("Matched Features", row_features)

            if (not skip_row) and (feature_val is not None):
                yield row_features, feature_val

    def get_form_features(
        self,
        form: str,
        features: Dict[str, str],
        ufeat: Optional[str] = None,
        only_try_ud_if_no_um: bool = False,
    ) -> List[str]:
        um_features = self.ud2um_features(features)
        if ufeat in um_features:
            del um_features[ufeat]

        form_features = set()

        if self.has_unimorph_df:
            df_ufeat = self.inflection_map[0] if ufeat is None else ufeat
            if df_ufeat is not None:
                form_rows = self.partial_df_match(self.form_groups, form, um_features, prefer_tight_match=False)
                if (len(form_rows) > 0) and (df_ufeat in form_rows.columns):
                    form_features.update({allval2um(val) for val in set(form_rows[df_ufeat]) if isinstance(val, str)})

        if self.ud_inflector is not None:
            if only_try_ud_if_no_um and len(form_features) > 0:
                return form_features
            ud_form_features = self.ud_inflector.get_form_features(form, features, ufeat)
            form_features.update(ud_form_features)

        return form_features
