import os
from typing import *

import pandas as pd
from tqdm import *

ud_lang2gb_lang = {
    "Yakut": "Sakha",
    "Persian": "Western Farsi",
    "Western Sierra Puebla Nahuatl": "Zacatlán-Ahuacatlán-Tepetzintla Nahuatl",  # zaca1241, nhi
    "Komi Zyrian": "Komi-Zyrian",
    "Xavante": "Xavánte",
    "Azerbaijani": "North Azerbaijani",
    "Komi Permyak": "Komi-Permyak",
    "Pesh": "Pech",
    "Haitian Creole": "Haitian",
    "Chinese": "Mandarin Chinese",
    "Hebrew": "Modern Hebrew",
    "Armenian": "Eastern Armenian",
    "Arabic": "Standard Arabic",
    "Albanian": "Northern Tosk Albanian",  # Standard Albanian is a standardised form of spoken Albanian based on Tosk.
    "North Sami": "North Saami",
    "Croatian": "Serbian-Croatian-Bosnian",
    "Zaar": "Saya",
    "Yupik": "Central Siberian Yupik",
    "Munduruku": "Mundurukú",
    "Greek": "Modern Greek",
    "Tupinamba": "Tupinambá",
    "Nheengatu": "Nhengatu",
    "Skolt Sami": "Skolt Saami",
    "Uzbek": "Northern Uzbek",
    "Mbya Guarani": "Mbyá Guaraní",
    "Serbian": "Serbian-Croatian-Bosnian",
    "Serbo-Croatian": "Serbian-Croatian-Bosnian",
    "Pashto": "Southern Pashto",
    "Karelian": "Northern Karelian",
    "Gheg": "Gheg Albanian",
    "Classical Armenian": "Classical-Middle Armenian",
    "Apurina": "Apurinã",
    "Kiche": "K'iche'",
    "Slovene": "Slovenian",
    "Ancient Greek (to 1453)": "Ancient Greek",
    "Kirghiz": "Kyrgyz",
    "Norwegian Bokmål": "Norwegian",
    "Old English (ca. 450-1100)": "Old English",
    "Old French (842-ca. 1400)": "Old French",
    "Old Irish (to 900)": "Old Irish",
    "Modern Greek (1453-)": "Modern Greek",
    "Uighur": "Uyghur",
    "Pushto": "Southern Pashto",
    "Egyptian": "Egyptian (Ancient)",
}


def load_grambank(
    path: str,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str], Dict[str, str]]:
    grambank = pd.read_csv(os.path.join(path, "values.csv"))
    gb_langs = pd.read_csv(os.path.join(path, "languages.csv"))

    id2lang = dict(zip(gb_langs.ID, gb_langs.Name))
    lang2id = dict(zip(gb_langs.Name, gb_langs.ID))
    lang2fam = dict(zip(gb_langs.Name, gb_langs.Family_name))

    return grambank, id2lang, lang2id, lang2fam


def get_grambank_feature(
    grambank: pd.DataFrame, lang2id: Dict[str, str], lang: str, param: str
) -> str:
    if lang in lang2id:
        lang = lang2id[lang]

    return grambank[(grambank.Language_ID == lang) & (grambank.Parameter_ID == param)]


def intersect_grambank(grambank, parameters: List[str]):
    filtered_langs = []

    for lang, lang_df in grambank.groupby("Language_ID"):
        keep = True
        for param in parameters:
            if len(lang_df[lang_df.Parameter_ID == param].Value) == 0:
                keep = False
                break

            if lang_df[lang_df.Parameter_ID == param].Value.item() != "1":
                keep = False
                break

        if keep:
            filtered_langs.append(lang)

    return filtered_langs
