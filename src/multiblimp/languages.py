import os
import re
import unicodedata
from glob import glob

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

from .config import UD_PATH


def lang2langcode(name: str):
    name = name.replace("_", " ")
    name = udlang2iso639.get(name, name)

    try:
        return Lang(name).pt3
    except InvalidLanguageValue:
        return name


udlang2iso639 = {
    "Abkhaz": "Abkhazian",
    "Ancient Greek": "Ancient Greek (to 1453)",
    "Apurina": "Apurinã",
    "Arabic": "Standard Arabic",
    "Assyrian": "Assyrian Neo-Aramaic",  # UD_Assyrian-AS files are aii_*
    "Bororo": "Borôro",
    "Bokota": "Buglere",
    "Buryat": "Buriat",
    "Cantonese": "Yue Chinese",
    "Cappadocian": "Cappadocian Greek",
    "Chintang": "Chhintange",
    "Chukchi": "Chukot",
    "Classical Chinese": "Literary Chinese",
    "Egyptian": "Egyptian (Ancient)",
    "Gheg": "Gheg Albanian",
    "Greek": "Modern Greek (1453-)",
    "Guajajara": "Guajajára",
    "Gwichin": "Gwich'in",
    "Kadiweu": "Kadiwéu",
    "Karo": "Karo (Ethiopia)",
    "Kiche": "K'iche'",
    "Komi Permyak": "Komi-Permyak",
    "Komi Zyrian": "Komi-Zyrian",
    "Kurmanji": "Northern Kurdish",
    "Makurap": "Makuráp",
    "Mbya Guarani": "Mbyá Guaraní",
    "Middle French": "Middle French (ca. 1400-1600)",
    "Munduruku": "Mundurukú",
    "Nheengatu": "Nhengatu",
    "Naga": "suan1234",
    "Naija": "Nigerian Pidgin",
    "North Sami": "Northern Sami",
    "Occitan": "Occitan (post 1500)",
    "Old Occitan": "Old Provençal (to 1500)",
    "Old East Slavic": "Old Russian",
    "Old English": "Old English (ca. 450-1100)",
    "Old French": "Old French (842-ca. 1400)",
    "Old Irish": "Old Irish (to 900)",
    "Ottoman Turkish": "Ottoman Turkish (1500-1928)",
    "Paumari": "Paumarí",
    "Pomak": "qpm",
    "Pesh": "Pech",
    "Runyankore": "Nyankole",
    "Serbian-Croatian-Bosnian": "Serbo-Croatian",
    "Shanghainese": "Wu Chinese",  # Not happy with this, but no iso-639-3 for it
    "South Levantine Arabic": "Levantine Arabic",
    "Teko": "Tektiteko",
    "Tupinamba": "Tupinambá",
    "Western Sierra Puebla Nahuatl": "Zacatlán-Ahuacatlán-Tepetzintla Nahuatl",  # zaca1241, nhi
    "Xavante": "Xavánte",
    "isiXhosa": "Xhosa",
    "Yupik": "Central Siberian Yupik",
    "Zaar": "Saya",
}

gblang2udlang = {
    "Modern Greek": "Greek",
    "Classical-Middle Armenian": "Classical Armenian",
    "Western Farsi": "Persian",
    "Northern Tosk Albanian": "Albanian",
    "North Azerbaijani": "Azerbaijani",
    "Sakha": "Yakut",
    "Standard Arabic": "Arabic",
    "Eastern Armenian": "Armenian",
    "Mandarin Chinese": "Chinese",
    "Northern Uzbek": "Uzbek",
    "Southern Pashto": "Pashto",
    "Northern Karelian": "Karelian",
    "Modern Hebrew": "Hebrew",
}

# Maps a language to the treebanks that should NOT be used for that language.
# All other treebanks of the language are used, including ones added in later
# UD releases. Based on the treebanks in UD v2.18. Treebanks whose words are
# removed (all forms _) are listed explicitly, although their trees would be
# dropped as malformed anyway.
# fmt: off
udlang2excluded_treebanks = {
    "Arabic": ["NYUAD"],             # no text
    "Chinese": [
        "CFL",                       # L2
        "GSD",                       # duplicate of GSDSimp, traditional
        "HK",                        # spoken
    ],
    "Classical_Chinese": ["TueCL"],  # small
    "Czech": ["Poetry"],             # poetry
    "English": [
        "Atis",                      # spoken, flight queries
        "CHILDES",                   # child language, spoken
        "CTeTex",                    # software requirements
        "ESL",                       # L2, words removed
        "ESLSpok",                   # L2, spoken
        "GENTLE",                    # out-of-domain genres
        "GUM",                       # partly spoken
        "GUMReddit",                 # no text
        "Pronouns",                  # synthetic examples
    ],
    "Faroese": ["FarPaHC"],          # historical
    "French": [
        "ALTS",                      # historical
        "GTB",                       # words removed
        "ParisStories",              # spoken
        "PoitevinDIVITAL",           # dialect
        "Rhapsodie",                 # spoken
    ],
    "Galician": ["CTG"],             # automatic parse
    "German": ["LIT"],               # historical, aphorisms
    "Gheg": ["GPS"],                 # code switching
    "Greek": [
        "Cretan",                    # dialect, automatic transcription
        "GLCII",                     # L2
        "Lesbian",                   # dialect
        "Messinian",                 # dialect
    ],
    "Hebrew": [
        "IAHLTknesset",              # spoken, parliament
        "IAHLTwiki",                 # incompatible annotation
        "PostRab",                   # historical
    ],
    "Hindi_English": ["HIENCS"],     # code switching, words removed
    "Icelandic": ["IcePaHC"],        # historical, up to 1350 split into Old Norse
    "Irish": ["TwittIrish"],         # social media
    "Italian": [
        "KIParlaForest",             # spoken
        "MarkIT",                    # marked constructions
        "Old",                       # historical, poetry
        "ParlaMint",                 # parliament transcripts
        "PoSTWITA",                  # social media
        "TWITTIRO",                  # social media
        "Valico",                    # L2
    ],
    "Japanese": [
        "BCCWJ",                     # no text
        "BCCWJLUW",                  # no text, duplicate of BCCWJ
        "GSDLUW",                    # duplicate of GSD
        "KTC",                       # words removed
        "PUDLUW",                    # duplicate of PUD
    ],
    "Korean": [
        "KSL",                       # L2
        "LittlePrince",              # automatic parse
    ],
    "Latvian": ["Cairo"],            # grammar examples
    "Maghrebi_Arabic_French": [
        "Arabizi",                   # code switching, low quality
    ],
    "Mbya_Guarani": ["Dooley"],      # words removed
    "Polish": ["MPDT"],              # historical
    "Portuguese": ["DANTEStocks"],   # social media
    "Romanian": [
        "MolDoRo",                   # grammar examples
        "Nonstandard",               # historical, poetry
        "TueCL",                     # social media
    ],
    "Russian": ["Poetry"],           # poetry
    "Sanskrit": ["UFAL"],            # Classical Sanskrit, small
    "Slovenian": ["SST"],            # spoken
    "Spanish": ["COSER"],            # spoken
    "Swedish": [
        "Old",                       # historical
        "SweLL",                     # L2
    ],
    "Turkish": ["Atis"],             # spoken, flight queries
    "Vietnamese": ["TueCL"],         # spoken
}
# fmt: on

# Sentences to exclude per treebank, as a pattern on their sent_id (see
# Treebank(remove_excluded=True)).
excluded_sentences = {
    # The news headlines of PADT, the first unit of each article, which follow a word order of their own (SVO).
    "UD_Arabic-PADT": r":p1u1$",
    # The spoken dialogues of PDTC (the PDTSC part).
    "UD_Czech-PDTC": r"^pdtsc_",
    # The poetry of Taiga.
    "UD_Russian-Taiga": r"poetry",
}

lang2unimorph_lang = {
    "Standard Arabic": "Arabic",
    "Southern Pashto": "Pashto",
    "Guarani": "Mbyá Guaraní",
    "Northern Uzbek": "Uzbek",
    "North Azerbaijani": "Azerbaijani",
}

remove_diacritics_langs = {
    # Only needed to match UD forms to UniMorph in the original MultiBLiMP
    # pipeline; the word order pairs keep the actual spelling.
    # "Latin",
    # "Slovenian",
    # "Western Farsi",
}

remove_multiples_langs = {
    "Modern Greek",
    "Ancient Greek",
}

# Historical languages: their treebanks mark archaic/variant forms with the Style
# feature (e.g. Style=Arch), so the Style filter in has_typo is skipped for them.
ancient_langs = {
    "Akkadian",
    "Ancient Greek",
    "Ancient Hebrew",
    "Classical Armenian",
    "Classical Chinese",
    "Coptic",
    "Egyptian",
    "Gothic",
    "Hittite",
    "Latin",
    "Middle Armenian",
    "Middle French",
    "Old Church Slavonic",
    "Old East Slavic",
    "Old English",
    "Old French",
    "Old Georgian",
    "Old Irish",
    "Old Norse",
    "Old Occitan",
    "Old Turkish",
    "Ottoman Turkish",
    "Phrygian",
    "Sanskrit",
    "Tupinamba",
    "Umbrian",
}

convert_arabic_to_latin_langs = {
    # Only needed to match UD forms to UniMorph in the original MultiBLiMP
    # pipeline; the word order pairs keep the script of the treebank.
    # "Uyghur",
}

# Code-switched languages and sign languages are omitted from our setup
skip_langs = {
    "Frisian Dutch",
    "Turkish German",
    "Maghrebi Arabic French",
    "Telugu English",
    "Turkish English",
    "Spanish Sign Language",
    "Swedish Sign Language",
}


def latin_to_cyrillic(text):
    """Specific mapping for Tatar.
    Source: https://suzlek.antat.ru/about/TAAT2019/8.pdf
    """
    text = unicodedata.normalize("NFC", text)

    mapping = dict(
        [
            ("Uwa", "Уa"),
            ("uwa", "уa"),
            ("Ts", "Ц"),
            ("ts", "ц"),
            ("Gö", "Гө"),
            ("gö", "гө"),
            ("Gä", "Гә"),
            ("gä", "гә"),
            ("Ge", "Гe"),
            ("ge", "гe"),
            ("Yı", "Е"),
            ("yı", "е"),
            ("Ye", "Е"),
            ("ye", "е"),
            ("Yo", "Й"),
            ("yo", "й"),
            ("Yö", "Й"),
            ("yö", "й"),
            ("Aw", "Ay"),
            ("aw", "ay"),
            ("Ya", "Я"),
            ("ya", "я"),
            ("Yä", "Я"),
            ("yä", "я"),
            ("Yu", "Ю"),
            ("yu", "ю"),
            ("Yü", "Ю"),
            ("yü", "ю"),
            ("Şç", "Щ"),
            ("şç", "щ"),
            ("A", "А"),
            ("a", "а"),
            ("Ä", "Ә"),
            ("ä", "ә"),
            ("B", "Б"),
            ("b", "б"),
            ("C", "Җ"),
            ("c", "җ"),
            ("Ç", "Ч"),
            ("ç", "ч"),
            ("D", "Д"),
            ("d", "д"),
            ("E", "Е"),
            ("e", "е"),
            ("F", "Ф"),
            ("f", "ф"),
            ("Ğ", "Г"),
            ("ğ", "г"),
            ("G", "Г"),
            ("g", "г"),
            ("H", "Һ"),
            ("h", "һ"),
            ("İ", "И"),
            ("i", "и"),
            ("I", "Ы"),
            ("ı", "ы"),
            ("J", "Ж"),
            ("j", "ж"),
            ("K", "К"),
            ("k", "к"),
            ("L", "Л"),
            ("l", "л"),
            ("M", "М"),
            ("m", "м"),
            ("N", "Н"),
            ("n", "н"),
            ("Ñ", "Ң"),
            ("ñ", "ң"),
            ("O", "О"),
            ("o", "о"),
            ("Ö", "Ө"),
            ("ö", "ө"),
            ("P", "П"),
            ("p", "п"),
            ("Q", "К"),
            ("q", "к"),
            ("R", "Р"),
            ("r", "р"),
            ("S", "С"),
            ("s", "с"),
            ("Ş", "Ш"),
            ("ş", "ш"),
            ("T", "Т"),
            ("t", "т"),
            ("Y", "Й"),
            ("y", "й"),
            ("U", "У"),
            ("u", "у"),
            ("Ü", "Ү"),
            ("ü", "ү"),
            ("V", "В"),
            ("v", "в"),
            ("W", "В"),
            ("w", "в"),
            ("X", "X"),
            ("x", "x"),
            ("Z", "З"),
            ("z", "з"),
        ]
    )

    text = text[0].replace("E", "Э").replace("e", "э") + text[1:]

    idx = 0
    new_text = ""
    while idx < len(text):
        for j in range(3, 0, -1):
            if text[idx : idx + j] in mapping:
                new_text += mapping[text[idx : idx + j]]
                idx += j
                break
            elif j == 1:
                new_text += text[idx]
                idx += 1
                break

    return new_text


def get_ud_langs(resource_dir, ud_dir=None, do_skip_langs=True):
    if ud_dir is None:
        ud_dir = UD_PATH

    def ud_dir2lang(x):
        return("_".join(x.split("/")[-1].replace("UD_", "").split("-")[:-1]))

    treebank_dirs = glob(os.path.join(resource_dir, ud_dir, "*"))
    treebank_langs = map(ud_dir2lang, treebank_dirs)
    treebank_langs = sorted(set(treebank_langs))

    if do_skip_langs:
        treebank_langs = [lang for lang in treebank_langs if lang.replace("_", " ") not in skip_langs]

    return treebank_langs
