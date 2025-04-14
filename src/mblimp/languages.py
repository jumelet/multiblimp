import os
import re
import unicodedata
from glob import glob

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue


def lang2langcode(name: str):
    try:
        return Lang(udlang2iso639.get(name, name)).pt3
    except InvalidLanguageValue:
        return name


udlang2iso639 = {
    "Abkhaz": "Abkhazian",
    "Ancient Greek": "Ancient Greek (to 1453)",
    "Apurina": "Apurinã",
    "Arabic": "Standard Arabic",
    "Assyrian": "Akkadian",
    "Bororo": "Borôro",
    "Buryat": "Buriat",
    "Cantonese": "Yue Chinese",
    "Chukchi": "Chukot",
    "Classical Chinese": "Literary Chinese",
    "Egyptian": "Egyptian (Ancient)",
    "Gheg": "Gheg Albanian",
    "Greek": "Modern Greek (1453-)",
    "Guajajara": "Guajajára",
    "Gwichin": "Gwich'in",
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
    "Naija": "Nigerian Pidgin",
    "North Sami": "Northern Sami",
    "Old East Slavic": "Old Russian",
    "Old French": "Old French (842-ca. 1400)",
    "Old Irish": "Old Irish (to 900)",
    "Ottoman Turkish": "Ottoman Turkish (1500-1928)",
    "Paumari": "Paumarí",
    "Pesh": "Pech",
    "Serbian-Croatian-Bosnian": "Serbo-Croatian",
    "South Levantine Arabic": "Levantine Arabic",
    "Teko": "Tektiteko",
    "Tupinamba": "Tupinambá",
    "Western Sierra Puebla Nahuatl": "Zacatlán-Ahuacatlán-Tepetzintla Nahuatl",  # zaca1241, nhi
    "Xavante": "Xavánte",
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

udlang2treebanks = {
    "Vietnamese": ["VTB"],
    "Hebrew": ["HTB"],
    "Latvian": ["LVTB"],
    "German": ["GSD", "PUD"],
    "Czech": ["CAC", "CLTT", "FicTree", "PDT", "PUD"],
    "Russian": ["GSD", "PUD", "SynTagRus", "Taiga"],
    "Slovenian": ["SSJ"],
    "Faroese": ["OFT"],
    "Icelandic": ["GC", "Modern", "PUD"],
    "English": ["Atis", "EWT", "GENTLE", "GUMReddit", "LinES", "ParTUT", "PUD"],
    "Sanskrit": ["Vedic"],
    "French": ["FQB", "GSD", "ParTUT", "PUD", "Sequoia"],
    "Galician": ["PUD", "TreeGal"],
    "Italian": [
        "ISDT",
        "MarkIT",
        "ParlaMint",
        "ParTUT",
        "PoSTWITA",
        "PUD",
        "TWITTIRO",
        "VIT",
    ],
    "Romanian": ["RRT", "SiMoNERo", "TueCL"],
    "Spanish": ["AnCora", "GSD", "PUD"],
    "Japanese": ["GSD", "PUD", "BCCWJ"],
    "Classical_Chinese": ["Kyoto"],
}

lang2unimorph_lang = {
    "Standard Arabic": "Arabic",
    "Southern Pashto": "Pashto",
    "Guarani": "Mbyá Guaraní",
    "Northern Uzbek": "Uzbek",
    "North Azerbaijani": "Azerbaijani",
}

remove_diacritics_langs = {
    "Latin",
    "Slovenian",
    "Western Farsi",
    # "Northern Uzbek",
    # "Kazakh",
}

remove_multiples_langs = {
    "Modern Greek",
    "Ancient Greek",
}

convert_arabic_to_latin_langs = {
    "Uyghur",
}

use_udlex_langs = {
    # "Galician",
}

skip_langs = {
    "Frisian Dutch",
    "Turkish German",
    "Cappadocian",
    "Maghrebi Arabic French",
    "Pomak",
    "Telugu English",
}


def latin_to_cyrillic(text):
    """ Specific mapping for Tatar.
    Source: https://suzlek.antat.ru/about/TAAT2019/8.pdf
    """
    text = unicodedata.normalize('NFC', text)

    mapping = dict([
        ('Uwa', 'Уa'),
        ('uwa', 'уa'),

        ('Ts', 'Ц'),
        ('ts', 'ц'),
        ('Gö', 'Гө'),
        ('gö', 'гө'),
        ('Gä', 'Гә'),
        ('gä', 'гә'),
        ('Ge', 'Гe'),
        ('ge', 'гe'),
        ('Yı', 'Е'),
        ('yı', 'е'),
        ('Ye', 'Е'),
        ('ye', 'е'),
        ('Yo', 'Й'),
        ('yo', 'й'),
        ('Yö', 'Й'),
        ('yö', 'й'),
        ('Aw', 'Ay'),
        ('aw', 'ay'),
        ('Ya', 'Я'),
        ('ya', 'я'),
        ('Yä', 'Я'),
        ('yä', 'я'),
        ('Yu', 'Ю'),
        ('yu', 'ю'),
        ('Yü', 'Ю'),
        ('yü', 'ю'),
        ('Şç', 'Щ'),
        ('şç', 'щ'),

        ('A', 'А'),
        ('a', 'а'),
        ('Ä', 'Ә'),
        ('ä', 'ә'),
        ('B', 'Б'),
        ('b', 'б'),
        ('C', 'Җ'),
        ('c', 'җ'),
        ('Ç', 'Ч'),
        ('ç', 'ч'),
        ('D', 'Д'),
        ('d', 'д'),
        ('E', 'Е'),
        ('e', 'е'),
        ('F', 'Ф'),
        ('f', 'ф'),
        ('Ğ', 'Г'),
        ('ğ', 'г'),
        ('G', 'Г'),
        ('g', 'г'),
        ('H', 'Һ'),
        ('h', 'һ'),
        ('İ', 'И'),
        ('I', 'И'),
        ('i', 'и'),
        ('I', 'Ы'),
        ('ı', 'ы'),
        ('J', 'Ж'),
        ('j', 'ж'),
        ('K', 'К'),
        ('k', 'к'),
        ('L', 'Л'),
        ('l', 'л'),
        ('M', 'М'),
        ('m', 'м'),
        ('N', 'Н'),
        ('n', 'н'),
        ('Ñ', 'Ң'),
        ('ñ', 'ң'),
        ('O', 'О'),
        ('o', 'о'),
        ('Ö', 'Ө'),
        ('ö', 'ө'),
        ('P', 'П'),
        ('p', 'п'),
        ('Q', 'К'),
        ('q', 'к'),
        ('R', 'Р'),
        ('r', 'р'),
        ('S', 'С'),
        ('s', 'с'),
        ('Ş', 'Ш'),
        ('ş', 'ш'),
        ('T', 'Т'),
        ('t', 'т'),
        ('Y', 'Й'),
        ('y', 'й'),
        ('U', 'У'),
        ('u', 'у'),
        ('Ü', 'Ү'),
        ('ü', 'ү'),
        ('V', 'В'),
        ('v', 'в'),
        ('W', 'В'),
        ('w', 'в'),
        ('X', 'X'),
        ('x', 'x'),
        ('Z', 'З'),
        ('z', 'з'),
    ])

    text = text[0].replace("E", "Э").replace("e", "э") + text[1:]

    idx = 0
    new_text = ""
    while idx < len(text):
        for j in range(3, 0, -1):
            if text[idx:idx+j] in mapping:
                new_text += mapping[text[idx:idx+j]]
                idx += j
                break
            elif j == 1:
                new_text += text[idx]
                idx += 1
                break

    return new_text


def get_ud_langs(resource_dir, ud_dir="ud/ud-treebanks-v2.15/*"):
    ud_pattern = r".*UD_([A-Za-zå_\-]+)-[A-Za-z]+"

    def ud_dir2lang(x):
        return re.match(ud_pattern, x).group(1).replace("_", " ")

    treebank_dirs = glob(os.path.join(resource_dir, ud_dir))
    treebank_langs = map(ud_dir2lang, treebank_dirs)
    treebank_langs = sorted(set(treebank_langs))

    return treebank_langs
