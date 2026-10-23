"""Speaker numbers and other language metadata from LinguaMeta.

LinguaMeta (Ritchie et al., LREC-COLING 2024) unifies metadata for 7,500+ languages:
https://github.com/google-research/url-nlp/tree/main/linguameta

The summary table (linguameta.tsv) holds the main fields per language; the
per-language JSON files additionally give the source of every value and the speaker
numbers per locale. Languages are matched on their ISO 639-3 code, or on their
Glottocode for languages without one (e.g. Naga, "suan1234").
"""

import json
import urllib.request

import pandas as pd

LINGUAMETA_URL = "https://raw.githubusercontent.com/google-research/url-nlp/main/linguameta"
LIST_COLUMNS = ["writing_systems", "locales", "cldr_official_status", "individual_language_bcp_47_code"]


def load_linguameta() -> pd.DataFrame:
    """The LinguaMeta summary table, one row per language (BCP-47 code)."""
    df = pd.read_csv(f"{LINGUAMETA_URL}/linguameta.tsv", sep="\t", dtype=str, keep_default_na=False)
    df["estimated_number_of_speakers"] = pd.to_numeric(df.estimated_number_of_speakers, errors="coerce").astype("Int64")
    df["is_macrolanguage"] = df.is_macrolanguage == "True"
    for col in LIST_COLUMNS:
        df[col] = df[col].map(lambda val: val.split(", ") if val else [])
    return df


def load_language_details(bcp_47_code: str) -> dict:
    """The full LinguaMeta record of a language, including the source of each value."""
    with urllib.request.urlopen(f"{LINGUAMETA_URL}/data/{bcp_47_code}.json") as f:
        return json.load(f)


def get_linguameta(codes: list[str], with_details: bool = False) -> dict[str, dict]:
    """
    Map each code in `codes` (ISO 639-3, or a Glottocode) to its LinguaMeta row, as a
    dict of the summary table's columns. With `with_details`, the full record (with
    sources and per-locale speaker numbers) is added under "details"; this fetches
    one file per language. Codes not found in LinguaMeta are left out.
    """
    df = load_linguameta()
    by_iso = df[df.iso_639_3_code != ""].set_index("iso_639_3_code")
    by_glottocode = df[df.glottocode != ""].set_index("glottocode")

    mapping = {}
    for code in codes:
        if code in by_iso.index:
            row = by_iso.loc[code]
        elif code in by_glottocode.index:
            row = by_glottocode.loc[code]
        else:
            continue
        if isinstance(row, pd.DataFrame):  # several entries for the code: keep the first
            row = row.iloc[0]
        info = row.to_dict()
        if with_details:
            info["details"] = load_language_details(info["bcp_47_code"])
        mapping[code] = info

    return mapping
