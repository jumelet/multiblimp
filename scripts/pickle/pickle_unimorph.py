import os
from pathlib import Path
import pickle
import sys
sys.path.append("../../src")

from mblimp.languages import (
    get_ud_langs,
    remove_diacritics_langs,
    remove_multiples_langs,
    lang2unimorph_lang,
    lang2langcode,
)
from mblimp.unimorph import UnimorphInflector


if __name__ == "__main__":
    resource_dir = "../../resources"
    ud_langs = get_ud_langs(resource_dir)

    for lang in ["Tatar"]:  # ud_langs:
        print(lang)
        umlang = lang2unimorph_lang.get(lang, lang)
        langcode = lang2langcode(umlang)

        inflector = UnimorphInflector(
            langcode,
            None,
            remove_diacritics=(lang in remove_diacritics_langs),
            remove_multiples=(lang in remove_multiples_langs),
            use_ud_inflections=False,
            resource_dir=resource_dir,
            fill_unk_values=False,
        )
        if len(inflector) > 0:
            pickle_path = os.path.join(resource_dir, "unimorph/um_pickles")
            Path(pickle_path).mkdir(parents=True, exist_ok=True)
            pickle_path = os.path.join(pickle_path, f"{langcode}.pickle")
            print("#UM entries", lang, len(inflector.unimorph_df))
            inflector.pickle_unimorph_df(pickle_path)

        # inflector = UnimorphInflector(
        #     langcode,
        #     None,
        #     remove_diacritics=(lang in remove_diacritics_langs),
        #     remove_multiples=(lang in remove_multiples_langs),
        #     use_ud_inflections=True,
        #     resource_dir=resource_dir,
        #     fill_unk_values=False,
        # )
        # if len(inflector) > 0:
        #     pickle_path = os.path.join(resource_dir, "ud_unimorph/ud_pickles")
        #     Path(pickle_path).mkdir(parents=True, exist_ok=True)
        #     pickle_path = os.path.join(pickle_path, f"{langcode}.pickle")
        #     print("#UD entries", lang, len(inflector.unimorph_df))
        #     inflector.pickle_unimorph_df(pickle_path)
