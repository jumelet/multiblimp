import os
import sys
import random

from tqdm import tqdm_notebook

sys.path.append("../src")

from multiblimp.languages import get_ud_langs

from word_order.process_treebank import load_treebank, create_word_order_df
from word_order.prediction_target import *
from word_order.utils import MAX_TREEBANK_LEN


random.seed(42)


if __name__ == "__main__":
    for target in get_all_deprel_targets():
        print(target.child_deprels)

        deprel_dir = "_".join(target.child_deprels)
        resource_dir = "/media/jaap/81b6ce8a-28e5-4eda-9c68-b13e0637cc4f/WORD_ORDER/resources"
        word_order_dir = f"/media/jaap/81b6ce8a-28e5-4eda-9c68-b13e0637cc4f/WORD_ORDER/treebank_features/{deprel_dir}"
        
        langs = get_ud_langs(resource_dir)
        
        for lang in tqdm_notebook(sorted(langs)):
            if os.path.exists(f"{word_order_dir}/{lang.replace(' ', '_')}.csv"):
                print("skipping", lang)
                continue
        
            treebank = load_treebank(lang, resource_dir, max_treebank_len=MAX_TREEBANK_LEN)
            
            df = create_word_order_df(
                lang=lang, 
                treebank=treebank,
                target=target, 
                resource_dir=resource_dir,
                save_to=word_order_dir,
                max_treebank_len=MAX_TREEBANK_LEN,
                drop_singleton_columns=True,
                lexicalize=True,
            )
        
            print(lang, len(df))