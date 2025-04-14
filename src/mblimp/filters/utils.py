from typing import Dict, Optional

import pandas as pd
from conllu import TokenList


def tokenlist2sen(
    tokenlist: TokenList,
    replacements: Dict[int, str] = {},
    sen_range: Optional[range] = None,
):
    sen = ""
    for idx, tok in enumerate(tokenlist):
        if sen_range is not None and idx not in sen_range:
            break

        if idx in replacements:
            sen += replacements[idx]
        else:
            sen += tok["form"]

        if (tok["misc"] or {}).get("SpaceAfter") == "No":
            continue

        sen += " "

    return sen.strip()


def set_minimal_pairs(df: pd.DataFrame, item_attr: str) -> pd.DataFrame:
    for idx, filtered_item in df.iterrows():
        tree = filtered_item["tree"]
        swap_idx = filtered_item[f"{item_attr}_idx"]
        swap_item = filtered_item[f"swap_{item_attr}"]

        if tree[swap_idx]['form'][0].isupper():
            swap_item = swap_item.capitalize()

        wrong_sen = tokenlist2sen(tree, replacements={swap_idx: swap_item})

        df.loc[idx, "prefix"] = tokenlist2sen(tree, sen_range=range(0, swap_idx))
        df.loc[idx, "wrong_sen"] = wrong_sen

    return df
