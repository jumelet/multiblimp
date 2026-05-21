import os
import re

from sklearn.tree import plot_tree
from sklearn import set_config

import matplotlib.pyplot as plt


def pprint_node(txt):
    new_txt = (
        txt.replace(" <= 0.5", "")
        .replace("000", "")
        .replace(".0,", ",")
        .replace(".0]", "]")
        .strip()
    )

    if len(new_txt.split("_")) == 3:
        splits = new_txt.split("_")
        new_txt = f"{splits[0]}_{splits[1]} = {splits[2]}"
    elif len(new_txt.split("_")) == 4:
        splits = new_txt.split("_")
        new_txt = f"{splits[0]}_{splits[1]}_{splits[2]} = {splits[3]}"
    elif "True" in new_txt and not "\n" in new_txt:
        new_txt = "     False     "  # labels must be flipped, trust me it is right
    elif "False" in new_txt and not "\n" in new_txt:
        new_txt = "     True     "

    if " = nan" in new_txt:
        new_txt = new_txt.replace(" = nan", " is not set")

    return new_txt


def plot_dt(model, save_to=None, show_plot=True, class_names=None):
    set_config(transform_output="default")

    clf = model.named_steps["clf"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = [x.split("__")[1] for x in preprocessor.get_feature_names_out()]
    class_names = class_names or model.classes_

    fig = plt.figure(figsize=(25, 15))
    artists = plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        node_ids=True,
    )

    for node_id, artist in enumerate(artists):
        txt = artist.get_text()
        m = re.match(r"node #(\d+)\n(.*)", txt, re.S)
        if m is not None:
            node_id = int(m.group(1))
            rest = m.group(2)
            new_txt = pprint_node(rest)

            artist.set_text(f"[{node_id}] {new_txt}")
        else:
            new_txt = pprint_node(txt)
            artist.set_text(new_txt)

    if save_to is not None:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)