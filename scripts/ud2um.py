import sys
sys.path.append("../src")

from mblimp.ud2um import create_all_unimorph_from_ud
from mblimp.languages import get_ud_langs


if __name__ == "__main__":
    resource_dir = "../resources"
    ud_langs = get_ud_langs(resource_dir)

    create_all_unimorph_from_ud(
        ud_langs,
        resource_dir=resource_dir,
        load_from_pickle=True,
        dup_form_threshold=3.0,
        dup_feat_threshold=10.0,
    )
