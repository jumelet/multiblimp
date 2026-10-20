"""
General-purpose CoNLL-U tree utilities.

All functions work with conllu.TokenList objects, as produced by
load_treebank() or deserialize_tree().
"""

from copy import deepcopy

from conllu import TokenList, parse as conllu_parse


# ── Serialisation ─────────────────────────────────────────────────────────────


# conllu cannot round trip "=" or "|" inside a misc value (e.g. the multiword form
# "šar-ti-an-n=a" in Hittite), so multiword forms are percent-escaped when stored.
_MWT_FORM_ESCAPES = [("%", "%25"), ("=", "%3D"), ("|", "%7C")]


def deserialize_tree(serialized: str) -> TokenList:
    """Parse a CoNLL-U string (as stored in the swapped_tree column) back into a TokenList."""
    tree = conllu_parse(serialized)[0]
    for tok in tree:
        misc = tok["misc"] or {}
        if "MWTForm" in misc:
            for char, escaped in reversed(_MWT_FORM_ESCAPES):
                misc["MWTForm"] = misc["MWTForm"].replace(escaped, char)
    return tree


def serialize_tree(tree: TokenList) -> str:
    """Serialize a TokenList to a CoNLL-U string; see deserialize_tree."""
    tree = deepcopy(tree)
    for tok in tree:
        misc = tok["misc"] or {}
        if "MWTForm" in misc:
            for char, escaped in _MWT_FORM_ESCAPES:
                misc["MWTForm"] = misc["MWTForm"].replace(char, escaped)
    return tree.serialize()


# ── Surface string ────────────────────────────────────────────────────────────


def tree_to_sentence(tree) -> str:
    """
    Reconstruct the surface string from a TokenList, respecting SpaceAfter=No
    annotations in the misc field.

    The words of a multiword token (stored as MWT/MWTForm in misc, see
    multiblimp.treebank.remove_multiword_tokens) are rendered as its surface form,
    e.g. "zu" + "dem" as "zum".
    """
    tokens = list(tree)
    parts = []
    i = 0
    while i < len(tokens):
        misc = tokens[i]["misc"] or {}
        form, last = tokens[i]["form"], i
        if "MWTForm" in misc:
            form = misc["MWTForm"]
            while (
                last + 1 < len(tokens)
                and (tokens[last + 1]["misc"] or {}).get("MWT") == misc["MWT"]
            ):
                last += 1
        no_space = (tokens[last]["misc"] or {}).get("SpaceAfter") == "No"
        parts.append(form if no_space else f"{form} ")
        i = last + 1
    return "".join(parts).strip()


# ── Visualisation ─────────────────────────────────────────────────────────────


def _to_displacy_data(tree) -> dict:
    """Convert a TokenList to spaCy displacy manual-render format."""
    words = [{"text": tok["form"], "tag": tok["upos"] or ""} for tok in tree]
    arcs = []
    for tok in tree:
        if tok["head"] == 0:
            continue
        dep = tok["id"] - 1  # 0-based dependent position
        head = tok["head"] - 1  # 0-based head position
        if dep < head:
            arcs.append(
                {"start": dep, "end": head, "label": tok["deprel"], "dir": "left"}
            )
        else:
            arcs.append(
                {"start": head, "end": dep, "label": tok["deprel"], "dir": "right"}
            )
    return {"words": words, "arcs": arcs}


def visualize_tree(tree, jupyter: bool = True, compact: bool = False, **kwargs):
    """
    Render a CoNLL-U TokenList as a dependency arc diagram.

    Parameters
    ----------
    tree     : TokenList (from load_treebank, deserialize_tree, etc.)
    jupyter  : True to render inline HTML in a Jupyter notebook cell.
               False returns the raw SVG string instead.
    compact  : True for a more compact layout (spaCy compact mode).
    **kwargs : passed through to spacy.displacy.render (e.g. options={...}).

    Examples
    --------
    visualize_tree(treebank[0])
    visualize_tree(deserialize_tree(row.swapped_tree))

    # Side-by-side in one cell:
    visualize_tree(original_tree)
    visualize_tree(swapped_tree)
    """
    from spacy import displacy

    options = kwargs.pop("options", {})
    options.setdefault("compact", compact)
    options.setdefault("distance", 100)

    data = _to_displacy_data(tree)
    return displacy.render(
        [data], style="dep", manual=True, jupyter=jupyter, options=options, **kwargs
    )


def compare_trees(
    tree_a: TokenList | str, tree_b: TokenList | str, jupyter: bool = True, **kwargs
):
    """
    Render two trees side by side in a single displacy call.
    Useful for comparing an original tree and its swapped version.

    tree_a and tree_b may be TokenLists or serialized CoNLL-U strings.
    """
    from spacy import displacy

    if isinstance(tree_a, str):
        tree_a = deserialize_tree(tree_a)
    if isinstance(tree_b, str):
        tree_b = deserialize_tree(tree_b)

    options = kwargs.pop("options", {})
    options.setdefault("distance", 100)

    data = [_to_displacy_data(tree_a), _to_displacy_data(tree_b)]
    return displacy.render(
        data, style="dep", manual=True, jupyter=jupyter, options=options, **kwargs
    )
