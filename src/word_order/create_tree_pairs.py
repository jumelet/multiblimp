import copy
import os
import unicodedata
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from conllu import TokenList

from .process_treebank import PredictionTarget
from .tree_utils import serialize_tree, tree_to_sentence
from .utils import shorten_cls
from .validate_swap import JointClassifier, score_swap_df


def get_subtree_indices(tokens, index):
    """
    Return all token indices for which `index` is an ancestor.
    `index` should be 1-based, like CoNLL-U IDs.
    """
    # Build child lists
    children = {}
    for tok in tokens:
        head = tok["head"] - 1
        tid = tok["id"] - 1
        if head is not None:
            children.setdefault(head, []).append(tid)

    # Collect descendants
    stack = children.get(index, [])[:]  # immediate children
    subtree = set(stack)

    while stack:
        node = stack.pop()
        for child in children.get(node, []):
            if child not in subtree:
                subtree.add(child)
                stack.append(child)

    subtree.add(int(index))

    return sorted(subtree)


def _build_swap_logic(target, dep_ids, h_ids):
    """
    Build the swap_logic dict for the current row.

    Each entry maps a target order string to (base_order, ids_to_move, pivot_idx).
    Within each source dict the insertion order matters: later entries may reference
    base orders produced by earlier entries in the same iteration.
    """
    h = shorten_cls("head", target)

    if len(target.child_deprels) == 1:
        dep1 = target.child_deprels[0]
        a = shorten_cls(dep1, target)
        a_ids = dep_ids[dep1]
        ah, ha = f"{a}{h}", f"{h}{a}"
        return {
            ah: {ha: (ah, a_ids, h_ids[-1])},
            ha: {ah: (ha, a_ids, h_ids[0])},
        }

    dep1, dep2 = target.child_deprels
    a = shorten_cls(dep1, target)
    b = shorten_cls(dep2, target)
    a_ids = dep_ids[dep1]
    b_ids = dep_ids[dep2]

    # All 6 permutations of (dep1, head, dep2), mirroring the SVO structure:
    # a ↔ subj, h ↔ verb/head, b ↔ obj
    ahb, abh = f"{a}{h}{b}", f"{a}{b}{h}"
    hab, hba = f"{h}{a}{b}", f"{h}{b}{a}"
    bha, bah = f"{b}{h}{a}", f"{b}{a}{h}"

    return {
        ahb: {
            abh: (ahb, b_ids, h_ids[0]),
            bah: (ahb, b_ids, a_ids[0]),
            hab: (ahb, a_ids, h_ids[-1]),
            hba: (ahb, a_ids, b_ids[-1]),
            bha: (hab, b_ids, h_ids[0] - len(a_ids)),
        },
        abh: {
            ahb: (abh, b_ids, h_ids[-1]),
            bah: (abh, b_ids, a_ids[0]),
            hab: (abh, h_ids, a_ids[0]),
            bha: (abh, a_ids, h_ids[-1]),
            hba: (bah, h_ids, b_ids[0] - len(a_ids)),
        },
        hab: {
            ahb: (hab, a_ids, h_ids[0]),
            hba: (hab, b_ids, a_ids[0]),
            bha: (hab, b_ids, h_ids[0]),
            abh: (hab, h_ids, b_ids[-1]),
            bah: (ahb, b_ids, a_ids[0] - len(h_ids)),
        },
        hba: {
            hab: (hba, a_ids, b_ids[-1]),
            bha: (hba, b_ids, h_ids[0]),
            ahb: (hba, a_ids, h_ids[0]),
            bah: (hba, h_ids, a_ids[-1]),
            abh: (bha, a_ids, b_ids[0] - len(h_ids)),
        },
        bha: {
            bah: (bha, a_ids, h_ids[-1]),
            hba: (bha, h_ids, b_ids[0]),
            hab: (bha, b_ids, a_ids[-1]),
            abh: (bha, a_ids, b_ids[0]),
            ahb: (hba, a_ids, h_ids[0] - len(b_ids)),
        },
        bah: {
            bha: (bah, a_ids, h_ids[-1]),
            abh: (bah, a_ids, b_ids[0]),
            ahb: (bah, b_ids, h_ids[-1]),
            hba: (bah, h_ids, b_ids[0]),
            hab: (abh, h_ids, a_ids[0] - len(b_ids)),
        },
    }


def _get_space_after(token):
    return (token["misc"] or {}).get("SpaceAfter") == "No"


def _set_space_after(token, no_space):
    if no_space:
        if token["misc"] is None:
            token["misc"] = {}
        token["misc"]["SpaceAfter"] = "No"
    else:
        if token["misc"] is not None:
            token["misc"].pop("SpaceAfter", None)
            if not token["misc"]:
                token["misc"] = None


@dataclass
class Orthography:
    """How a treebank writes case and word boundaries; see detect_orthography."""

    # Whether sentences start with a capital (used when the first word of a
    # sentence does not start with a cased letter, e.g. a number).
    sentence_case: bool = True
    # Whether the orthography has Turkic dotted/dotless i: i <-> İ and ı <-> I.
    dotted_i: bool = False
    # Whether words are separated by spaces (not the case for e.g. Chinese,
    # Japanese or Thai, where SpaceAfter=No says nothing about word boundaries).
    spaced_words: bool = True


def detect_orthography(treebank) -> Orthography:
    """
    Derive the orthography of a treebank: sentence case if most sentences whose
    first word starts with a cased letter start with a capital (not the case for
    e.g. Egyptian or Akkadian transliterations); Turkic dotted/dotless i if ı or İ
    occurs in more than 0.5% of the tokens; and spaced words unless most words are
    glued to a following word (SpaceAfter=No), as in Chinese, Japanese or Thai.
    """
    n_cased = n_upper = n_tokens = n_dotless = n_words = n_glued = 0
    for tree in treebank:
        tokens = list(tree)
        first = next((tok["form"] for tok in tokens if tok["upos"] != "PUNCT"), "")
        n_cased += first[:1].isupper() or first[:1].islower()
        n_upper += first[:1].isupper()
        n_tokens += len(tokens)
        n_dotless += sum("ı" in tok["form"] or "İ" in tok["form"] for tok in tokens)
        for i, tok in enumerate(tokens):
            if tok["upos"] != "PUNCT":
                n_words += 1
                n_glued += (
                    i + 1 < len(tokens)
                    and _get_space_after(tok)
                    and tokens[i + 1]["upos"] != "PUNCT"
                )
    return Orthography(
        sentence_case=n_cased > 0 and n_upper / n_cased > 0.5,
        dotted_i=n_tokens > 0 and n_dotless / n_tokens > 0.005,
        spaced_words=n_words == 0 or n_glued / n_words < 0.5,
    )


def _change_first_case(text, upper, convention):
    """
    Upper- or lower-case the first letter of text. Changes that would not round
    trip (e.g. Egyptian ẖ, whose capital is H + combining macron) are skipped.
    """
    first = text[:1]
    if convention.dotted_i and first in "iıİI":
        new = {"i": "İ", "ı": "I"}.get(first, first) if upper else {"İ": "i", "I": "ı"}.get(first, first)
    else:
        new = first.upper() if upper else first.lower()
        if len(new) != 1 or (new.lower() if upper else new.upper()) != first:
            return text
    return new + text[1:]


def _set_first_case(token, upper, convention):
    """Change the case of a token, including the multiword token form it starts."""
    token["form"] = _change_first_case(token["form"], upper, convention)
    misc = token["misc"] or {}
    if "MWTForm" in misc:
        misc["MWTForm"] = _change_first_case(misc["MWTForm"], upper, convention)


def _capitalized_by_itself(token):
    """Proper nouns and words whose lemma is capitalised (German nouns, English "I")."""
    lemma = token.get("lemma") or "_"
    return token.get("upos") == "PROPN" or (lemma != "_" and lemma[0].isupper())


def _first_word_idx(tokens):
    """Index of the first token that is not punctuation, or None."""
    return next((i for i, tok in enumerate(tokens) if tok["upos"] != "PUNCT"), None)


def _uses_sentence_case(first_word, convention):
    """
    Whether the sentence starts with a capital: decided by its first word, unless
    that word doesn't start with a cased letter or is capitalised by itself, in
    which case the treebank's convention decides.
    """
    form = (first_word["misc"] or {}).get("MWTForm", first_word["form"])
    # A first word starting with two capitals (e.g. a headline "GIRO ederrean ...",
    # "IT-sjefen" or an acronym) says nothing about sentence case, and lower-casing
    # it would give forms like "gIRO".
    letters = [ch for ch in form if ch.isalpha()]
    if len(letters) >= 2 and letters[0].isupper() and letters[1].isupper():
        return False
    if form[:1].islower():
        return False
    if form[:1].isupper() and not _capitalized_by_itself(first_word):
        return True
    return convention.sentence_case


def swap_tree_tokens(base_tokens, ids, pivot_idx, orthography=Orthography()):
    """
    Reorder `base_tokens` by moving the subtree at 0-based `ids` relative to
    `pivot_idx` (0-based).  Returns a new list of deep-copied token dicts with
    updated `id`, `head`, `misc` (SpaceAfter), and `form` (capitalisation),
    or None if the subtree is non-projective w.r.t. the pivot.
    """
    sorted_ids = sorted(ids)
    ids_set = set(sorted_ids)
    len_ids = len(sorted_ids)

    if all(i > pivot_idx for i in sorted_ids):
        insert_pos = pivot_idx  # subtree moves left of pivot
        right_to_left = True
    elif all(i < pivot_idx for i in sorted_ids):
        insert_pos = pivot_idx - len_ids + 1  # subtree moves right of pivot
        right_to_left = False
    else:
        return None

    # Capture SpaceAfter from relevant boundary tokens before reordering.
    orig_nsa_subtree_last = _get_space_after(base_tokens[sorted_ids[-1]])
    orig_nsa_pivot = _get_space_after(base_tokens[pivot_idx])

    # Reorder (still pointing at the original dicts — copy happens next).
    items = [base_tokens[i] for i in sorted_ids]
    remainder = [base_tokens[i] for i in range(len(base_tokens)) if i not in ids_set]
    new_tokens = [
        copy.deepcopy(t)
        for t in remainder[:insert_pos] + items + remainder[insert_pos:]
    ]

    # Build old 1-based id → new 1-based id mapping before we overwrite ids.
    old_id_to_new_id = {
        tok["id"]: new_pos + 1 for new_pos, tok in enumerate(new_tokens)
    }

    # Update id and head on every token.
    for new_pos, tok in enumerate(new_tokens):
        tok["id"] = new_pos + 1
        if tok["head"] != 0:
            tok["head"] = old_id_to_new_id[tok["head"]]

    # SpaceAfter, at the two places where the token order changed:
    # 1. Where the subtree left: whatever followed it now follows the last
    #    non-subtree token before the subtree's end (the pivot, an intervening token
    #    or a gap token). If the subtree's last token was glued to what followed
    #    (e.g. a full stop), that token is now glued to it instead. Its own
    #    SpaceAfter=No (e.g. an opening bracket) is kept.
    prev_ids = [i for i in range(sorted_ids[-1]) if i not in ids_set]
    if prev_ids and orig_nsa_subtree_last:
        new_prev_idx = old_id_to_new_id[base_tokens[prev_ids[-1]]["id"]] - 1
        _set_space_after(new_tokens[new_prev_idx], True)
    # 2. Where the subtree landed, next to the pivot: a new word boundary, which is
    #    a space, unless the language is written without spaces (e.g. Chinese).
    #    Right-to-left: subtree at [pivot_idx .. pivot_idx+len_ids-1], pivot after it.
    #    Left-to-right: pivot at pivot_idx-len_ids, subtree after it, taking over
    #                   whatever followed the pivot.
    no_space_between_words = not orthography.spaced_words
    if right_to_left:
        _set_space_after(new_tokens[pivot_idx + len_ids - 1], no_space_between_words)
    else:
        _set_space_after(new_tokens[pivot_idx], orig_nsa_pivot)
        _set_space_after(new_tokens[pivot_idx - len_ids], no_space_between_words)

    # Moving a non-contiguous subtree closes its gaps: a subtree token before a gap
    # now directly precedes the next subtree token, so its SpaceAfter (which
    # referred to the gap) no longer applies.
    for i in sorted_ids[:-1]:
        if i + 1 not in ids_set:
            gap_edge = new_tokens[old_id_to_new_id[base_tokens[i]["id"]] - 1]
            _set_space_after(gap_edge, no_space_between_words)

    # Capitalisation: if another word became sentence-initial (leading punctuation
    # aside), capitalise it and lower-case the former first word, but only if the
    # sentence starts with a capital in the first place.
    old_first = _first_word_idx(base_tokens)
    new_first = _first_word_idx(new_tokens)
    if old_first is not None and new_first is not None:
        old_first_now = old_id_to_new_id[base_tokens[old_first]["id"]] - 1
        if old_first_now != new_first and _uses_sentence_case(
            base_tokens[old_first], orthography
        ):
            _set_first_case(new_tokens[new_first], True, orthography)
            if not _capitalized_by_itself(new_tokens[old_first_now]):
                _set_first_case(new_tokens[old_first_now], False, orthography)

    return new_tokens


def _moved_root_original_id(base_tokens, ids) -> int:
    """Return the 1-based id of the subtree root in the original tree."""
    sorted_ids = sorted(ids)
    subtree_1based = {base_tokens[i]["id"] for i in sorted_ids}
    root_pos = next(
        (i for i in sorted_ids if base_tokens[i]["head"] not in subtree_1based),
        sorted_ids[0],
    )
    return base_tokens[root_pos]["id"]


def _moved_root_new_id(base_tokens, ids, pivot_idx) -> int:
    """
    Return the new 1-based id (in the swapped tree) of the root of the moved
    subtree — the token whose head is outside the subtree (i.e. the topmost node).
    """
    sorted_ids = sorted(ids)
    subtree_1based = {base_tokens[i]["id"] for i in sorted_ids}
    root_pos = next(
        (i for i in sorted_ids if base_tokens[i]["head"] not in subtree_1based),
        sorted_ids[0],
    )
    rank = sorted_ids.index(root_pos)
    if all(i > pivot_idx for i in sorted_ids):
        insert_pos = pivot_idx
    else:
        insert_pos = pivot_idx - len(sorted_ids) + 1
    return insert_pos + rank + 1  # 1-based


def _strands_punct(tokens, ids) -> bool:
    """
    Whether moving the subtree at 0-based `ids` would leave punctuation stranded:
    - punctuation on a boundary between the subtree and the rest, i.e. at the
      subtree's outer edges or next to one of its gaps ("in de , die ... partij");
    - punctuation on both sides of the subtree, which become adjacent (", ,").
    """
    ids_set = set(ids)
    lo, hi = min(ids), max(ids)

    def is_punct(i):
        return 0 <= i < len(tokens) and tokens[i]["upos"] == "PUNCT"

    def on_boundary(i):
        return (i - 1 in ids_set) != (i in ids_set) or (i + 1 in ids_set) != (i in ids_set)

    return any(is_punct(i) and on_boundary(i) for i in range(lo, hi + 1)) or (
        is_punct(lo - 1) and is_punct(hi + 1)
    )


def _can_open_sentence(punct):
    """Brackets, quotes and dashes can start a sentence; a comma or full stop can't."""
    first = punct[:1]
    return first in "\"'¿¡" or unicodedata.category(first) in ("Ps", "Pi", "Pf", "Pd")


def _strands_punct_at_start(base_tokens, new_tokens, ids, orthography) -> bool:
    """
    Whether moving a phrase away from the start of the sentence leaves punctuation
    at the start that belonged to it: a comma or full stop, or a bracket, quote or
    dash glued to the phrase's last word (e.g. Czech "Budeme- li" -> "- Li ...").
    """
    if new_tokens[0]["upos"] != "PUNCT" or base_tokens[0]["upos"] == "PUNCT":
        return False
    if not _can_open_sentence(new_tokens[0]["form"]):
        return True
    # The punctuation now at the start directly followed the moved phrase; in
    # languages written without spaces, SpaceAfter=No says nothing about that.
    ids_set = set(ids)
    first_kept = min(i for i in range(len(base_tokens)) if i not in ids_set)
    return orthography.spaced_words and _get_space_after(base_tokens[first_kept - 1])


def _part_of_word(tokens, j) -> bool:
    """
    Whether token j belongs to a written word. A written word is what lies between
    two spaces: a chain of tokens linked by SpaceAfter=No. Punctuation inside such
    a chain belongs to the word (e.g. the hyphens in "90-hectare" or "Není-li", but
    also "thing--makes"); punctuation at its edges does not (the full stop in
    "Juan.").
    """
    if tokens[j]["upos"] != "PUNCT":
        return True
    k = j
    while k > 0 and _get_space_after(tokens[k - 1]) and tokens[k - 1]["upos"] == "PUNCT":
        k -= 1
    word_before = k > 0 and _get_space_after(tokens[k - 1])
    k = j
    while k < len(tokens) - 1 and _get_space_after(tokens[k]) and tokens[k + 1]["upos"] == "PUNCT":
        k += 1
    word_after = k < len(tokens) - 1 and _get_space_after(tokens[k])
    return word_before and word_after


def _inside_written_word(tokens, i, orthography) -> bool:
    """
    Whether tokens i and i+1 are part of the same written word, or of the same word
    as annotated by the treebank (the long unit words of Japanese, which is written
    without spaces).
    """
    if not 0 <= i < len(tokens) - 1:
        return False
    if (tokens[i + 1]["misc"] or {}).get("LUWBILabel") == "I":
        return True
    return (
        orthography.spaced_words
        and _get_space_after(tokens[i])
        and _part_of_word(tokens, i)
        and _part_of_word(tokens, i + 1)
    )


# Relations that make their dependent part of the same lexical unit as the head,
# e.g. "apple pie", "New York", "pick up", Hindi light verbs or Vietnamese words
# that are written as separate syllables.
LEXICAL_UNIT_DEPRELS = {"compound", "flat", "fixed", "goeswith"}


def _lexical_unit(tokens, idx, ids) -> set:
    """
    The 0-based indices of the lexical unit token `idx` belongs to: all tokens
    connected to it by LEXICAL_UNIT_DEPRELS relations (in either direction),
    leaving out the moved subtree at 0-based `ids`.
    """

    def in_unit_with(i, j):  # token j is i's head or dependent in such a relation
        tok_i, tok_j = tokens[i], tokens[j]
        return (tok_j["head"] == i + 1 and tok_j["deprel"].split(":")[0] in LEXICAL_UNIT_DEPRELS) or (
            tok_i["head"] == j + 1 and tok_i["deprel"].split(":")[0] in LEXICAL_UNIT_DEPRELS
        )

    ids_set = set(ids)
    unit, todo = {idx}, [idx]
    while todo:
        i = todo.pop()
        for j in range(len(tokens)):
            if j not in unit and j not in ids_set and in_unit_with(i, j):
                unit.add(j)
                todo.append(j)
    return unit


def _word_edge_pivot(tokens, ids, pivot_idx, orthography) -> int:
    """
    If the subtree at 0-based `ids` would land inside a written word next to the
    pivot (e.g. before "up" in "mark-up", or after "tug" in "tug-of-war"), or inside
    the pivot's lexical unit (e.g. between "apple" and "pie"), move the pivot to the
    edge of that word or unit, so the subtree lands before it when moving left and
    after it when moving right. Words and units can follow on from each other (e.g.
    Arabic "بالنسبة للمرضى", where the clitic ل forms a unit with the preceding word),
    so the whole chain is skipped.
    """

    def joined(i):  # tokens i and i+1 must stay together
        return 0 <= i < len(tokens) - 1 and (
            _inside_written_word(tokens, i, orthography) or i + 1 in _lexical_unit(tokens, i, ids)
        )

    if min(ids) > pivot_idx:
        while joined(pivot_idx - 1):
            pivot_idx -= 1
    else:
        while joined(pivot_idx):
            pivot_idx += 1
    return pivot_idx


def _splits_written_word(tokens, ids, pivot_idx, orthography) -> bool:
    """
    Whether moving the subtree at 0-based `ids` next to `pivot_idx` separates two
    tokens of the same written word (see _part_of_word) that is not covered by a
    multiword token, e.g. the clitic in French "l'arbre", Welsh "Mae'r", a Korean
    particle, a hyphenated word like "90-hectare" or a Japanese long unit word.
    """
    ids_set = set(ids)
    if any(
        _inside_written_word(tokens, i, orthography) and (i in ids_set) != (i + 1 in ids_set)
        for i in range(len(tokens) - 1)
    ):
        return True
    # The subtree lands right before the pivot (moving left) or right after it.
    landing = pivot_idx - 1 if min(ids) > pivot_idx else pivot_idx
    return _inside_written_word(tokens, landing, orthography)


def _splits_multiword_token(base_tokens, ids, new_tokens) -> bool:
    """
    Whether moving the subtree at 0-based `ids` breaks up a multiword token (e.g.
    "zum" = "zu" + "dem"), which can then no longer be rendered as one form: either
    only some of its words are moved, or the subtree is inserted in between them.
    """

    def mwt(token):
        return (token["misc"] or {}).get("MWT")

    ids_set = set(ids)
    moved = {mwt(tok) for i, tok in enumerate(base_tokens) if i in ids_set}
    stayed = {mwt(tok) for i, tok in enumerate(base_tokens) if i not in ids_set}
    if (moved & stayed) - {None}:
        return True

    positions = {}
    for pos, tok in enumerate(new_tokens):
        if mwt(tok) is not None:
            positions.setdefault(mwt(tok), []).append(pos)
    return any(pos[-1] - pos[0] + 1 != len(pos) for pos in positions.values())


def create_all_constituent_swaps_tree(
    df,
    treebank,
    target: PredictionTarget,
    max_sen_len=100,
    min_sen_len=6,
    orthography=Orthography(),
):
    assert len(target.child_deprels) in (1, 2)

    result_rows = []

    for _, row in df.iterrows():
        if (row.sen_len > max_sen_len) or (row.sen_len < min_sen_len):
            continue

        tree = treebank[row.tree_idx]

        # "_" forms are placeholders (e.g. compound parts in Sanskrit-Vedic), so the
        # sentence is not real text. Words of a multiword token can have "_" as form
        # (e.g. Gujarati), but are rendered by the multiword form instead.
        if any(tok["form"] == "_" and "MWT" not in (tok["misc"] or {}) for tok in tree):
            continue

        dep_ids = {
            dep: get_subtree_indices(tree, int(row[f"{dep}_idx"]) - 1)
            for dep in target.child_deprels
        }
        h_ids = [int(row.head_idx) - 1]

        sen_str = tree_to_sentence(tree)
        tree_orders = {row.deprel_order: list(tree)}
        swap_logic = _build_swap_logic(target, dep_ids, h_ids)
        swap_candidates = set(row.swap_order_candidates)

        for swap_order, (base_sen_order, ids, pivot_idx) in swap_logic[
            row.deprel_order
        ].items():
            pivot_idx = int(pivot_idx)
            if base_sen_order not in tree_orders:
                # Built on an order that was skipped (stranded punctuation or a
                # split multiword token).
                continue
            base_tokens = tree_orders[base_sen_order]
            # Land next to the whole written word or lexical unit the pivot belongs
            # to (e.g. before "mark-up" rather than between "mark-" and "up").
            pivot_idx = _word_edge_pivot(base_tokens, ids, pivot_idx, orthography)

            # Where punctuation goes depends on the word order, so there is no
            # mechanical way to place it after a swap; such swaps are skipped.
            if _strands_punct(base_tokens, ids):
                continue
            # Words written together (e.g. clitics) can't be separated by a swap.
            if _splits_written_word(base_tokens, ids, pivot_idx, orthography):
                continue

            # The head/verb is the moved constituent (rather than a dependent) for
            # roughly half the 2-deprel swaps; flag these since the moved root is then
            # the head, not a child, which changes what attachment validation means.
            head_was_moved = ids == h_ids

            new_tokens = swap_tree_tokens(base_tokens, ids, pivot_idx, orthography)
            if new_tokens is None:
                break
            if _splits_multiword_token(base_tokens, ids, new_tokens):
                continue
            if _strands_punct_at_start(base_tokens, new_tokens, ids, orthography):
                continue

            tree_orders[swap_order] = new_tokens

            if swap_order in swap_candidates:
                new_metadata = dict(tree.metadata)
                new_metadata["text"] = tree_to_sentence(new_tokens)
                # Swapping two identical forms (e.g. reduplication) leaves the
                # sentence unchanged (up to casing), so it is not a minimal pair.
                if new_metadata["text"].lower() == sen_str.lower():
                    continue
                swapped_tree = TokenList(new_tokens, metadata=new_metadata)

                original_child_id = _moved_root_original_id(base_tokens, ids)
                swapped_child_id = _moved_root_new_id(base_tokens, ids, pivot_idx)

                result_row = row.to_dict()
                result_row["sen_str"] = sen_str
                result_row["original_order"] = row.deprel_order
                result_row["swap_order"] = swap_order
                result_row["swapped_sen_str"] = new_metadata["text"]
                result_row["original_child_id"] = original_child_id
                result_row["swapped_child_id"] = swapped_child_id
                result_row["head_was_moved"] = head_was_moved
                # Moving a subtree with a gap also closes the gap (e.g. hyperbaton).
                result_row["discontinuous"] = max(ids) - min(ids) + 1 != len(ids)
                result_row["original_head_id"] = tree[original_child_id - 1]["head"]
                result_row["swapped_head_id"] = swapped_tree[swapped_child_id - 1]["head"]
                result_row["original_tree"] = serialize_tree(tree)
                result_row["swapped_tree"] = serialize_tree(swapped_tree)
                result_rows.append(result_row)

    return pd.DataFrame(result_rows)


# The cascade of DT leaves from which swaps are created: each step is only used
# for a (lang, deprel) for which the previous steps yield no pairs at all. Each
# maps to the DT it uses (fitted without or with form/lemma features) and the
# entropy range (in bits, Jeffreys-smoothed) of its leaves' order distribution.
# Strict: at most ~1.3% exceptions in a large leaf; near-strict: ~1.3-3.1%.
ORDER_STRENGTHS = {
    "strict": ("non_lexical", (0.0, 0.1)),
    "near_strict": ("non_lexical", (0.1, 0.2)),
    "lexical_strict": ("lexical", (0.0, 0.1)),
}
NON_ORDER_ENTROPIES = {"leaf_full_entropy", "leaf_top1_entropy"}


def swap_candidates(dt_df, min_entropy, max_entropy) -> pd.Series:
    """
    Per row of dt_df, the orders it can be swapped to: those whose entropy
    against the row's own order within its DT leaf lies in [min_entropy,
    max_entropy). Only rows with their leaf's majority order (leaf_decision)
    are swapped.
    """
    orders = [
        col.removesuffix("_entropy")
        for col in dt_df.columns
        if col.endswith("_entropy") and col not in NON_ORDER_ENTROPIES
    ]

    def candidates(row):
        if not row.leaf_decision:
            return []
        return [
            order
            for order in orders
            if order != row.deprel_order
            and min_entropy <= row[f"{order}_entropy"] < max_entropy
        ]

    return dt_df.apply(candidates, axis=1)


def _balance_weights(df, balance_features):
    """
    Inverse-frequency sample weights, so that sampling flattens the distribution
    of each balance feature.  Uniform if balance_features is None.
    """
    weights = np.ones(len(df))
    for feat in balance_features or []:
        if feat in df.columns:
            counts = Counter(df[feat])
            weights /= [counts[value] / len(df) for value in df[feat]]
    return pd.Series(weights, index=df.index)


def _draw_sentences(pool, deficits, rng):
    """
    Draw deficits[stratum] sentences (without replacement) from each stratum of
    the pool, weighted by sample_weight.  The returned indices are in draw order.
    """
    drawn = []
    for stratum, group in pool.groupby("stratum"):
        n = min(deficits.get(stratum, 0), len(group))
        if n > 0:
            p = group.sample_weight / group.sample_weight.sum()
            drawn.extend(rng.choice(group.index, size=n, replace=False, p=p))
    return drawn


def create_pairs_tree(
    dt_df: pd.DataFrame,
    treebank,
    target: PredictionTarget,
    max_total_items: int = 1000,
    min_total_items: int = 10,
    max_sen_len: int = 100,
    min_sen_len: int = 6,
    balance_features: list[str] | None = None,
    balance_leaves: bool = False,
    classifier: JointClassifier | None = None,
    validate_threshold: float = 0.9,
    seed: int = 42,
    save_to_pairs_only: str | None = None,
    save_to: str | None = None,
    lexical_dt_df: pd.DataFrame | None = None,
):
    """
    Sample up to max_total_items minimal pairs from dt_df, stratified per leaf
    (unless balancing over balance_features without balance_leaves, in which case
    the whole df forms a single stratum).

    Pairs are first created from strict leaves of dt_df. Only if that yields fewer
    than min_total_items pairs (after every filter and the validation) are its
    near-strict leaves used instead, and if that falls short too, the strict leaves
    of lexical_dt_df (a DT fitted with form/lemma features), if given; see
    ORDER_STRENGTHS. The order_strength column records which. If no step yields
    min_total_items pairs, None is returned and nothing is saved.

    If a classifier is passed, every pair is validated with score_swap_df and
    rejected if its max_swapped_head_prob exceeds validate_threshold.  Sampling
    then proceeds in rounds: each round draws as many new sentences as a stratum
    is still short of pairs, and scores them in one batched pass.  This stops
    once every stratum is full or has run out of sentences.
    """
    orthography = detect_orthography(treebank)

    dt_dfs = {"non_lexical": dt_df, "lexical": lexical_dt_df}

    swap_df = None
    for order_strength, (dt_name, (min_entropy, max_entropy)) in ORDER_STRENGTHS.items():
        source_df = dt_dfs[dt_name]
        if source_df is None:
            continue
        candidates_df = source_df.assign(
            swap_order_candidates=swap_candidates(source_df, min_entropy, max_entropy)
        )
        swap_df = _sample_pairs(
            candidates_df,
            treebank,
            target,
            max_total_items=max_total_items,
            max_sen_len=max_sen_len,
            min_sen_len=min_sen_len,
            balance_features=balance_features,
            balance_leaves=balance_leaves,
            classifier=classifier,
            validate_threshold=validate_threshold,
            seed=seed,
            orthography=orthography,
        )
        if swap_df is not None and len(swap_df) >= min_total_items:
            swap_df["order_strength"] = order_strength
            break
        swap_df = None

    if swap_df is not None:
        if save_to_pairs_only is not None:
            pair_df = swap_df[["sen_str", "swapped_sen_str", "leaf_rule"]].copy()
            pair_df = pair_df.sort_values("leaf_rule")

            os.makedirs(os.path.dirname(save_to_pairs_only), exist_ok=True)
            pair_df.to_parquet(save_to_pairs_only, index=False)

        if save_to is not None:
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            swap_df.to_parquet(save_to, index=False)

    return swap_df


def _sample_pairs(
    dt_df: pd.DataFrame,
    treebank,
    target: PredictionTarget,
    max_total_items: int,
    max_sen_len: int,
    min_sen_len: int,
    balance_features: list[str] | None,
    balance_leaves: bool,
    classifier: JointClassifier | None,
    validate_threshold: float,
    seed: int,
    orthography: Orthography,
):
    """
    Rejection sampling of pairs from the rows of dt_df that have swap candidates
    (swap_order_candidates); see create_pairs_tree.
    """
    keep_df = dt_df[dt_df.swap_order_candidates.str.len() > 0].copy()
    keep_df["sen_len"] = [len(sen) for sen in keep_df.sen]
    keep_df = keep_df[keep_df.sen_len.between(min_sen_len, max_sen_len)]
    keep_df = keep_df.reset_index(drop=True)

    if len(keep_df) == 0:
        return None

    per_leaf = balance_features is None or balance_leaves
    keep_df["stratum"] = keep_df.leaf_id if per_leaf else 0
    keep_df["sample_weight"] = pd.concat(
        _balance_weights(group, balance_features)
        for _, group in keep_df.groupby("stratum")
    )

    strata = keep_df.stratum.unique()
    quota = max_total_items // len(strata)
    deficits = {stratum: quota for stratum in strata}

    rng = np.random.default_rng(seed)
    pool = keep_df
    accepted = []
    swap_df = pd.DataFrame()

    while any(deficit > 0 for deficit in deficits.values()):
        drawn_idx = _draw_sentences(pool, deficits, rng)
        if len(drawn_idx) == 0:
            break
        pool = pool.drop(drawn_idx)

        round_df = create_all_constituent_swaps_tree(
            keep_df.loc[drawn_idx],
            treebank,
            target,
            max_sen_len=max_sen_len,
            min_sen_len=min_sen_len,
            orthography=orthography,
        )
        if len(round_df) > 0 and classifier is not None:
            round_df = score_swap_df(round_df, classifier)
            # head_was_moved rows have no score (None) and are kept.
            round_df = round_df[~(round_df.max_swapped_head_prob > validate_threshold)]
        if len(round_df) == 0:
            continue
        accepted.append(round_df)

        # Pairs are in draw order, so trimming keeps the earliest-drawn ones.
        # Sentences that occur more than once in a treebank would yield the same
        # pair again; only the first is kept.
        swap_df = pd.concat(accepted, ignore_index=True)
        swap_df = swap_df.drop_duplicates(["sen_str", "swapped_sen_str"])
        swap_df = swap_df.groupby("stratum").head(quota)
        counts = swap_df.stratum.value_counts()
        deficits = {stratum: quota - counts.get(stratum, 0) for stratum in strata}

    swap_df = swap_df.drop(columns=["stratum", "sample_weight"], errors="ignore")
    return swap_df.reset_index(drop=True)
