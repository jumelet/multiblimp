from dataclasses import dataclass


# deprel, head_pos, child_pos
ALL_NOUN_DEPRELS = [
    ("amod", "NOUN", "ADJ"),
    ("det", "NOUN", "DET"),
    ("nummod", "NOUN", "NUM"),
    ("nmod", "NOUN", "NOUN"),
    ("case", "NOUN", "ADP"),
    ("cop", ["NOUN", "ADJ"], "AUX"),
    ("acl", "NOUN", "VERB")
]

ALL_VERB_DEPRELS = [
    ("nsubj", "VERB", ["NOUN", "PRON"]),
    ("obj", "VERB", ["NOUN", "PRON"]),
    ("iobj", "VERB", ["NOUN", "PRON"]),
    ("csubj", "VERB", "VERB"),
    ("ccomp", "VERB", "VERB"),
    ("xcomp", "VERB", "VERB"),
    ("obl", "VERB", ["NOUN", "PRON"]),
    ("advcl", "VERB", ["NOUN", "VERB"]),
    ("advmod", "VERB", "ADV"),
    ("aux", "VERB", "AUX"),
]

ALL_DEPRELS = ALL_NOUN_DEPRELS + ALL_VERB_DEPRELS


@dataclass
class PredictionTarget:
    """
    Defines what word order pattern to predict.

    Examples:
        PredictionTarget(child_deprels=["amod"])
        PredictionTarget(child_deprels=["nsubj", "obj"], head_pos="VERB", head_deprel="root")
    """

    child_deprels: list[str]

    # Optional: filter by head's deprel (e.g., only root verbs for SVO)
    head_deprel: str | None = None

    # Optional: filter by head's POS
    head_pos: list[str] | None = None

    # Optional: filter by child's POS (either all children same POS or mapping deprel->POS)
    child_pos: list[str] | dict[str, list[str]] | None = None

    head_feats: dict | None = None

    # If True, child deprels are matched on their base deprel (e.g. acl:relcl matches acl)
    include_subtypes: bool = True

    def __post_init__(self):
        if not self.child_deprels:
            raise ValueError("child_deprels must be non-empty")

    def match_deprel(self, deprel: str) -> str:
        """Return the deprel used for matching against child_deprels."""
        if self.include_subtypes:
            return deprel.split(":", 1)[0]
        return deprel


def get_all_deprel_targets():
    for deprel, head_pos, child_pos in ALL_DEPRELS:
        head_pos = [head_pos] if isinstance(head_pos, str) else head_pos
        child_pos = [child_pos] if isinstance(child_pos, str) else child_pos

        target = PredictionTarget(
            child_deprels=[deprel],
            head_pos=head_pos,
            child_pos=child_pos,
        )

        yield target


amod_target = PredictionTarget(
    child_deprels=["amod"],
    head_pos=["NOUN"],
    child_pos=["ADJ"],
)

core_arg_target = PredictionTarget(
    child_deprels=["nsubj", "obj"],
    head_pos=["VERB"],
    head_deprel="root",
)

iobj_target = PredictionTarget(
    child_deprels=["iobj"],
    head_pos=["VERB"],
    head_deprel="root",
)

dnan_target = PredictionTarget(
    child_deprels=["det", "nummod", "amod"],
    head_pos=["NOUN"],
    child_pos={"det": ["DET"], "nummod": ["NUM"], "amod": ["ADJ"]},
)

acl_relcl_target = PredictionTarget(
    child_deprels=["acl:relcl"],
    head_pos=["NOUN"],
    child_pos=["VERB"],
    include_subtypes=False,
)

obl_target = PredictionTarget(
    child_deprels=["obl"],
    head_pos=["VERB"],
    child_pos=["NOUN", "PROPN", "PRON"],
)

obl_noun_target = PredictionTarget(
    child_deprels=["obl"],
    head_pos=["VERB"],
    child_pos=["NOUN", "PROPN", "PRON"],
)

case_target = PredictionTarget(
    child_deprels=["case"],
    head_pos=["NOUN"],
    child_pos=["ADP"],
)

advmod_target = PredictionTarget(
    child_deprels=["advmod"],
    head_pos=["VERB"],
    child_pos=["ADV"],
)

nmod_noun_target = PredictionTarget(
    child_deprels=["nmod"],
    head_pos=["NOUN"],
    child_pos=["NOUN"],
)

advcl_noun_verb_target = PredictionTarget(
    child_deprels=["advcl"],
    head_pos=["VERB"],
    child_pos=["NOUN"],
)
