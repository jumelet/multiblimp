# NUMBER
swap_number_subj = (
    ["Number[subj]", "Number"],
    {
        "SG": "PL",
        "PL": "SG",
    },
)
swap_number_subj_any = (["Number[subj]", "Number"], None)
swap_number = (
    "Number",
    {
        "SG": "PL",
        "PL": "SG",
    },
)

# PERSON
swap_person = (
    "Person",
    {
        "1": "3",
        "3": "1",
    }
)
swap_any_person = ("Person", None)

# CASE
swap_case = (
    "Case",
    {
        "NOM": "ACC",
        "ACC": "NOM",
    },
)

# GENDER
swap_gender_any = ("Gender", None)
swap_gender_f2m = (
    "Gender",
    {
        "MASC": "FEM",
        "FEM": "MASC",
    },
)
swap_gender_fm2n = (
    "Gender",
    {
        "MASC": "NEUT",
        "FEM": "NEUT",
        "COM": "NEUT",
        "NEUT": "COM",
    },
)
