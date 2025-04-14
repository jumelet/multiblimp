def load_um_features():
    # https://huggingface.co/datasets/unimorph/universal_morphologies/blob/main/universal_morphologies.py
    feat2val_UM = {
        "Aktionsart": [
            "STAT",
            "DYN",
            "TEL",
            "ATEL",
            "PCT",
            "DUR",
            "ACH",
            "ACCMP",
            "SEMEL",
            "ACTY",
        ],
        "Animacy": ["ANIM", "INAN", "HUM", "NHUM"],
        "Argument_Marking": [
            "ARGNO1S",
            "ARGNO2S",
            "ARGNO3S",
            "ARGNO1P",
            "ARGNO2P",
            "ARGNO3P",
            "ARGAC1S",
            "ARGAC2S",
            "ARGAC3S",
            "ARGAC1P",
            "ARGAC2P",
            "ARGAC3P",
            "ARGAB1S",
            "ARGAB2S",
            "ARGAB3S",
            "ARGAB1P",
            "ARGAB2P",
            "ARGAB3P",
            "ARGER1S",
            "ARGER2S",
            "ARGER3S",
            "ARGER1P",
            "ARGER2P",
            "ARGER3P",
            "ARGDA1S",
            "ARGDA2S",
            "ARGDA3S",
            "ARGDA1P",
            "ARGDA2P",
            "ARGDA3P",
            "ARGBE1S",
            "ARGBE2S",
            "ARGBE3S",
            "ARGBE1P",
            "ARGBE2P",
            "ARGBE3P",
            "ARGABS1",  # All Basque
            "ARGABS2",
            "ARGABS3",
            "ARGABSPL",
            "ARGABSSG",
            "ARGABSINFM",
            "ARGABSMASC",
            "ARGABSFEM",
            "ARGIO1",
            "ARGIO2",
            "ARGIO3",
            "ARGIOPL",
            "ARGIOSG",
            "ARGIOINFM",
            "ARGIOMASC",
            "ARGIOFEM",
            "ARGERG1",
            "ARGERG2",
            "ARGERG3",
            "ARGERGPL",
            "ARGERGSG",
            "ARGERGINFM",
            "ARGERGMASC",
            "ARGERGFEM",
            "ARBEB1S",  # Chukchi
            "ARBEB1P",
            "ARBAB1S",
            "ARBAB1P",
            "ARBAB2S",
            "ARBAB2P",
            "ARBAB3S",
            "ARBAB3P",
        ],
        "Aspect": [
            "IPFV",
            "PFV",
            "PRF",
            "PROG",
            "PROSP",
            "ITER",
            "HAB",
            "INCH",
            "Imp,Perf",
            "FREQ",  # Lithuanian
            "RAPID",  # Turkish
        ],
        "Case": [
            "NOM",
            "ACC",
            "ERG",
            "ABS",
            "NOMS",
            "DAT",
            "BEN",
            "PRP",
            "GEN",
            "REL",
            "PRT",
            "INS",
            "COM",
            "VOC",
            "COMPV",
            "EQTV",
            "PRIV",
            "PROPR",
            "AVR",
            "FRML",
            "TRANS",
            "BYWAY",
            "INTER",
            "AT",
            "POST",
            "IN",
            "CIRC",
            "ANTE",
            "APUD",
            "ON",
            "ONHR",
            "ONVR",
            "SUB",
            "REM",
            "PROXM",
            "ESS",
            "ALL",
            "ABL",
            "APPRX",
            "TERM",
            "LOC",
            "INST",
            "ILL",  # finnish
            "LAT",  # finnish
            "INE",  # finnish
            "TRA",  # finnish
            "PAR",  # finnish
            "ADE",  # finnish
            "ELA",  # finnish
            "ABE",  # finnish
            "(non)NOM",  # Pashto
            "EQU",  # Tatar
            "PROL",  # Livvi
            "Ter",  # Livvi
            "OBL",  # Gujarati
            "non{NOM}",  # Old French
            "not{NOM}",  # Old French
            "TER",
        ],
        "Degree": [
            "CMPR",
            "SPRL",
            "AB",
            "RL",
            "EQT",
            "BASE",  # aka "Pos" added for the normal adjective case
            "SUP",  # Afrikaans
            "CMP",
            "DIM",
        ],
        "Definite": ["DEF", "INDF", "SPEC", "NSPEC", "CONS"],  # Arabic
        "Deixis": [
            "PROX",
            "MED",
            "REMT",
            "REF1",
            "REF2",
            "NOREF",
            "PHOR",
            "VIS",
            "NVIS",
            "ABV",
            "EVEN",
            "BEL",
        ],
        "Evidentiality": [
            "FH",
            "DRCT",
            "SEN",
            "VISU",
            "NVSEN",
            "AUD",
            "NFH",
            "QUOT",
            "RPRT",
            "HRSY",
            "INFER",
            "ASSUM",
            "NFH2",
        ],
        "Finiteness": ["NFIN"],
        "Gender": [
            "MASC",
            "FEM",
            "NEUT",
            "NAKH1",
            "NAKH2",
            "NAKH3",
            "NAKH4",
            "NAKH5",
            "NAKH6",
            "NAKH7",
            "NAKH8",
            "BANTU1",
            "BANTU2",
            "BANTU3",
            "BANTU4",
            "BANTU5",
            "BANTU6",
            "BANTU7",
            "BANTU8",
            "BANTU9",
            "BANTU10",
            "BANTU11",
            "BANTU12",
            "BANTU13",
            "BANTU14",
            "BANTU15",
            "BANTU16",
            "BANTU17",
            "BANTU18",
            "BANTU19",
            "BANTU20",
            "BANTU21",
            "BANTU22",
            "BANTU23",
            "MASV",  # Irish
            "NEU",  # Sanskrit
            "{MASC/FEM/NEU}",  # Sanskrit
        ],
        "Information_Structure": ["TOP", "FOC"],
        "Interrogativity": ["DECL", "INT"],
        "Language_Specific": [
            "LGSPEC1",
            "LGSPEC2",
            "LGSPEC3",
            "LGSPEC4",
            "LGSPEC5",
            "LGSPEC6",
            "LGSPEC7",
            "LGSPEC8",
            "LGSPEC9",
            "LGSPEC10",
            "LGSPEC11",  # Old French
            "LGSPEC12",  # Old French
            "LGSPEC",
            "LGSPEC01",  # Eastern Armenian
            "LGSPEC02",  # Eastern Armenian
            "LGSPEC03",  # Eastern Armenian
            "LGSPEC04",  # Eastern Armenian
        ],
        "Mood": [
            "IND",
            "SBJV",
            "REAL",
            "IRR",
            "AUPRP",
            "AUNPRP",
            "IMP",
            "COND",
            "PURP",
            "INTEN",
            "POT",
            "LKLY",
            "ADM",
            "OBLIG",
            "DEB",
            "PERM",
            "DED",
            "SIM",
            "OPT",
            "QOT",
            "NEC",
            "JUS",  # Arabic
            "SUBJ",  # Portuguese
            "INFR",  # Turkish
            "GENPOT",  # Turkish
            "GENNEC",  # Turkish
            "CndGen",  # Turkish
            "CndPot",  # Turkish
            "Des",  # Turkish
            "DesPot",  # Turkish
            "CndGenPot",  # Turkish
            "GenPotPot",  # Turkish
            "NecPot",  # Turkish
            "HYP",  # Basque
            "GenNecPot",
            "CND,IND",
            "CND2",
        ],
        "Number": [
            "SG",
            "PL",
            "GRPL",
            "DU",
            "TRI",
            "PAUC",
            "GRPAUC",
            "INVN",
            "Coll",  # Arabic
            "Assoc",  # Armenian
            "Ptan",  # Latvian
            "Count",  # Macedonian
            "SG ADP",
            "PL ADP",  # German
            "CARD",
        ],
        "upos": [
            "N",
            "PROPN",
            "ADJ",
            "PRO",
            "CLF",
            "ART",
            "DET",
            "V",
            "ADV",
            "AUX",
            "ADP",
            "COMP",
            "CONJ",
            "NUM",
            "PART",
            "INTJ",
            "AJD",  # classical armenian
            "PRE",  # Livvi
            "ADJ.CVB",  # Korean
            "PRON",
        ],
        "VerbForm": [
            "V.PTCP",
            "V.PCTP",
            "V.MSDR",
            "V.CVB",
            "PTCP",
            "GER",
            "INF",
            "FIN",
            "CONV",
            "GDV",
            "VNOUN",
            "COP",
            "FINREL",
        ],
        "Subcat": [
            "CVB",
            "MSDR",
            "INTENS",  # Sanskrit
            "INJ",  # Sanskrit
            "DESID",  # Sanskrit
            "Tran",  # Afrikaans
            "Prep",  # Afrikaans
            "Indir",  # Georgian
        ],
        "noun_pos": [
            "NDEF",
            "NDEM",
            "NDEFDEM",
            "IIC",  # Sanskrit
            "IIV",  # Sanskrit
            "IIP",  # Sanskrit
        ],
        "NounType": [  # Irish https://universaldependencies.org/ga/feat/NounType.html
            "WEAK",
            "STRONG",
            "NOTSLENDER",
            "SLENDER",
        ],
        "Person": [
            "0",
            "1",
            "2",
            "3",
            "4",
            "INCL",
            "EXCL",
            "PRX",
            "OBV",
        ],
        "Polarity": ["POS", "NEG"],
        "Politeness": [
            "INFM",
            "FORM",
            "ELEV",
            "HUMB",
            "POL",
            "AVOID",
            "LOW",
            "HIGH",
            "STELEV",
            "STSUPR",
            "LIT",
            "FOREG",
            "COL",
        ],
        "Possession": [
            "ALN",
            "NALN",
            "PSS1S",
            "PSS2S",
            "PSS2SF",
            "PSS2SM",
            "PSS2SINFM",
            "PSS2SFORM",
            "PSS3S",
            "PSS3SF",
            "PSS3SM",
            "PSS1D",
            "PSS1DI",
            "PSS1DE",
            "PSS2D",
            "PSS2DM",
            "PSS2DF",
            "PSS3D",
            "PSS3DF",
            "PSS3DM",
            "PSS1P",
            "PSS1PI",
            "PSS1PE",
            "PSS2P",
            "PSS2PF",
            "PSS2PM",
            "PSS3PF",
            "PSS3PM",
            "PSS3",
            "PSS3P",
            "PSSD",
            "PSS4S",  # Hungarian
            "PSS4P",  # Hungarian
            "PSS",  # Hungarian
        ],
        "Switch_Reference": [
            "SS",
            "SSADV",
            "DS",
            "DSADV",
            "OR",
            "SIMMA",
            "SEQMA",
            "LOG",
        ],
        "Tense": [
            "PRS",
            "PST",
            "FUT",
            "IMMED",
            "HOD",
            "1DAY",
            "RCT",
            "RMT",
            "PQP",
            "PastPerf",
            "NPST",
            "NearPast",
            "PastResultI",  # Yakut/Sakha
            "PST:ELEV",
            "FUT:ELEV",  # Korean
        ],
        "Valency": ["IMPRS", "INTR", "TR", "DITR", "REFL", "RECP", "CAUS", "APPL"],
        "Voice": [
            "ACT",
            "MID",
            "PASS",
            "ANTIP",
            "DIR",
            "INV",
            "AGFOC",
            "PFOC",
            "LFOC",
            "BFOC",
            "ACFOC",
            "IFOC",
            "CFOC",
            "AUTO",  # Irish
            "CAU",
            "CAUCAU",
            "CAURCP",
            "CAUPASS",  # Classical Armenian
            "RCP",  # Xibe
        ],
        "nan": ["nan", "NaN", "-"],
    }

    val2feat_UM = {}
    for ufeat, vals in feat2val_UM.items():
        for val in vals:
            assert val not in val2feat_UM, val
            val2feat_UM[val] = ufeat

    return feat2val_UM, val2feat_UM
