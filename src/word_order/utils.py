ALL_CORE_ARGS = ["svo", "sov", "vso", "vos", "osv", "ovs"]


def capitalize_first(word: str) -> str:
    if len(word) == 0:
        return ""
    return word[0].upper() + word[1:]
