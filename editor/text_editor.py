from utils.data_utils import x_map

X_KEYS = list(x_map.keys())
FORMAL_CHARGE_IDX = X_KEYS.index("formal_charge")

def graph_consistency_fix(text: str, data) -> str:
    # 1. Extraire les charges atomiques
    formal_charges = data.x[:, FORMAL_CHARGE_IDX].cpu().tolist()

    has_pos = any(fc > 0 for fc in formal_charges)
    has_neg = any(fc < 0 for fc in formal_charges)
    net_charge = sum(formal_charges)

    new_text = text

    # 2. Conjugate acid / base
    if "conjugate acid" in text or "conjugate base" in text:
        if net_charge < 0:
            new_text = new_text.replace("conjugate acid", "conjugate base")
        elif net_charge > 0:
            new_text = new_text.replace("conjugate base", "conjugate acid")

    # 3. Anion / cation / zwitterion
    if has_pos and has_neg:
        new_text = replace_species(new_text, target="zwitterion")
    elif has_neg:
        new_text = replace_species(new_text, target="anion")
    elif has_pos:
        new_text = replace_species(new_text, target="cation")

    return new_text

def replace_species(text, target):
    species = ["anion", "cation", "zwitterion"]
    for s in species:
        if s in text and s != target:
            return text.replace(s, target)
    return text

import re

BIO_PATTERNS = [
    r"has a role as .*?(?:\.|$)",
    r"acts as .*?(?:\.|$)",
    r"is involved in .*?(?:\.|$)",
    r"is found in .*?(?:\.|$)",
]

def graph_consistency_fix_v2(text: str, data) -> str:
    original_text = text
    new_text = text

    # --------------------------------------------------
    # 1) Charge-based correction (TON code)
    # --------------------------------------------------
    formal_charges = data.x[:, FORMAL_CHARGE_IDX].cpu().tolist()

    has_pos = any(fc > 0 for fc in formal_charges)
    has_neg = any(fc < 0 for fc in formal_charges)
    net_charge = sum(formal_charges)

    if "conjugate acid" in new_text or "conjugate base" in new_text:
        if net_charge < 0:
            new_text = new_text.replace("conjugate acid", "conjugate base")
        elif net_charge > 0:
            new_text = new_text.replace("conjugate base", "conjugate acid")

    if has_pos and has_neg:
        new_text = replace_species(new_text, target="zwitterion")
    elif has_neg:
        new_text = replace_species(new_text, target="anion")
    elif has_pos:
        new_text = replace_species(new_text, target="cation")

    # --------------------------------------------------
    # 2) Remove unreliable biological roles
    # --------------------------------------------------
    for pat in BIO_PATTERNS:
        new_text = re.sub(pat, "", new_text, flags=re.IGNORECASE)

    # --------------------------------------------------
    # 3) Clean spacing / punctuation
    # --------------------------------------------------
    new_text = re.sub(r"\s+", " ", new_text)
    new_text = re.sub(r"\s+\.", ".", new_text).strip()

    # --------------------------------------------------
    # 4) Safety fallback
    # --------------------------------------------------
    if len(new_text.split()) < 6:
        return original_text

    return new_text


def graph_consistency_fix_adaptive(text: str, data) -> str:
    """
    Adaptive editor:
    - short / safe sentences -> v1
    - long or risky sentences -> v2
    """

    LENGTH_THRESHOLD = 85

    RISK_PATTERNS = [
        "has a role as",
        "acts as",
        "is involved in",
        "is found in",
    ]

    text_lower = text.lower()
    word_len = len(text.split())

    force_v2 = word_len >= LENGTH_THRESHOLD
    #force_v2 |= any(pat in text_lower for pat in RISK_PATTERNS)

    if force_v2:
        return graph_consistency_fix_v2(text, data)
    else:
        return graph_consistency_fix(text, data)  # v1 originale
    
LEXICAL_NORMALIZATION = {
    # verbes
    "acts as": "has a role as",
    "functions as": "has a role as",
    "serves as": "has a role as",

    # prépositions
    "involved in": "has a role in",
    "plays a role in": "has a role in",

    # déterminants
    "one of the": "a",

    # classes vagues
    "chemical compound": "chemical entity",
    "compound": "chemical entity",

    # ponctuation / liaison
    " ,": ",",
}

def lexical_normalize(text: str) -> str:
    out = text
    for src, tgt in LEXICAL_NORMALIZATION.items():
        out = out.replace(src, tgt)
    return out


def graph_consistency_fix_v3(text: str, data) -> str:
    # v1 = ton editor actuel (charges, acid/base, etc.)
    text_v1 = graph_consistency_fix(text, data)

    # normalisation lexicale (BLEU-friendly)
    text_norm = lexical_normalize(text_v1)

    return text_norm
