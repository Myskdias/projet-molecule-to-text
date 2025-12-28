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