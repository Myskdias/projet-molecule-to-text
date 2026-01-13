import re
import random
from typing import List, Tuple, Dict

# --- 1) Règles "chimiquement sémantiques" (counterfactuals plausibles) ---
# Important: ordonner du +spécifique au +général (regex).
CHEM_RULES: List[Tuple[str, List[str]]] = [
    # Acid/base & conjugates
    (r"\bconjugate base\b", ["conjugate acid"]),
    (r"\bconjugate acid\b", ["conjugate base"]),
    (r"\bacidic\b", ["basic", "neutral"]),
    (r"\bbasic\b", ["acidic", "neutral"]),
    (r"\bacid\b", ["base"]),
    (r"\bbase\b", ["acid"]),

    # Charges / ions
    (r"\bcation\b", ["anion", "zwitterion"]),
    (r"\banion\b", ["cation", "zwitterion"]),
    (r"\bzwitterion\b", ["cation", "anion"]),
    (r"\bpositively charged\b", ["negatively charged", "neutral"]),
    (r"\bnegatively charged\b", ["positively charged", "neutral"]),
    (r"\bneutral\b", ["charged"]),

    # Saturation / aromaticity
    (r"\bunsaturated\b", ["saturated"]),
    (r"\bsaturated\b", ["unsaturated"]),
    (r"\baromatic\b", ["aliphatic"]),
    (r"\baliphatic\b", ["aromatic"]),

    # Polarity / solubility
    (r"\bpolar\b", ["non-polar"]),
    (r"\bnon-polar\b", ["polar"]),
    (r"\bhydrophilic\b", ["hydrophobic"]),
    (r"\bhydrophobic\b", ["hydrophilic"]),
    (r"\bwater-soluble\b", ["poorly soluble in water", "insoluble in water"]),
    (r"\binsoluble\b", ["soluble"]),

    # Bioactivity phrasing (souvent présent dans tes desc)
    (r"\binhibitor\b", ["activator", "antagonist", "agonist"]),
    (r"\bagonist\b", ["antagonist"]),
    (r"\bantagonist\b", ["agonist"]),
]

# Clauses qu'on peut retirer / permuter sans casser le "format"
# (on reste proche du style dataset)
CLAUSE_STARTERS = [
    "It is ", "It has ", "It contains ", "It exhibits ", "It acts as ", "It results from ",
    "It is obtained from ", "It is isolated from ", "It is a conjugate "
]

def _fix_a_an(text: str) -> str:
    # Fix simple "a/an" after edits (approx, mais utile)
    text = re.sub(r"\ba ([aeiouAEIOU])", r"an \1", text)
    text = re.sub(r"\ban ([^aeiouAEIOU\W])", r"a \1", text)
    return text

def _apply_one_rule(desc: str, rng: random.Random) -> Tuple[str, bool]:
    candidates = []
    for pat, repls in CHEM_RULES:
        if re.search(pat, desc, flags=re.IGNORECASE):
            candidates.append((pat, repls))
    if not candidates:
        return desc, False

    pat, repls = rng.choice(candidates)
    # pick replacement with matching case roughly
    m = re.search(pat, desc, flags=re.IGNORECASE)
    if not m:
        return desc, False
    old = m.group(0)
    new = rng.choice(repls)

    # Preserve capitalization if the matched token starts uppercase
    if old[:1].isupper():
        new = new[:1].upper() + new[1:]

    new_desc = re.sub(pat, new, desc, count=1, flags=re.IGNORECASE)
    new_desc = _fix_a_an(new_desc)
    return new_desc, (new_desc != desc)

def _split_sentences(desc: str) -> List[str]:
    # Split conservatively on ". "
    parts = [p.strip() for p in desc.split(". ")]
    # Restore dots (except last if already)
    sents = []
    for i, p in enumerate(parts):
        if not p:
            continue
        if i < len(parts) - 1 and not p.endswith("."):
            p += "."
        sents.append(p)
    return sents

def _light_clause_ops(desc: str, rng: random.Random) -> str:
    sents = _split_sentences(desc)
    if len(sents) < 2:
        return desc

    op = rng.choice(["drop", "swap", "noop"])
    if op == "drop":
        # Drop one non-first sentence (keep "The molecule is ..." intact)
        idxs = [i for i in range(1, len(sents)) if any(sents[i].startswith(st) for st in CLAUSE_STARTERS)]
        if idxs:
            i = rng.choice(idxs)
            sents.pop(i)
    elif op == "swap" and len(sents) >= 3:
        # Swap two mid sentences (not first)
        i, j = rng.sample(range(1, len(sents)), 2)
        sents[i], sents[j] = sents[j], sents[i]

    out = " ".join(sents)
    return out

def generate_variants(
    desc: str,
    n: int = 30,
    seed: int = 0,
    edits_per_variant: Tuple[int, int] = (1, 3),
    clause_ops_prob: float = 0.35,
    max_tries: int = 500
) -> List[str]:
    """
    Génère n variantes en gardant le style, via:
    - 1 à 3 substitutions sémantiques (CHEM_RULES)
    - + optionnel: drop/swap de clauses ("It is ...", etc.)
    """
    rng = random.Random(seed)
    base = desc.strip()
    variants = set([base])

    tries = 0
    while len(variants) < n + 1 and tries < max_tries:
        tries += 1
        v = base

        # Apply k edits
        k = rng.randint(edits_per_variant[0], edits_per_variant[1])
        changed = False
        for _ in range(k):
            v2, did = _apply_one_rule(v, rng)
            v = v2
            changed = changed or did

        # Light clause operations sometimes
        if rng.random() < clause_ops_prob:
            v = _light_clause_ops(v, rng)

        v = v.strip()
        if not v.startswith("The molecule is"):
            # keep the canonical opening (very common in dataset)
            # If your dataset also has "The molecule is the ..." keep it.
            pass

        # Basic sanity filters
        if len(v) < 40 or len(v) > 1000:
            continue
        if not changed:
            continue

        variants.add(v)

    # Return without the original first (tu peux choisir)
    out = [base] + [x for x in variants if x != base]
    return out[: n + 1]

# --- Example ---
if __name__ == "__main__":
    example = "The molecule is the alpha-amino-acid cation formed from L-lysine by protonation. It is a conjugate base of a carboxynorspermidine. It exhibits acidic properties."
    vars_ = generate_variants(example, n=25, seed=42)
    for i, t in enumerate(vars_[:10]):
        print(i, t)
