import pandas as pd
from nltk.util import ngrams

FAMILIES = [
    "fatty acid", "alkaloid", "terpenoid", "steroid",
    "glycoside", "nucleotide", "peptide", "amino acid",
    "lipid", "polyketide", "carbohydrate"
]

STATE_KEYWORDS = [
    "acid", "ester", "anion", "dianion", "coenzyme a",
    "phosphate", "sulfate", "protonated"
]

def lexical_overlap(a, b):
    a_set = set(a.split())
    b_set = set(b.split())
    return len(a_set & b_set) / max(1, len(a_set | b_set))

def detect_family(text):
    return {f for f in FAMILIES if f in text}

def detect_states(text):
    return {s for s in STATE_KEYWORDS if s in text}

def classify(pred, gt):
    p = pred.lower()
    g = gt.lower()

    fam_p = detect_family(p)
    fam_g = detect_family(g)

    state_p = detect_states(p)
    state_g = detect_states(g)

    overlap = lexical_overlap(p, g)

    # Type A
    if fam_p and fam_g and fam_p.isdisjoint(fam_g):
        return "A"

    # Type C
    if fam_p & fam_g and state_p != state_g:
        return "C"

    # Type E
    if overlap > 0.8:
        return "E"

    # Type B
    if fam_p & fam_g:
        return "B"

    # Type D
    return "D"


# === Charger et annoter ===
df = pd.read_csv("pred_v_res.csv")
df["error_type"] = df.apply(lambda r: classify(r["prediction"], r["ground_truth"]), axis=1)

df.to_csv("pred_v_res_annotated.csv", index=False)

df["error_type"].value_counts(normalize=True)