import re

DIGIT_RE = re.compile(r"\d")
ION_CHARGE_RE = re.compile(r"\(\d*[+-]\)")   # (+), (-), (2+), (3-)

# Matches EC-like or dot-separated identifiers, optionally ending with .* or *
# Examples: 3.1.3, 3.1.3.*, 1.2.3.4, 1.2.3.*
DOT_ID_RE = re.compile(r"^\d+(?:\.\d+)+(?:\.\*)?\*?$")

TRAILING_PUNCT = {".", ";", ":"}
SYMBOL_ONLY = {"*", "+", "="}

STEREO_KEEP = {
    "(R)-configuration",
    "(S)-configuration",
    "(R)-enantiomer",
    "(S)-enantiomer",
}

def clean_token(tok: str) -> str:
    """Strip useless parentheses for linguistic tokens, keep for structural ones."""

    # remove square brackets everywhere (no semantic value)
    tok = tok.replace("[", "").replace("]", "")

    if tok.startswith("(") or tok.endswith(")"):
        core = tok.strip("()")
        if DIGIT_RE.search(core) or "-" in core or "->" in core:
            return tok
        # keep stereo generic tokens as-is
        if tok in STEREO_KEEP:
            return tok
        return core
    return tok

def split_trailing_punct(tok: str):
    """
    Split terminal punctuation only (.,;:) at END of token.
    Do NOT split dot-notation identifiers like 3.1.3.*.
    """
    if not tok:
        return []

    last = tok[-1]
    if last in TRAILING_PUNCT and tok not in TRAILING_PUNCT:
        base = tok[:-1]
        # If base is a dot-notation identifier, keep punctuation attached (rare)
        # (Most dot IDs don't end with '.', but this is safe.)
        if DOT_ID_RE.fullmatch(base):
            return [tok]
        return [base, last]

    return [tok]

def postprocess_tokens(tokens):
    """Merge ionic charges and wildcard '*' with the previous token."""
    merged = []
    for tok in tokens:
        if ION_CHARGE_RE.fullmatch(tok) and merged:
            merged[-1] = merged[-1] + tok
        elif tok == "*" and merged:
            merged[-1] = merged[-1] + "*"
        else:
            merged.append(tok)
    return merged

def tokenize(text: str):
    assert isinstance(text, str)

    # 1) Separate ONLY linguistic commas
    text = text.replace(", ", " , ")

    # 2) Split on spaces (no global '.' splitting!)
    raw_tokens = text.split()

    # 3) Split trailing punctuation per-token + clean parentheses
    tokens = []
    for rt in raw_tokens:
        for t in split_trailing_punct(rt):
            t = clean_token(t)
            if t:
                tokens.append(t)

    # 4) Merge (+)/(-) and '*' when they become standalone tokens
    tokens = postprocess_tokens(tokens)

    return tokens
