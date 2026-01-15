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
    """
    Clean tokens conservatively.

    - Remove square brackets everywhere (kept from your original rule).
    - Only strip parentheses if the token is fully parenthesized: "(...)".
    - Never strip if content looks chemical/structural:
        digits, charge signs, arrows, hyphens, commas, stereochem.
    """
    tok = tok.replace("[", "").replace("]", "")

    # keep stereo generic tokens as-is
    if tok in STEREO_KEEP:
        return tok

    # Only consider stripping when token is fully wrapped "( ... )"
    if tok.startswith("(") and tok.endswith(")") and len(tok) >= 3:
        core = tok[1:-1]  # no strip() to avoid removing meaningful parentheses pairs

        # If looks structural/chemical, keep as is
        # digits, hyphen, arrows, comma, plus/minus charges, slash are typical chemical patterns
        if (
            DIGIT_RE.search(core)
            or "-" in core
            or "->" in core
            or "," in core
            or "+" in core
            or "/" in core
        ):
            return tok

        # If core contains spaces, it's likely syntactic, but at this stage
        # syntactic parentheses should already be separated by separate_syntactic_parentheses().
        # We keep this as a fallback.
        return core

    # If token has only one-sided parenthesis, DO NOT strip here.
    # Those cases come from tokenization boundaries and must be handled upstream.
    return tok

def clean_numeric_token(tok: str) -> str:
    """
    Clean numeric tokens polluted by punctuation,
    BUT preserve chemically meaningful parentheses.

    Rule:
    - If parentheses are balanced, do nothing.
    - If unbalanced, strip only at the borders.
    """
    if not any(c.isdigit() for c in tok):
        return tok

    n_open = tok.count("(")
    n_close = tok.count(")")

    if n_open == n_close:
        # balanced → keep as is
        return tok

    # unbalanced → safe border cleaning
    return tok.strip("()'\".,;:")

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

PAREN_BLOCK_RE = re.compile(r"\([^()]*\)")

def separate_syntactic_parentheses(text: str) -> str:
    """
    Separate syntactic parentheses using balanced parsing.
    Keep chemical parentheses intact.
    """

    out = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] != "(":
            out.append(text[i])
            i += 1
            continue

        # try to find matching ')'
        depth = 1
        j = i + 1
        while j < n and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1

        if depth != 0:
            # unbalanced, keep as-is
            out.append(text[i])
            i += 1
            continue

        block = text[i:j]        # "(...)"
        content = block[1:-1]

        prev_char = text[i - 1] if i > 0 else ""
        next_char = text[j] if j < n else ""

        has_space_inside = " " in content
        has_space_before = prev_char.isspace()
        has_space_after = next_char.isspace()

        # syntactic parentheses
        if has_space_inside and has_space_before and has_space_after:
            out.append(" ( ")
            out.append(content)
            out.append(" ) ")
        else:
            # chemical / lexical
            out.append(block)

        i = j

    return "".join(out)


CHEM_PAREN_RE = re.compile(r"[A-Za-z]+-\([^()]+\)")

def protect_chemical_parens(text):
    protected = {}
    idx = 0

    def repl(match):
        nonlocal idx
        key = f"__CHEM_PAREN_{idx}__"
        protected[key] = match.group(0)
        idx += 1
        return key

    text = CHEM_PAREN_RE.sub(repl, text)
    return text, protected

def restore_chemical_parens(text, protected):
    for k, v in protected.items():
        text = text.replace(k, v)
    return text

def normalize_parentheses_tokens(tokens):
    """
    Ensure no token contains unbalanced parentheses.
    Works with nested parentheses and chemical formulas.
    """
    out = []

    for tok in tokens:
        if tok.count("(") == tok.count(")"):
            out.append(tok)
            continue

        buf = ""
        balance = 0

        for ch in tok:
            if ch == "(":
                if buf:
                    out.append(buf)
                    buf = ""
                out.append("(")
                balance += 1
            elif ch == ")":
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(")")
                balance -= 1
            else:
                buf += ch

        if buf:
            out.append(buf)

    return out

def tokenize(text: str):
    assert isinstance(text, str)

    # 1) Separate ONLY linguistic commas
    text = text.replace(", ", " , ")

    # 2) Split on spaces
    raw_tokens = text.split()

    # 3) Normalize parentheses (CRITICAL STEP)
    raw_tokens = normalize_parentheses_tokens(raw_tokens)

    # 4) Split trailing punctuation + clean
    tokens = []
    for rt in raw_tokens:
        for t in split_trailing_punct(rt):
            t = clean_token(t)
            t = clean_numeric_token(t)
            if t:
                tokens.append(t)

    # 5) Merge charges / wildcard / optional N-(
    tokens = postprocess_tokens(tokens)

    return tokens
