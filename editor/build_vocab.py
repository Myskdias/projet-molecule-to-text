from collections import Counter
import re
from typing import List, Set

# ------------------
# Regex / constantes
# ------------------
DIGIT_RE = re.compile(r"\d")

SYMBOL_ONLY = {"*", "+", "="}

STEREO_KEEP = {
    "(R)-configuration",
    "(S)-configuration",
    "(R)-enantiomer",
    "(S)-enantiomer",
}

# ------------------
# Token exclusion
# ------------------
def is_malformed(tok: str) -> bool:
    # unbalanced parentheses
    if tok.count("(") != tok.count(")"):
        return True

    # dangling dash
    if tok.endswith("-"):
        return True

    return False


def should_exclude_from_vocab(tok: str) -> bool:
    # symbols alone
    if tok in SYMBOL_ONLY:
        return True

    # any digit → exclude
    if DIGIT_RE.search(tok):
        return True
    if tok == "℃":
        return True
    # malformed tokens
    if is_malformed(tok):
        return True

    # chemical proper names in parentheses
    if tok.startswith("(") and tok not in STEREO_KEEP:
        return True

    return False

# ------------------
# Build vocabulary
# ------------------
def build_vocab(
    captions: List[str],
    tokenize_fn,
    min_freq: int = 15,
):
    """
    Build a closed vocabulary from GT(train).

    Args:
        captions: list of GT(train) captions (strings)
        tokenize_fn: final tokenizer function
        min_freq: minimum frequency threshold

    Returns:
        vocab: set of tokens
        counter: Counter with full token frequencies
    """

    counter = Counter()

    for text in captions:
        tokens = tokenize_fn(text)
        counter.update(tokens)

    vocab: Set[str] = set()

    for tok, freq in counter.items():
        if freq < min_freq:
            continue
        if should_exclude_from_vocab(tok):
            continue
        vocab.add(tok)

    # Special tokens (always kept)
    vocab.update({
        "<PAD>",
        "<UNK>",
        "<KEEP>",
    })

    return vocab, counter
