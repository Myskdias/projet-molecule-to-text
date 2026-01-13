def mask_out_of_vocab(text: str, vocab: set, tokenize_fn) -> str:
    """
    Replace tokens not in vocab by <UNK>, keeping everything else unchanged.

    Args:
        text: input sentence
        vocab: set of allowed tokens
        tokenize_fn: final tokenizer

    Returns:
        masked sentence as a string
    """
    tokens = tokenize_fn(text)

    masked_tokens = []
    for tok in tokens:
        if tok in vocab:
            masked_tokens.append(tok)
        else:
            masked_tokens.append("<UNK>")

    return " ".join(masked_tokens)

import pickle

def load_vocab(vocab_path: str) -> set:
    """
    Load vocabulary from a pickle file.

    Args:
        vocab_path: path to .pkl vocabulary file

    Returns:
        vocab: set of tokens
    """
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    assert isinstance(vocab, set), f"Expected vocab to be a set, got {type(vocab)}"
    return vocab
