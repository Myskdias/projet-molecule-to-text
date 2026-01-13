import torch

def analyze_editability(
    text: str,
    tokenizer,
    editor_model,
    device,
    threshold=0.85,
):
    """
    Retourne les tokens avec proba REPLACE élevée.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        logits = editor_model(
            enc["input_ids"],
            enc["attention_mask"]
        )
        probs = torch.softmax(logits, dim=-1)[0]  # [L, 3]

    tokens = tokenizer.convert_ids_to_tokens(
        enc["input_ids"][0]
    )

    editable = []
    for tok, p in zip(tokens, probs):
        if p[2].item() > threshold and is_editable_token(tok):  # REPLACE
            editable.append((tok, p[2].item()))

    return editable

def is_editable_token(tok: str) -> bool:
    # ignore special tokens
    if tok in {"[CLS]", "[SEP]", "[PAD]"}:
        return False
    # ignore wordpiece continuation
    if tok.startswith("##"):
        return False
    # ignore pure punctuation / symbols
    if all(ch in ".,;:!?()[]{}-+/" for ch in tok):
        return False
    return True