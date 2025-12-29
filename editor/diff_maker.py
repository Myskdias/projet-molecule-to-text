import difflib
from typing import List
from torch.utils.data import Dataset

import torch

KEEP, DELETE, REPLACE = 0, 1, 2

def compute_edit_labels(retrieved_tokens, gold_tokens):
    """
    Args:
        retrieved_tokens: List[str]
        gold_tokens: List[str]

    Returns:
        edit_labels: List[int]  # same length as retrieved_tokens
    """
    matcher = difflib.SequenceMatcher(
        a=retrieved_tokens,
        b=gold_tokens,
        autojunk=False
    )

    edit_labels = [KEEP] * len(retrieved_tokens)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # tokens identical → KEEP (déjà initialisé)
            continue

        elif tag == "delete":
            # retrieved[i1:i2] deleted in gold
            for i in range(i1, i2):
                edit_labels[i] = DELETE

        elif tag == "replace":
            # retrieved[i1:i2] replaced by gold[j1:j2]
            for i in range(i1, i2):
                edit_labels[i] = REPLACE

        elif tag == "insert":
            # gold has extra tokens → ignored at Niveau 1
            continue

    return edit_labels

class EditDataset(Dataset):
    def __init__(
        self,
        retrieved_texts: List[str],
        gold_texts: List[str],
        tokenizer,
        max_length: int = 128,
        debug: bool = False,
    ):
        assert len(retrieved_texts) == len(gold_texts)

        self.retrieved_texts = retrieved_texts
        self.gold_texts = gold_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug = debug

    def __len__(self):
        return len(self.retrieved_texts)

    def __getitem__(self, idx):
        retrieved = self.retrieved_texts[idx]
        gold = self.gold_texts[idx]

        # Tokenisation SANS padding (important pour l’alignement)
        retrieved_enc = self.tokenizer(
            retrieved,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )
        gold_enc = self.tokenizer(
            gold,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=False,
        )

        input_ids = retrieved_enc["input_ids"]
        attention_mask = retrieved_enc["attention_mask"]

        # Conversion ids → tokens (pour le diff)
        retrieved_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        gold_tokens = self.tokenizer.convert_ids_to_tokens(gold_enc["input_ids"])

        # Calcul des labels KEEP / DELETE / REPLACE
        edit_labels = compute_edit_labels(retrieved_tokens, gold_tokens)

        # Sécurité : même longueur
        assert len(edit_labels) == len(input_ids)

        # Padding manuel (important pour labels)
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            edit_labels = edit_labels + [KEEP] * pad_len

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "edit_labels": torch.tensor(edit_labels, dtype=torch.long),
        }

        if self.debug:
            item["tokens"] = retrieved_tokens

        return item
