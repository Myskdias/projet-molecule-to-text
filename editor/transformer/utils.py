from collections import Counter

def build_char_vocab(spans, min_freq=1):
    """
    spans: list[str] — all UNK target strings from train
    """
    counter = Counter()
    for s in spans:
        counter.update(list(s))

    chars = [c for c, f in counter.items() if f >= min_freq]

    # Special tokens
    vocab = ["<PAD>", "<BOS>", "<EOS>"] + sorted(chars)

    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for c, i in char2id.items()}

    return char2id, id2char


import torch
from torch.utils.data import Dataset

class UNKCharDataset(Dataset):
    def __init__(
        self,
        samples,            # list of dicts
        char2id,
        pad_char="<PAD>",
        bos_char="<BOS>",
        eos_char="<EOS>",
        max_len=64,
    ):
        self.samples = samples
        self.char2id = char2id
        self.pad_id = char2id[pad_char]
        self.bos_id = char2id[bos_char]
        self.eos_id = char2id[eos_char]
        self.max_len = max_len

    def encode_target(self, s):
        ids = [self.bos_id] + [self.char2id[c] for c in s] + [self.eos_id]
        ids = ids[: self.max_len]
        pad_len = self.max_len - len(ids)
        ids += [self.pad_id] * pad_len
        return torch.tensor(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        return {
            "ctx_left": torch.tensor(s["ctx_left_ids"]),
            "ctx_right": torch.tensor(s["ctx_right_ids"]),
            "graph_emb": torch.tensor(s["graph_emb"], dtype=torch.float),
            "target": self.encode_target(s["target_span"]),
        }

def split_sentences(text: str):
    """
    Split text into sentences using '. ' as delimiter.
    Returns list of (start, end, sentence_text).
    """
    sentences = []
    start = 0
    i = 0
    n = len(text)

    while i < n - 1:
        if text[i:i+2] == ". ":
            end = i + 1  # include '.'
            sentences.append((start, end, text[start:end]))
            start = i + 2
            i += 2
        else:
            i += 1

    # last sentence
    if start < n:
        sentences.append((start, n, text[start:]))

    return sentences


def extract_unk_samples(
    original_text: str,
    masked_text: str,
    graph_emb,
    tokenize_fn,
    token2id,
    text_encoder,                 # callable: str -> Tensor
    unk_token="<UNK>",
    ctx_size=8,
):
    # --------------------------------------------------
    # 0) Sentence split BEFORE tokenization
    # --------------------------------------------------
    orig_sents = split_sentences(original_text)
    masked_sents = split_sentences(masked_text)

    assert len(orig_sents) == len(masked_sents), "Sentence split mismatch"

    # --------------------------------------------------
    # 1) Tokenize sentence by sentence
    # --------------------------------------------------
    orig_tok_sents = [tokenize_fn(s[2]) for s in orig_sents]
    masked_tok_sents = [tokenize_fn(s[2]) for s in masked_sents]

    # --------------------------------------------------
    # 2) Flatten tokens + sentence index tracking
    # --------------------------------------------------
    orig_tokens = []
    masked_tokens = []
    sent_idx_of_token = []

    for si, (ot, mt) in enumerate(zip(orig_tok_sents, masked_tok_sents)):
        assert len(ot) == len(mt), "Token alignment failed inside sentence"
        for o, m in zip(ot, mt):
            orig_tokens.append(o)
            masked_tokens.append(m)
            sent_idx_of_token.append(si)

    # --------------------------------------------------
    # 3) First sentence embedding (MASKED, computed ONCE)
    # --------------------------------------------------
    first_sentence_text = masked_sents[0][2]
    first_sentence_emb = text_encoder(first_sentence_text)

    # --------------------------------------------------
    # 4) Extract UNK spans
    # --------------------------------------------------
    samples = []
    i = 0
    n = len(masked_tokens)

    while i < n:
        if masked_tokens[i] != unk_token:
            i += 1
            continue

        # UNK group [i:j)
        j = i
        while j < n and masked_tokens[j] == unk_token:
            j += 1

        target = " ".join(orig_tokens[i:j])

        # -------- target filter ----------
        if not is_informative_target(target):
            i = j
            continue

        # -------- local context (KEEP UNK) ----------
        left_tokens = []
        k = i - 1
        while k >= 0 and len(left_tokens) < ctx_size:
            left_tokens.append(masked_tokens[k])
            k -= 1
        left_tokens = left_tokens[::-1]

        right_tokens = []
        k = j
        while k < n and len(right_tokens) < ctx_size:
            right_tokens.append(masked_tokens[k])
            k += 1

        ctx_left_ids = [token2id[t] for t in left_tokens]
        ctx_right_ids = [token2id[t] for t in right_tokens]

        # -------- sentence-level info ----------
        span_sent_idx = sent_idx_of_token[i]
        in_first_sentence = (span_sent_idx == 0)

        if in_first_sentence:
            use_first_sentence_emb = False
            first_sentence_emb_sample = None
        else:
            use_first_sentence_emb = True
            first_sentence_emb_sample = first_sentence_emb

        # -------- current sentence first words (RAW TEXT) ----------
        curr_sentence_text = masked_sents[span_sent_idx][2]
        current_sentence_words = curr_sentence_text.split()[:3]

        samples.append({
            # core
            "ctx_left_ids": ctx_left_ids,
            "ctx_right_ids": ctx_right_ids,
            "graph_emb": graph_emb,
            "target_span": target,

            # global conditioning
            "use_first_sentence_embedding": use_first_sentence_emb,
            "first_sentence_emb": first_sentence_emb_sample,

            # local sentence typing
            "current_sentence_words": current_sentence_words,

            # optional (debug / analysis)
            "sentence_index": span_sent_idx,
        })

        i = j

    return samples



import re

CHEM_SUFFIX_RE = re.compile(
    r"(yl|oyl|ate|ine|one|ene|acid|amide|amine|ol|ose|ene|yne|ium|ate)s?$",
    re.IGNORECASE,
)

def is_informative_target(target: str) -> bool:
    target = target.strip()

    # Rule 1: length
    if len(target) < 4 or len(target) > 120:
        return False

    tokens = target.split()

    # Rule 3: reject single common-looking words
    if (
        len(tokens) == 1
        and tokens[0].islower()
        and tokens[0].isalpha()
    ):
        # keep if chemically suggestive
        if CHEM_SUFFIX_RE.search(tokens[0]):
            return True
        return False

    # Rule 2: chemical signals
    if any(c.isdigit() for c in target):
        return True
    if any(c in target for c in "-(),"):
        return True
    if any(c.isupper() for c in target[1:]):  # internal capital
        return True
    if CHEM_SUFFIX_RE.search(target):
        return True

    return False

