import pickle
from collections import Counter
import random

import torch
from torch_geometric.data import Batch

from editor.masker import load_vocab, mask_out_of_vocab
from editor.tokenizer import tokenize
from editor.transformer.utils import extract_unk_samples, is_informative_target
from retrieval.architecture import DeepGINEEncoder

# ========= IMPORTS DE TON PROJET =========
# - tokenize
# - mask_out_of_vocab
# - extract_unk_samples
# - load_vocab

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
VOCAB_PKL = "vocab_minfreq15_nodigits.pkl"
DATA_PKL = "data/train_graphs.pkl"

N_SHOW = 20
CTX_SIZE = 8
SEED = 42

random.seed(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NODE_VOCAB = [200, 20]
EDGE_VOCAB = [50, 20]
HIDDEN_GRAPH = 300
GRAPH_CKPT = "other/weights_stage1_clip.pt"  # même fichier que pour l'éval

print("[INFO] Loading graph encoder...")
graph_encoder = DeepGINEEncoder(
    NODE_VOCAB,
    EDGE_VOCAB,
    hidden_dim=HIDDEN_GRAPH,
).to(DEVICE)

ckpt = torch.load(GRAPH_CKPT, map_location=DEVICE)
graph_encoder.load_state_dict(ckpt["graph_encoder"])
graph_encoder.eval()

# ------------------------------------------------------------------
# LOAD
# ------------------------------------------------------------------
print("[INFO] Loading vocab...")
vocab = load_vocab(VOCAB_PKL)

token2id = {t: i for i, t in enumerate(sorted(vocab))}
id2token = {i: t for t, i in token2id.items()}

print("[INFO] Loading data...")
with open(DATA_PKL, "rb") as f:
    data = pickle.load(f)

# ------------------------------------------------------------------
# BUILD SAMPLES
# ------------------------------------------------------------------
all_samples = []

for mol in data:
    caption = mol.description

    masked = mask_out_of_vocab(caption, vocab, tokenize)

    with torch.no_grad():
        batch = Batch.from_data_list([mol]).to(DEVICE)
        graph_emb = graph_encoder(batch)[0].squeeze(0).cpu()

    samples = extract_unk_samples(
        original_text=caption,
        masked_text=masked,
        graph_emb=graph_emb,
        tokenize_fn=tokenize,
        token2id=token2id,
        text_encoder=None
        ctx_size=CTX_SIZE,
    )
    for sample in samples:
        if is_informative_target(sample["target_span"]):
            all_samples.extend(samples)



print(f"\n[INFO] Total UNK samples: {len(all_samples)}")

print("\n================ EXAMPLES (WITH ORIGINAL) ================\n")

for s in random.sample(all_samples, min(N_SHOW, len(all_samples))):
    left = [id2token[i] for i in s["ctx_left_ids"]]
    right = [id2token[i] for i in s["ctx_right_ids"]]

    print("ORIGINAL :", s.get("original_text", "N/A"))
    print("MASKED   :", s.get("masked_text", "N/A"))
    print("LEFT     :", " ".join(left))
    print("TARGET   :", repr(s["target_span"]))
    print("RIGHT    :", " ".join(right))
    print("-" * 80)
# ------------------------------------------------------------------
# STATS
# ------------------------------------------------------------------
lengths = [len(s["target_span"]) for s in all_samples]

print("\n================ STATS ===================\n")
print(f"Mean span length: {sum(lengths) / len(lengths):.2f}")
print(f"Max  span length: {max(lengths)}")

length_bins = Counter(lengths)
print("\nTop span lengths:")
for k, v in length_bins.most_common(10):
    print(f"  len={k:2d} → {v}")

# ------------------------------------------------------------------
# COMMON PATTERNS
# ------------------------------------------------------------------
print("\n================ COMMON PREFIXES =========\n")

prefix_counter = Counter()
for s in all_samples:
    prefix = s["target_span"][:10]
    prefix_counter[prefix] += 1

for p, c in prefix_counter.most_common(10):
    print(f"{p!r}: {c}")


print("\n================ UNBALANCED PAREN SPANS ================\n")

count = 0
for s in all_samples:
    t = s["target_span"]
    if t.count("(") != t.count(")"):
        print("ORIGINAL :", s["original_text"])
        print("MASKED   :", s["masked_text"])
        print("TARGET   :", repr(t))
        print("open:", t.count("("), "close:", t.count(")"))
        print("-" * 80)
        count += 1
        if count >= 10:
            break

print(f"\nFound {count} unbalanced spans")







import re

DIGIT_RE = re.compile(r"\d")

def _is_chem_paren_attached(prev_char: str, next_char: str) -> bool:
    # If parens are attached to non-space on either side -> chemical/lexical
    return (prev_char != "" and not prev_char.isspace()) or (next_char != "" and not next_char.isspace())

def _is_x_dash_paren_prefix(text: str, i: int) -> bool:
    """
    Detect patterns like 'N-(' or 'O-(' or 'S-(' etc right at position i where text[i] == '('.
    We look back for a short alphabetic prefix ending with '-'.
    """
    if i < 2:
        return False
    # find immediate pattern "...<letters>-("
    return (text[i-1] == "-" and text[i-2].isalpha())

def tokenize(text: str):
    """
    Robust tokenizer:
    - Emits syntactic '(' and ')' as standalone tokens.
    - Keeps chemical parentheses attached (manganese(II), acyl-CoA(4-), (KDO)2, NAD(+)).
    - Keeps X-(...)Y (e.g., N-(polyunsaturated fatty acyl)ethanolamine) as ONE token.
    - Keeps your basic punctuation tokens.
    """
    tokens = []
    i = 0
    n = len(text)

    # basic punctuation to isolate
    punct = set([",", ".", ":", ";"])

    while i < n:
        c = text[i]

        # spaces
        if c.isspace():
            i += 1
            continue

        # punctuation
        if c in punct:
            tokens.append(c)
            i += 1
            continue

        # parentheses handling
        if c == "(":
            # If this is X-(...)Y chemical pattern, capture whole word until it ends
            if _is_x_dash_paren_prefix(text, i):
                # capture from the start of the prefix word (go left to previous space/punct)
                l = i - 2
                while l > 0 and (not text[l-1].isspace()) and (text[l-1] not in punct):
                    l -= 1

                # now capture balanced parentheses starting at i, then continue consuming
                depth = 1
                j = i + 1
                while j < n and depth > 0:
                    if text[j] == "(":
                        depth += 1
                    elif text[j] == ")":
                        depth -= 1
                    j += 1

                # after matching ')', keep consuming chars of the same lexical token (until space/punct)
                k = j
                while k < n and (not text[k].isspace()) and (text[k] not in punct):
                    k += 1

                tokens.append(text[l:k])
                i = k
                continue

            # Otherwise decide syntactic vs chemical by attachment
            prev_char = text[i-1] if i > 0 else ""
            next_char = text[i+1] if i + 1 < n else ""

            if _is_chem_paren_attached(prev_char, next_char):
                # chemical: absorb until we reach a space/punct (keep as part of the token)
                j = i
                while j < n and (not text[j].isspace()) and (text[j] not in punct):
                    j += 1
                tokens.append(text[i:j])
                i = j
                continue
            else:
                # syntactic '('
                tokens.append("(")
                i += 1
                continue

        if c == ")":
            # same decision: if attached, keep in the current lexical token; else standalone
            prev_char = text[i-1] if i > 0 else ""
            next_char = text[i+1] if i + 1 < n else ""

            if _is_chem_paren_attached(prev_char, next_char):
                j = i
                while j < n and (not text[j].isspace()) and (text[j] not in punct):
                    j += 1
                tokens.append(text[i:j])
                i = j
                continue
            else:
                tokens.append(")")
                i += 1
                continue

        # default: consume a token until whitespace/punct, but STOP before a standalone syntactic '(' or ')'
        j = i
        while j < n:
            if text[j].isspace() or text[j] in punct:
                break
            # stop before syntactic parentheses (we'll handle them next loop)
            if text[j] in "()":
                # if paren is attached, we don't stop
                prev = text[j-1] if j > 0 else ""
                nxt = text[j+1] if j + 1 < n else ""
                if not _is_chem_paren_attached(prev, nxt) and not _is_x_dash_paren_prefix(text, j):
                    break
            j += 1

        tokens.append(text[i:j])
        i = j

    # optional: remove square brackets everywhere like you did
    tokens = [t.replace("[", "").replace("]", "") for t in tokens if t != ""]

    return tokens
