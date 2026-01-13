from editor.tokenizer import tokenize
from editor.build_vocab import build_vocab
from utils.data_utils import PreprocessedGraphDataset, load_descriptions_from_graphs

import pickle

VOCAB_PATH = "vocab_minfreq15_nodigits.pkl"

TRAIN_GRAPHS = "data/train_graphs.pkl"
id2desc_train = load_descriptions_from_graphs(TRAIN_GRAPHS)
ds_train = PreprocessedGraphDataset(TRAIN_GRAPHS)
pool_texts = []
pool_meta = []
for g in ds_train.graphs:
        gid = g.id
        pool_texts.append(id2desc_train[gid])
        pool_meta.append({"mol_id": gid, "source": "gt_train"})


vocab, counter = build_vocab(
    captions=pool_texts,
    tokenize_fn=tokenize,
    min_freq=15,
)

print(f"Vocab size: {len(vocab)}")

# inspect top-20 tokens hors vocab
rare_tokens = [
    (tok, freq) for tok, freq in counter.items()
    if tok not in vocab
]
rare_tokens = sorted(rare_tokens, key=lambda x: -x[1])[:20]

for tok, freq in rare_tokens:
    print(tok, freq)

with open(VOCAB_PATH, "wb") as f:
    pickle.dump(vocab, f)

print(f"Vocab saved to {VOCAB_PATH} (size={len(vocab)})") 

import json

VOCAB_JSON_PATH = "vocab_minfreq15_nodigits.json"

with open(VOCAB_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(sorted(vocab), f, indent=2, ensure_ascii=False)

print(f"Vocab saved to {VOCAB_JSON_PATH}")

