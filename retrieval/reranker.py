from sacrebleu import CHRF
import torch
import torch.nn.functional as F


@torch.no_grad()
def rerank_topk_simple(captions, text_encoder):
    """
    captions: list[str] (top-k)
    text_encoder: MiniLMTextEncoder
    returns: best caption (str)
    """

    # Encode captions
    emb = text_encoder.encode_text(captions)  # [k, D]
    emb = F.normalize(emb, dim=-1)

    # Mean embedding (centroid lexical)
    centroid = emb.mean(dim=0, keepdim=True)  # [1, D]

    # Similarity to centroid
    sims = (emb @ centroid.T).squeeze(1)  # [k]

    best_idx = sims.argmax().item()
    return captions[best_idx]


@torch.no_grad()
def rerank_topk_hybrid(captions, graph_scores, text_encoder, alpha=0.7):
    """
    captions: list[str] (top-k)
    graph_scores: list[float] (scores graphe↔texte)
    text_encoder: MiniLMTextEncoder
    alpha: poids du score graphe
    """

    # Encode captions with MiniLM
    text_emb = text_encoder(captions)  # [k, D]
    text_emb = F.normalize(text_emb, dim=-1)

    # Lexical centroid
    centroid = text_emb.mean(dim=0, keepdim=True)
    lexical_sim = (text_emb @ centroid.T).squeeze(1)  # [k]

    graph_sim = torch.tensor(graph_scores, device=lexical_sim.device)

    score = alpha * graph_sim + (1 - alpha) * lexical_sim
    best_idx = score.argmax().item()

    return captions[best_idx]

@torch.no_grad()
def rerank_topk_with_graph(graph_emb, captions, text_encoder, alpha=0.7):
    """
    graph_emb: [D] embedding du graphe query
    captions: list[str]
    """

    # Encode captions
    text_emb = text_encoder.encode_text(captions)
    text_emb = F.normalize(text_emb, dim=-1)

    graph_emb = F.normalize(graph_emb.unsqueeze(0), dim=-1)

    # Score graphe↔texte
    graph_sim = (text_emb @ graph_emb.T).squeeze(1)

    # Score lexical
    centroid = text_emb.mean(dim=0, keepdim=True)
    lexical_sim = (text_emb @ centroid.T).squeeze(1)

    score = alpha * graph_sim + (1 - alpha) * lexical_sim
    return captions[score.argmax().item()]

chrf = CHRF(word_order=2)  # proche BLEU

@torch.no_grad()
def rerank_topk_mbr(captions):
    """
    captions: list[str] (top-k)
    returns: best caption (str)
    """

    k = len(captions)
    if k == 1:
        return captions[0]

    scores = []

    for i in range(k):
        c_i = captions[i]
        total = 0.0

        for j in range(k):
            if i == j:
                continue
            c_j = captions[j]
            total += chrf.sentence_score(c_i, [c_j]).score

        scores.append(total / (k - 1))

    best_idx = max(range(k), key=lambda i: scores[i])
    return captions[best_idx]

def rerank_topk_mbr_weighted(captions, graph_scores, beta=0.2):
    k = len(captions)
    scores = []

    for i in range(k):
        total = 0.0
        for j in range(k):
            if i != j:
                total += chrf.sentence_score(captions[i], [captions[j]]).score

        mbr_score = total / (k - 1)
        final_score = (1 - beta) * mbr_score + beta * graph_scores[i]
        scores.append(final_score)

    return captions[max(range(k), key=lambda i: scores[i])]
