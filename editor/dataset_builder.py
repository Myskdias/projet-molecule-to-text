import torch
from tqdm import tqdm
from retrieval.reranker import rerank_topk_hybrid


@torch.no_grad()
def build_edit_text_pairs(
    ds_train,
    dl_train,
    id2desc_train,
    clip_model,
    retriever,
    text_encoder,
    top_k=10,
):
    """
    Construit (retrieved_texts, gold_texts) pour le Niveau 1 editor.
    """

    retrieved_texts = []
    gold_texts = []

    clip_model.eval()
    text_encoder.eval()

    print("[EDITOR DATASET] Building retrieved/gold pairs from TRAIN...")

    for batch_graph, _ in tqdm(dl_train, desc="Editor dataset"):
        batch_graph = batch_graph.to(clip_model.device)
        graphs = batch_graph.to_data_list()

        # Retrieval top-k
        nn_indices, scores = retriever.query(
            clip_model,
            batch_graph,
            k=top_k + 1,  # +1 pour pouvoir exclure self
        )

        for b, graph in enumerate(graphs):
            graph_id = graph.id
            gold_text = id2desc_train[graph_id]

            idx_b = nn_indices[b].tolist()
            scores_b = scores[b].tolist()

            # Exclure self-match
            candidates = []
            candidate_scores = []

            for idx, score in zip(idx_b, scores_b):
                train_graph = ds_train.graphs[int(idx)]
                if train_graph.id != graph_id:
                    candidates.append(id2desc_train[train_graph.id])
                    candidate_scores.append(score)

                if len(candidates) == top_k:
                    break

            if len(candidates) == 0:
                continue  # sécurité (rare)

            # Reranking EXACTEMENT comme en inference
            retrieved_text = rerank_topk_hybrid(
                captions=candidates,
                graph_scores=candidate_scores,
                text_encoder=text_encoder,
                alpha=0.7,
            )

            retrieved_texts.append(retrieved_text)
            gold_texts.append(gold_text)

    print(f"[EDITOR DATASET] Built {len(retrieved_texts)} pairs.")
    return retrieved_texts, gold_texts