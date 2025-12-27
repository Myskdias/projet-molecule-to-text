import torch
from tqdm import tqdm


class RetrievalIndex:
    """
    Vector database for graph embeddings (MiniLM-compatible).
    Graph-only retrieval: no text or tokens involved.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.database_embeddings = None  # [N, D]

    @torch.no_grad()
    def build_index(self, encoder, dataloader):
        """
        Build retrieval index from graph embeddings only.
        """
        encoder.eval()
        encoder.to(self.device)

        all_embeddings = []

        print("Construction de l'Index Vectoriel...")
        for batch_graph, _ in tqdm(dataloader, desc="Indexing"):
            batch_graph = batch_graph.to(self.device)

            graph_emb = encoder.encode_graph(batch_graph)  # [B, D]
            all_embeddings.append(graph_emb.cpu())

        self.database_embeddings = torch.cat(all_embeddings, dim=0).to(self.device)
        print(f"Index prêt. {self.database_embeddings.shape[0]} molécules indexées.")

    @torch.no_grad()
    def query(self, encoder, query_batch, k=3):
        """
        Retrieve top-k nearest neighbors for query graphs.
        """
        encoder.eval()
        query_batch = query_batch.to(self.device)

        query_emb = encoder.encode_graph(query_batch)  # [B, D]

        # Cosine similarity (embeddings already normalized)
        similarity = query_emb @ self.database_embeddings.t()  # [B, N]

        scores, indices = torch.topk(similarity, k=k, dim=1)
        return indices, scores
