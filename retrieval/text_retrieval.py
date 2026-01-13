import torch
from tqdm import tqdm


class TextRetrievalIndex:
    """
    Vector database for TEXT embeddings in the SAME latent space as graph embeddings.

    IMPORTANT:
      - Graph embeddings come from: clip_model.encode_graph(batch_graph)  -> [B, D]
      - Text embeddings come from:  text_encoder(list[str])              -> [B, D]
        where `text_encoder` is your MiniLMTextEncoder (already normalized in your codebase).
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.database_embeddings = None  # [M, D]
        self.database_texts = None       # List[str], length M
        self.database_meta = None        # Optional[List[dict]]

    @torch.no_grad()
    def build_index_from_texts(self, text_encoder, texts, metas=None, batch_size=128):
        """
        Build retrieval index from text embeddings only.

        Args:
          text_encoder: MiniLMTextEncoder
          texts: List[str]
          metas: Optional[List[dict]] aligned with texts
        """
        assert isinstance(texts, list) and len(texts) > 0
        if metas is not None:
            assert len(metas) == len(texts)

        text_encoder.eval()

        self.database_texts = texts
        self.database_meta = metas

        all_embeddings = []

        print("Construction de l'Index Vectoriel (TEXT)...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Indexing TEXT"):
            batch_texts = texts[i:i + batch_size]

            # MiniLMTextEncoder in your project takes List[str] and returns [B, D]
            text_emb = text_encoder(batch_texts)  # [B, D] (assumed normalized)
            all_embeddings.append(text_emb.cpu())

        self.database_embeddings = torch.cat(all_embeddings, dim=0).to(self.device)
        print(f"Index texte prêt. {self.database_embeddings.shape[0]} captions indexées.")

    @torch.no_grad()
    def query_with_graphs(self, clip_model, query_batch, k=3):
        """
        Retrieve top-k nearest captions for query graphs.

        Returns:
          indices: [B, k] indices in database_texts
          scores : [B, k] cosine/dot scores
        """
        clip_model.eval()
        query_batch = query_batch.to(self.device)

        query_emb = clip_model.encode_graph(query_batch)  # [B, D] normalized
        similarity = query_emb @ self.database_embeddings.t()  # [B, M]

        scores, indices = torch.topk(similarity, k=k, dim=1)
        return indices, scores

    def get_texts(self, indices_1d):
        return [self.database_texts[int(i)] for i in indices_1d]

    def get_meta(self, indices_1d):
        if self.database_meta is None:
            return None
        return [self.database_meta[int(i)] for i in indices_1d]
