import torch
import torch.nn.functional as F
from tqdm import tqdm


class RetrievalIndex:
    """
    Stores text embeddings + raw text.
    Query with graph embeddings (same shared space).
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.text_emb = None          # [N, D] (on device)
        self.texts = None             # list[str]

    @torch.no_grad()
    def build_text_index(self, clip_model, tokenizer, train_texts, batch_size=64, max_len=256):
        """
        clip_model.encode_text(...) must return normalized vectors in shared space.
        """
        clip_model.eval()
        all_vecs = []
        self.texts = list(train_texts)

        for i in tqdm(range(0, len(train_texts), batch_size), desc="Index texts"):
            batch_txt = train_texts[i:i+batch_size]
            tok = tokenizer(
                batch_txt,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(self.device)
            v = clip_model.encode_text(tok["input_ids"], tok["attention_mask"])  # [B,D], normalized
            all_vecs.append(v.detach().cpu())

        self.text_emb = torch.cat(all_vecs, dim=0).to(self.device)
        self.text_emb = F.normalize(self.text_emb, dim=1)
        return self

    @torch.no_grad()
    def query(self, clip_model, batch_graph, k=5):
        """
        Returns: indices [B,k], scores [B,k]
        """
        clip_model.eval()
        batch_graph = batch_graph.to(self.device)
        q = clip_model.encode_graph(batch_graph)  # [B,D], normalized
        sims = q @ self.text_emb.t()              # [B,N]
        scores, idx = torch.topk(sims, k=k, dim=1)
        return idx, scores

    def get_texts(self, indices_1d):
        """
        indices_1d: shape [B] on cpu or gpu
        """
        idx = indices_1d.detach().cpu().tolist()
        return [self.texts[i] for i in idx]
