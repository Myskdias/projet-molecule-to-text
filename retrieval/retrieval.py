import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class RetrievalIndex:
    """
    Gestionnaire de la base de connaissances (Vector Database).
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.database_embeddings = None # [N, Dim]
        self.train_captions_tokens = [] # Liste de tenseurs [Seq_Len]

    def build_index(self, encoder, dataloader, captions_tokens_list):
        """
        Construit l'index en passant tout le dataset dans l'encodeur.
        Utilise l'encodeur (potentiellement pré-entraîné CLIP) pour vectoriser les graphes.
        """
        encoder.eval()
        encoder.to(self.device)
        self.train_captions_tokens = captions_tokens_list
        embeddings_list = []
        
        print("Construction de l'Index Vectoriel...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Indexing"):
                graph_batch = batch[0].to(self.device)
                
                # On récupère le vecteur global du graphe (sortie 1)
                graph_emb = encoder.encode_graph(graph_batch)
                
                # Stockage CPU pour éviter OOM
                embeddings_list.append(graph_emb.cpu())
        
        self.database_embeddings = torch.cat(embeddings_list, dim=0).to(self.device)
        print(f"Index prêt. {self.database_embeddings.shape[0]} molécules indexées.")

    def query(self, encoder, query_batch, k=3):
        """
        Recherche les k plus proches voisins pour un batch de requêtes.
        """
        encoder.eval()
        with torch.no_grad():
            query_batch = query_batch.to(self.device)
            query_emb = encoder.encode_graph(query_batch)
            
            # Produit scalaire (Cosinus Similarity sur vecteurs normalisés)
            # [Batch, Dim] @ [Dim, N] -> [Batch, N]
            similarity = torch.mm(query_emb, self.database_embeddings.t())
            
            scores, indices = torch.topk(similarity, k=k, dim=1)
            return indices, scores

    def get_retrieved_tokens(self, indices):
        """
        Renvoie les tokens des textes associés aux indices trouvés.
        Gère le padding dynamique.
        """
        indices_cpu = indices.cpu().tolist()
        batch_tokens = [self.train_captions_tokens[idx] for idx in indices_cpu]
        
        # Padding avec 0 (supposé PAD_IDX)
        padded_tokens = pad_sequence(batch_tokens, batch_first=True, padding_value=0)
        return padded_tokens.long()