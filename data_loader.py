import torch
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer

class MolecularDataset(Dataset):
    def __init__(self, graphs_path, tokenizer_path, node_limits=[200, 20], edge_limits=[50, 20]):
        print(f"--- Initialisation MolecularDataset ---")
        
        # 1. Chargement des Graphes (qui contiennent aussi les descriptions)
        print(f"Chargement des données depuis {graphs_path}...")
        try:
            with open(graphs_path, 'rb') as f:
                self.graphs = pickle.load(f)
            print(f"✅ Chargé {len(self.graphs)} exemples.")
        except FileNotFoundError:
            print(f"❌ Erreur critique : Fichier {graphs_path} introuvable.")
            self.graphs = []
            
        # 2. Chargement du Tokenizer
        print(f"Chargement du tokenizer {tokenizer_path}...")
        try:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            print(f"✅ Tokenizer chargé (Vocab: {self.tokenizer.get_vocab_size()})")
        except Exception as e:
            print(f"❌ Erreur chargement tokenizer: {e}")
            self.tokenizer = None
        
        # Limites de sécurité pour les embeddings (Atomes / Liaisons)
        self.node_limits = node_limits
        self.edge_limits = edge_limits

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        # A. Récupération de l'objet graphe complet
        original_graph = self.graphs[idx]
        
        # B. Extraction du texte (attribut .description)
        # On utilise getattr pour éviter un crash si l'attribut manque
        description_text = getattr(original_graph, 'description', "")
        if description_text is None:
            description_text = ""
        
        # C. Tokenization
        # encode() ajoute automatiquement START/END si le tokenizer a été configuré pour
        encoded = self.tokenizer.encode(str(description_text))
        caption_tensor = torch.tensor(encoded.ids, dtype=torch.long)
        
        # D. Sécurité Graphe (Clamping)
        # On modifie les tenseurs du graphe pour s'assurer qu'ils ne dépassent pas les limites
        # Note : PyG Data objects have attributes accessible directly
        if hasattr(original_graph, 'x'):
            # x shape: [num_nodes, num_node_features]
            for i, limit in enumerate(self.node_limits):
                if i < original_graph.x.size(1):
                    original_graph.x[:, i] = torch.clamp(original_graph.x[:, i], min=0, max=limit-1)
                    
        if hasattr(original_graph, 'edge_attr') and original_graph.edge_attr is not None:
            # edge_attr shape: [num_edges, num_edge_features]
            for i, limit in enumerate(self.edge_limits):
                if i < original_graph.edge_attr.size(1):
                    original_graph.edge_attr[:, i] = torch.clamp(original_graph.edge_attr[:, i], min=0, max=limit-1)

        # On retourne le graphe (qui contient x, edge_index...) et le tenseur de texte
        return original_graph, caption_tensor

def custom_collate_fn(batch):
    """
    Assemble un batch de (graph, caption_tensor).
    """
    graphs = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    
    # 1. Batcher les graphes (PyG method)
    # Batch.from_data_list va fusionner tous les petits graphes en un grand graphe déconnecté
    batch_graphs = Batch.from_data_list(graphs)
    
    # 2. Batcher les textes (Padding)
    # On met 0 comme padding value (C'est généralement l'ID du [PAD] dans les tokenizers BPE/ByteLevel)
    batch_captions = pad_sequence(captions, batch_first=True, padding_value=0)
    
    return batch_graphs, batch_captions