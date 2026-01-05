import os
import csv
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Chemins
TRAIN_EMB_CSV = "./descriptions_output.csv" # Contient les ID et probablement les textes (à vérifier)
# Si les textes sont ailleurs (ex: un fichier txt), change ce chemin.
# Pour l'exemple, je suppose qu'on a un fichier texte brut ou qu'on peut l'extraire.

def train_bpe_tokenizer(vocab_size=10000, output_file="vocab.json", ):
    print("--- ENTRAÎNEMENT DU TOKENIZER BPE ---")
    

    captions = [] 

    with open(TRAIN_EMB_CSV, mode='r', encoding='utf-8') as fichier:
        lecteur = csv.reader(fichier)
        
        for ligne in lecteur:
            print(ligne)

            if ligne: 
                captions.append(ligne[0])


    if len(captions) == 0:
        print("⚠️ ATTENTION : Liste de captions vide. Tu dois charger tes textes ici !")
        # Simulation pour l'exemple
        captions = ["The molecule is a monocarboxylic acid.", "It acts as a inhibitor."]
    
    # 2. Initialisation du Tokenizer BPE
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # 3. Pré-tokenization (découpage par espace et ponctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # 4. Configuration de l'entraîneur
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[PAD]", "[START]", "[END]", "[UNK]"],
        min_frequency=2 # Un mot doit apparaître au moins 2 fois
    )
    
    # 5. Entraînement
    tokenizer.train_from_iterator(captions, trainer=trainer)
    
    # 6. Post-processing (Ajout des tokens spéciaux automatiquement)
    # On veut que "molécule" devienne "[START] molécule [END]"
    start_id = tokenizer.token_to_id("[START]")
    end_id = tokenizer.token_to_id("[END]")
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[START] $A [END]",
        special_tokens=[
            ("[START]", start_id),
            ("[END]", end_id),
        ],
    )
    
    # 7. Décodeur (pour revenir au texte)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 8. Sauvegarde
    tokenizer.save(output_file)
    print(f"Tokenizer BPE sauvegardé dans {output_file}. Vocabulaire: {tokenizer.get_vocab_size()}")
    
    return tokenizer

if __name__ == "__main__":
    # Ajuste la taille selon tes besoins. 
    # Pour ce challenge, 3000-5000 est souvent suffisant et évite l'overfitting.
    train_bpe_tokenizer(vocab_size=4000, output_file="bpe_tokenizer.json")