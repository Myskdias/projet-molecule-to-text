import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm

class EditModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        # Geler l’encodeur
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 3)  # KEEP / DELETE / REPLACE

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        token_embs = outputs.last_hidden_state  # [B, L, H]
        logits = self.classifier(token_embs)    # [B, L, 3]
        return logits


def compute_loss(logits, labels, attention_mask):
    """
    logits: [B, L, 3]
    labels: [B, L]
    attention_mask: [B, L]
    """

    # Poids des classes (clé du succès)
    class_weights = torch.tensor(
        [0.05, 1.0, 1.0], device=logits.device
    )

    loss_fct = nn.CrossEntropyLoss(
        weight=class_weights,
        reduction="none"
    )

    # [B, L]
    loss = loss_fct(
        logits.view(-1, 3),
        labels.view(-1)
    ).view(labels.size())

    # Ignorer padding
    loss = loss * attention_mask

    return loss.sum() / attention_mask.sum()


def train_editor(
    model,
    dataloader,
    optimizer,
    device,
    num_epochs=3,
):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["edit_labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = compute_loss(logits, labels, attention_mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

def get_edit_probs(model, input_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
    return probs  # [B, L, 3]
