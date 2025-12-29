
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from editor.dataset_builder import build_edit_text_pairs
from editor.diff_maker import EditDataset
from editor.edit_model import EditModel, train_editor
from utils.utils import build

TOP_K = 10
SAVE_PATH = "weights/editor_level1_head.pt"


def main():


    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    models = build()

    retrieved_texts, gold_texts = build_edit_text_pairs(
        ds_train=models["ds_train"],
        dl_train=models["dl_train"],
        id2desc_train=models["id2desc_train"],
        clip_model=models["clip_model"],
        retriever=models["retriever"],
        text_encoder=models["clip_model"].text_encoder,
        top_k=TOP_K,
    )
    dataset = EditDataset(
        retrieved_texts=retrieved_texts,
        gold_texts=gold_texts,
        tokenizer=tokenizer,
        max_length=128,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EditModel("sentence-transformers/all-MiniLM-L6-v2")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),  # seulement la tête
        lr=2e-4
    )
    train_editor(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=3,
    )
    
    torch.save(
        {
            "classifier": model.classifier.state_dict(),
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "labels": {
                "KEEP": 0,
                "DELETE": 1,
                "REPLACE": 2,
            },
        },
        SAVE_PATH,
    )

    print(f"[EDITOR] Head saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
    