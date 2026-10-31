"""
Train the Career Guidance Transformer model on our custom dataset.
Run: python train.py
"""

import json
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import CareerGuidanceTransformer
from vocabulary import Vocabulary, tokenize, MAX_LEN, PAD, BOS, EOS, UNK

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_PATH = "data/career_dataset.json"
MODEL_SAVE_PATH = "data/career_model.pth"
VOCAB_SAVE_PATH = "data/vocab.pkl"
BATCH_SIZE = 8
EPOCHS = 100
LR = 3e-4
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 3
DIM_FF = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────────────────────────
class CareerDataset(Dataset):
    def __init__(self, data, vocab, max_len=MAX_LEN):
        self.samples = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_ids = self.vocab.encode(self.samples[idx]["input"], self.max_len)
        tgt_ids = self.vocab.encode(self.samples[idx]["response"], self.max_len)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    return src_padded, tgt_padded


# ── Training ─────────────────────────────────────────────────────────────────
def generate_square_subsequent_mask(sz, device):
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()


def train():
    os.makedirs("data", exist_ok=True)

    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Train] Loaded {len(data)} samples.")

    # Build vocabulary (include topic field if present)
    vocab = Vocabulary()
    all_texts = []
    for d in data:
        all_texts.append(d["input"])
        all_texts.append(d["response"])
        if "topic" in d:
            all_texts.append(d["topic"])
    vocab.build(all_texts)

    # Save vocabulary
    with open(VOCAB_SAVE_PATH, "wb") as f:
        pickle.dump(vocab, f)
    print(f"[Train] Vocabulary saved to {VOCAB_SAVE_PATH}")

    # Dataset & loader
    dataset = CareerDataset(data, vocab)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Model
    model = CareerGuidanceTransformer(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)

            # Teacher forcing: tgt_input is tgt minus last token
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), DEVICE)
            src_pad_mask = (src == 0)
            tgt_pad_mask = (tgt_input == 0)

            logits = model(src, tgt_input,
                           src_key_padding_mask=src_pad_mask,
                           tgt_key_padding_mask=tgt_pad_mask,
                           tgt_mask=tgt_mask)

            # logits: (B, T, vocab) → (B*T, vocab)
            loss = criterion(logits.reshape(-1, len(vocab)), tgt_output.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{EPOCHS}] | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "vocab_size": len(vocab),
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "num_layers": NUM_LAYERS,
                "dim_ff": DIM_FF,
            }, MODEL_SAVE_PATH)

    print(f"\n[Train] Training complete. Best loss: {best_loss:.4f}")
    print(f"[Train] Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
