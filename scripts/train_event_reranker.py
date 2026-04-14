"""
SIEVE Event Reranker Training Script
Run on Google Colab with GPU runtime.

Prerequisites:
1. Export training data: sieve export-training --output training.jsonl
2. Upload training.jsonl to Colab
3. Run this script
4. Download event_reranker.onnx
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "transformers",
        "onnx",
        "onnxruntime-gpu",
        "sentence-transformers",
    ]
)

# === Config ===
MAX_EVENT_TOKENS = 16
EVENT_FEATS = 12
GLOBAL_FEATS = 12
HIDDEN_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 64
BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
TRAINING_FILE = "training.jsonl"

# === Load cross-encoder teacher for distillation ===
from sentence_transformers import CrossEncoder
teacher = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", max_length=512)

# === Dataset ===
class EventDataset(Dataset):
    def __init__(self, path):
        self.examples = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        query = ex["query"]
        window = ex["window"]
        raw_text = window["raw_text"]
        formula_score = window["formula_score"]

        teacher_score = teacher.predict([(query, raw_text)])
        if isinstance(teacher_score, np.ndarray):
            teacher_score = float(teacher_score[0])
        else:
            teacher_score = float(teacher_score)

        events = window["events"][:MAX_EVENT_TOKENS]
        event_feats = np.zeros((MAX_EVENT_TOKENS, EVENT_FEATS), dtype=np.float32)
        mask = np.zeros(MAX_EVENT_TOKENS, dtype=np.float32)

        for i, evt in enumerate(events):
            event_feats[i] = [
                evt["relative_start"],
                evt["relative_end"],
                evt["relative_gap_prev"],
                evt["query_weight"],
                evt["idf"],
                evt["group_importance"],
                evt["normalized_ordinal"],
                float(evt["is_anchor"]),
                float(evt["is_phrase"]),
                float(evt["is_identifier_variant"]),
                float(evt["repeated_group"]),
                float(evt["same_group_as_prev"]),
            ]
            mask[i] = 1.0

        g = window["global_features"]
        global_feats = np.array(
            [
                g["formula_score"],
                g["matched_group_coverage"],
                g["anchor_count"] / float(MAX_EVENT_TOKENS),
                g["phrase_count"] / float(MAX_EVENT_TOKENS),
                g["top_proximity"],
                g["ordered_pair_mass"],
                g["common_term_penalty"],
                g["total_group_mass"],
                g["event_count"] / float(MAX_EVENT_TOKENS),
                g["query_group_count"] / 16.0,
                g["max_event_gain"],
                g["mean_idf"],
            ],
            dtype=np.float32,
        )

        return {
            "events": torch.tensor(event_feats),
            "mask": torch.tensor(mask),
            "global": torch.tensor(global_feats),
            "teacher_score": torch.tensor(teacher_score, dtype=torch.float32),
            "formula_score": torch.tensor(formula_score, dtype=torch.float32),
        }

# === Model ===
class TinyEventTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(EVENT_FEATS, HIDDEN_DIM)
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM + GLOBAL_FEATS, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, events, mask, global_feats):
        B = events.size(0)
        x = self.input_proj(events)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(B, 1, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        attn_mask = full_mask == 0
        x = self.encoder(x, src_key_padding_mask=attn_mask)
        cls_out = x[:, 0, :]
        combined = torch.cat([cls_out, global_feats], dim=1)
        score = self.head(combined).squeeze(-1)
        return score

# === Training ===
print("Loading training data...")
dataset = EventDataset(TRAINING_FILE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Loaded {len(dataset)} examples")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyEventTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
mse_loss = nn.MSELoss()

print(f"Training on {device}...")
for epoch in range(EPOCHS):
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        events = batch["events"].to(device)
        mask = batch["mask"].to(device)
        global_feats = batch["global"].to(device)
        teacher_scores = batch["teacher_score"].to(device)

        pred = model(events, mask, global_feats)
        loss = mse_loss(pred, teacher_scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    print(f"Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f}")

# === Export to ONNX ===
model.eval()
model.cpu()

dummy_events = torch.randn(1, MAX_EVENT_TOKENS, EVENT_FEATS)
dummy_mask = torch.ones(1, MAX_EVENT_TOKENS)
dummy_global = torch.randn(1, GLOBAL_FEATS)

torch.onnx.export(
    model,
    (dummy_events, dummy_mask, dummy_global),
    "event_reranker.onnx",
    input_names=["events", "mask", "global_features"],
    output_names=["score"],
    dynamic_axes={
        "events": {0: "batch"},
        "mask": {0: "batch"},
        "global_features": {0: "batch"},
        "score": {0: "batch"},
    },
    opset_version=14,
)

import onnxruntime as ort
sess = ort.InferenceSession("event_reranker.onnx")
out = sess.run(
    None,
    {
        "events": dummy_events.numpy(),
        "mask": dummy_mask.numpy(),
        "global_features": dummy_global.numpy(),
    },
)
print(f"Output shape: {out[0].shape}")
print(f"Model size: {Path('event_reranker.onnx').stat().st_size / 1024:.1f} KB")

from google.colab import files
files.download("event_reranker.onnx")
print("Done — download event_reranker.onnx and copy to ~/.sieve/models/event-reranker/")
