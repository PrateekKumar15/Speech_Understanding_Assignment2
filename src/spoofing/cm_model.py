from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.spoofing.lfcc import extract_lfcc


class CMDataset(Dataset):
    def __init__(self, manifest_csv: str, sample_rate: int = 16000):
        import pandas as pd

        self.df = pd.read_csv(manifest_csv)
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        wav, sr = torchaudio.load(row["audio_path"])
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        feat = extract_lfcc(wav, sample_rate=self.sample_rate)
        label = 1 if row["label"].lower() == "spoof" else 0
        return feat, torch.tensor(label, dtype=torch.long)


def collate_cm(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    feats = pad_sequence(feats, batch_first=True)
    return feats, torch.stack(labels), lengths


class CountermeasureNet(nn.Module):
    def __init__(self, in_dim: int = 20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


@torch.no_grad()
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    thresholds = np.unique(scores)
    best = 1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        fa = ((pred == 1) & (labels == 0)).sum() / max((labels == 0).sum(), 1)
        miss = ((pred == 0) & (labels == 1)).sum() / max((labels == 1).sum(), 1)
        eer = 0.5 * (fa + miss)
        best = min(best, abs(fa - miss) + eer)
    return float(best)


def train_cm(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    train_ds = CMDataset(args.train_manifest, sample_rate=args.sample_rate)
    val_ds = CMDataset(args.val_manifest, sample_rate=args.sample_rate)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_cm)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_cm)

    model = CountermeasureNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_eer = float("inf")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for feats, labels, _ in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss = F.cross_entropy(logits, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        scores, gt = [], []
        for feats, labels, _ in val_loader:
            feats = feats.to(device)
            logits = model(feats)
            prob_spoof = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            scores.extend(prob_spoof.tolist())
            gt.extend(labels.numpy().tolist())

        eer = compute_eer(np.array(scores), np.array(gt))
        print(f"epoch={epoch} val_eer={eer:.4f}")

        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), out_path)
            print(f"saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anti-spoofing CM on LFCC features.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output", default="models/cm_best.pt")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    train_cm(parse_args())
