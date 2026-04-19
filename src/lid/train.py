from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.lid.dataset import FrameLIDDataset, LIDFeatureConfig, lid_collate_fn
from src.lid.model import FrameLIDNet
from src.utils import resolve_device, set_seed


@torch.no_grad()
def frame_f1_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    mask = targets != -100

    preds = preds[mask]
    targets = targets[mask]

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision_hi = tp / max(tp + fp, 1)
    recall_hi = tp / max(tp + fn, 1)
    f1_hi = 2 * precision_hi * recall_hi / max(precision_hi + recall_hi, 1e-8)

    tp_en = ((preds == 0) & (targets == 0)).sum().item()
    fp_en = ((preds == 0) & (targets == 1)).sum().item()
    fn_en = ((preds == 1) & (targets == 0)).sum().item()

    precision_en = tp_en / max(tp_en + fp_en, 1)
    recall_en = tp_en / max(tp_en + fn_en, 1)
    f1_en = 2 * precision_en * recall_en / max(precision_en + recall_en, 1e-8)

    return 0.5 * (f1_hi + f1_en)


@torch.no_grad()
def evaluate(model: FrameLIDNet, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_f1 = 0.0
    n = 0

    for feats, targets, _ in loader:
        feats = feats.to(device)
        targets = targets.to(device)

        logits = model(feats)
        loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100)

        total_loss += float(loss.item())
        total_f1 += frame_f1_from_logits(logits, targets)
        n += 1

    return {
        "loss": total_loss / max(n, 1),
        "f1": total_f1 / max(n, 1),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)

    cfg = LIDFeatureConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        frame_length_ms=args.frame_length_ms,
        frame_shift_ms=args.frame_shift_ms,
    )

    train_ds = FrameLIDDataset(args.train_manifest, cfg)
    val_ds = FrameLIDDataset(args.val_manifest, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lid_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lid_collate_fn,
    )

    model = FrameLIDNet(
        input_dim=args.n_mels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_f1 = -1.0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch={epoch}")

        for feats, targets, _ in pbar:
            feats = feats.to(device)
            targets = targets.to(device)

            logits = model(feats)
            loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        metrics = evaluate(model, val_loader, device)
        print(f"[val] epoch={epoch} loss={metrics['loss']:.4f} f1={metrics['f1']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"saved best checkpoint to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frame-level Hinglish LID.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output", default="models/lid_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--frame-length-ms", type=int, default=25)
    parser.add_argument("--frame-shift-ms", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
