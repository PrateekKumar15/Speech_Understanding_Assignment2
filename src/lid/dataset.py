from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.utils import LANG_TO_ID


@dataclass
class LIDFeatureConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    frame_length_ms: int = 25
    frame_shift_ms: int = 10


class FrameLIDDataset(Dataset):
    def __init__(self, manifest_path: str | Path, cfg: LIDFeatureConfig):
        self.df = pd.read_csv(manifest_path)
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=int(cfg.frame_length_ms * cfg.sample_rate / 1000),
            hop_length=int(cfg.frame_shift_ms * cfg.sample_rate / 1000),
            n_mels=cfg.n_mels,
            power=2.0,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        label = row["lang"]

        waveform, sr = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.cfg.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.cfg.sample_rate)

        start = int(start_sec * self.cfg.sample_rate)
        end = int(end_sec * self.cfg.sample_rate)
        segment = waveform[:, start:end]

        feats = torch.log(self.mel(segment).squeeze(0).transpose(0, 1).clamp(min=1e-5))
        targets = torch.full((feats.size(0),), LANG_TO_ID[label], dtype=torch.long)
        return feats, targets


def lid_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    feats, targets = zip(*batch)
    feat_lengths = torch.tensor([x.size(0) for x in feats], dtype=torch.long)

    padded_feats = pad_sequence(feats, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-100)
    return padded_feats, padded_targets, feat_lengths
