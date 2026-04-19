from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio


@dataclass
class ProsodyFeatures:
    f0: torch.Tensor
    energy: torch.Tensor


def extract_prosody(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_time: float = 0.01,
    win_length: int = 1024,
) -> ProsodyFeatures:
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)

    hop = max(1, int(sample_rate * frame_time))
    f0 = torchaudio.functional.detect_pitch_frequency(
        waveform.unsqueeze(0),
        sample_rate=sample_rate,
        frame_time=frame_time,
        win_length=win_length,
    ).squeeze(0)

    frames = waveform.unfold(0, win_length, hop)
    energy = torch.sqrt((frames * frames).mean(dim=1) + 1e-8)

    min_len = min(f0.numel(), energy.numel())
    return ProsodyFeatures(f0=f0[:min_len], energy=energy[:min_len])


def dtw_path(x: torch.Tensor, y: torch.Tensor) -> List[Tuple[int, int]]:
    n, m = x.numel(), y.numel()
    cost = torch.full((n + 1, m + 1), float("inf"))
    cost[0, 0] = 0.0

    dist = torch.cdist(x.view(-1, 1), y.view(-1, 1), p=1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = dist[i - 1, j - 1]
            cost[i, j] = c + torch.min(torch.stack([cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]]))

    i, j = n, m
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        prev = torch.argmin(torch.tensor([cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]])).item()
        if prev == 0:
            i -= 1
        elif prev == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    path.reverse()
    return path


def warp_contour(source: torch.Tensor, target_reference: torch.Tensor) -> torch.Tensor:
    path = dtw_path(source, target_reference)
    warped = torch.zeros_like(source)

    map_dict: dict[int, list[float]] = {}
    for i, j in path:
        map_dict.setdefault(i, []).append(float(target_reference[j]))

    for i in range(source.numel()):
        if i in map_dict:
            warped[i] = torch.tensor(map_dict[i]).mean()
        else:
            warped[i] = source[i]
    return warped
