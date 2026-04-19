from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    for i in range(len(a) + 1):
        dp[i, 0] = i
    for j in range(len(b) + 1):
        dp[0, j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[-1, -1])


def word_error_rate(reference: str, hypothesis: str) -> float:
    r = reference.lower().split()
    h = hypothesis.lower().split()
    if len(r) == 0:
        return 0.0
    return _levenshtein(r, h) / len(r)


def mcd_db(ref_wav: str, syn_wav: str, sample_rate: int = 22050) -> float:
    ref, sr1 = torchaudio.load(ref_wav)
    syn, sr2 = torchaudio.load(syn_wav)

    if sr1 != sample_rate:
        ref = torchaudio.functional.resample(ref, sr1, sample_rate)
    if sr2 != sample_rate:
        syn = torchaudio.functional.resample(syn, sr2, sample_rate)

    mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=13)
    c1 = mfcc(ref.mean(dim=0, keepdim=True)).squeeze(0).transpose(0, 1)
    c2 = mfcc(syn.mean(dim=0, keepdim=True)).squeeze(0).transpose(0, 1)

    n = min(c1.size(0), c2.size(0))
    if n == 0:
        return float("inf")

    c1 = c1[:n]
    c2 = c2[:n]

    dist = torch.sqrt(((c1 - c2) ** 2).sum(dim=1) + 1e-8)
    scale = 10.0 / torch.log(torch.tensor(10.0)) * torch.sqrt(torch.tensor(2.0))
    return float((scale * dist).mean().item())


@dataclass
class SwitchPoint:
    timestamp_sec: float


def switching_precision_with_tolerance(
    predicted: Iterable[SwitchPoint],
    reference: Iterable[SwitchPoint],
    tolerance_ms: float = 200.0,
) -> float:
    pred = list(predicted)
    ref = list(reference)
    tol = tolerance_ms / 1000.0

    if not ref:
        return 1.0

    matched = 0
    used = set()
    for p in pred:
        best_idx = None
        best_delta = 1e9
        for i, r in enumerate(ref):
            if i in used:
                continue
            delta = abs(p.timestamp_sec - r.timestamp_sec)
            if delta < best_delta:
                best_delta = delta
                best_idx = i
        if best_idx is not None and best_delta <= tol:
            matched += 1
            used.add(best_idx)

    return matched / max(len(pred), 1)


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    thresholds = np.linspace(float(scores.min()), float(scores.max()), num=200)
    best_gap = 1e9
    best_eer = 1.0

    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        far = ((pred == 1) & (labels == 0)).sum() / max((labels == 0).sum(), 1)
        frr = ((pred == 0) & (labels == 1)).sum() / max((labels == 1).sum(), 1)
        gap = abs(far - frr)
        eer = 0.5 * (far + frr)
        if gap < best_gap:
            best_gap = gap
            best_eer = eer

    return float(best_eer)
